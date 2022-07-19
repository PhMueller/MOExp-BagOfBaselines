import numpy as np
import torch
from torch.utils.data import DataLoader
from MOHPOBenchExperimentUtils import nDS_index, crowdingDist
import torch.nn.functional as F
from loguru import logger


def sort_array(fit):
    index_list = np.array(list(range(len(fit))))

    a, index_return_list = nDS_index(np.array(fit), index_list)
    b, sort_index = crowdingDist(a, index_return_list)

    sorted_ = []
    for i, x in enumerate(sort_index):
        sorted_.extend(x)

    sorted_ = [sorted_.index(i) for i in range((len(fit)))]

    return sorted_


class Net(torch.nn.Module):

    def __init__(self, num_input_parameters, num_objectives=2):
        super(Net, self).__init__()

        self.fc2 = torch.nn.Linear(num_input_parameters, 10)
        torch.nn.init.normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc2.bias)

        self.fc3 = torch.nn.Linear(10, num_objectives)
        torch.nn.init.normal_(self.fc3.weight)
        torch.nn.init.normal_(self.fc3.bias)

    def forward(self, x):

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

    def train_fn(self, train_data, num_epochs):
        """
        Training method
        :param optimizer: optimization algorithm
        """
        self.train()
        batch_size = 32

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            i = 0
            for data in train_loader:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                self.to(device)

                inputs = torch.stack(data[0])
                inputs = torch.transpose(inputs, 0, 1)
                inputs = inputs.double().to(device)

                # get the inputs; data is a list of [inputs, labels]
                y_value = torch.stack(data[1])
                y_value = torch.transpose(y_value, 0, 1)
                y_value = y_value.type(torch.DoubleTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                self.double()
                outputs = self(inputs)
                loss = loss_criterion(outputs, y_value)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.debug('[%d] loss: %.10f' % (epoch + 1,  running_loss / len(train_data)))

        return

    def predict(self, x):

        self.eval()
        self.double()

        x = [float(m) for m in x]

        train_loader = DataLoader(dataset=[x],
                                  shuffle=True)

        for d in train_loader:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            inputs = torch.stack(d).to(device)
            inputs = torch.transpose(inputs, 0, 1)

            output = self(inputs)
        return output


class Neural_Predictor:
    """
    Class to group ensamble of NN
    """

    def __init__(self, num_input_parameters, num_objectives, num_epochs, num_ensemble_nets):
        self.num_epochs = num_epochs
        self.num_ensemble_nets = num_ensemble_nets
        self.networks = [Net(num_input_parameters=num_input_parameters, num_objectives=num_objectives)
                         for _ in range(self.num_ensemble_nets)]
        self.all_architecture = []

    def train_models(self, x):
        for model in self.networks:
            model.train_fn(x, self.num_epochs)

    def ensamble_predict(self, x):
        # We do the scaling inside the MOExpUtils
        predictions = [model.predict(x).tolist()[0] for model in self.networks]

        # predictions = [[- pred[0] * 10, pred[1]] for pred in predictions]
        # TODO: Scale to multiple objectives
        predictions = [[pred[0], pred[1]] for pred in predictions]
        mean1 = np.mean([pred[0] for pred in predictions])
        mean2 = np.mean([pred[1] for pred in predictions])
        return [mean1, mean2], predictions

    def independent_thompson_sampling_for_mo(self, x, arches_in, num_models):
        arches = arches_in.copy()
        mean_list = []
        prediction_list = [[] for _ in range(num_models)]

        # for the possible new configurations, ask the ensemble about the potential performance.
        for i_arch in range(len(arches_in)):
            # mean = mean pred. perf of param 1, mean predicted performance for param 2
            # mean = 1 x N_objectives | predictions: n_models x n_objectives
            mean, predictions = self.ensamble_predict(x[i_arch])
            mean_list.append(mean)

            # Collect for each model the predicted performances. [pred model 1, pred model 2, pred model 3, ...]
            for i in range(num_models):
                prediction_list[i].extend([predictions[i]])

        # mean_list: Num Configs X Num Objectives -> For each config the mean prediction of the neural predictors
        mean_ordering = sort_array(mean_list)  # sort means wrt crowding dist + nds. Return indices

        predictions_ordering_per_model = []
        for i in range(num_models):  # Does this make sense? Order the configurations of each ensemble member? potentially different orderings.
            predictions_ordering_per_model.append(sort_array(prediction_list[i]))  # sort predictions of a model wrt to nds + crowding distance.

        prob_ = []
        for i in range(len(arches_in)):
            prob1 = self.independent_thompson_sampling(mean_ordering[i], [order[i] for order in predictions_ordering_per_model])
            prob_.append(prob1)

        return prob_

    def sort_pop(self, list1, list2):
        z = []
        for m in list2:
            z.append(list1[int(m)])
        return z

    def independent_thompson_sampling(self, mean, predictions_fixed):
        mean = np.array(mean)
        predictions_fixed = np.array(predictions_fixed)

        M = self.num_ensemble_nets
        squared_differences = np.sum(np.square(predictions_fixed - mean))
        var = np.sqrt(squared_differences / (M - 1))
        prob = np.random.normal(mean, var)

        return prob

    def choose_models(self, architectures, test_data, select_models):

        architectures = architectures.copy()

        arch_lists = []
        probs = self.independent_thompson_sampling_for_mo(test_data, architectures, self.num_ensemble_nets)

        for _ in range(select_models):
            max_index = probs.index(min(probs))
            arch_lists.append(architectures[max_index])
            probs.pop(max_index)
            architectures.pop(max_index)

        return arch_lists
