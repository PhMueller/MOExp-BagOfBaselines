import numpy as np
from MOHPOBenchExperimentUtils import nDS_index, crowdingDist

from baselines.methods.mobananas.neural_predictor import Neural_Predictor
from baselines.methods.mobananas.member import Mutation, Member


class BANANAS:
    """
    Class to group ensamble of NN
    """

    def __init__(self, neural_predictor, experiment, search_space,
                 initial_samples, num_arch, budget, select_models, function_evaluations,
                 mutation_type=Mutation.GAUSSIAN, seed=0):

        self.num_arch = num_arch
        self.num_function_evaluations = function_evaluations
        self.search_space = search_space
        self.experiment = experiment
        self.initial_samples = initial_samples
        self.neural_predictor = neural_predictor
        self.fidelity = budget  # dict with {'name': <NAME>, 'limits': [<LOWER>, <UPPER>]
        self.budget = budget['limits'][1]

        np.random.seed(seed)
        self.select = select_models

        self.architecture_list = [
            Member(search_space=self.search_space, mutation=mutation_type, budget=self.fidelity,
                   experiment=self.experiment)
            for _ in range(self.initial_samples)
        ]

        # [Member.fitness for Member in self.architecture_list]
        self.iterations = (self.num_function_evaluations - self.initial_samples) // self.select

    def steps(self):

        it = 0

        while it < self.iterations:

            it += 1

            train_data = [member.return_train_data() for member in self.architecture_list]
            y_train_data = [member.fitness for member in self.architecture_list]

            train_data = [[train_data[i], y_train_data[i]] for i in range(len(train_data))]
            # train_data = [[train_data[i], [-y_train_data[i][0]/10, y_train_data[i][1]]] for i in range(len(train_data))]

            self.neural_predictor.train_models(train_data)

            # choose best configs
            best_configs = self._select_best_architectures_mo(self.num_arch)
            mutated_configs = [member.mutate() for member in best_configs]
            test_data = [member.return_train_data() for member in mutated_configs]
            chosen_models = self.neural_predictor.choose_models(mutated_configs, test_data, self.select)
            [member.fitness for member in chosen_models]

            self.architecture_list.extend(chosen_models)

        return

    def sort_pop(self, list1, list2):

        z = [list1[int(m)] for m in list2]
        return z

    def _select_best_architectures_mo(self, num_arch):

        index_list = np.array(list(range(len(self.architecture_list))))
        fitness = [member.fitness for member in self.architecture_list]
        a, index_return_list = nDS_index(np.array(fitness), index_list)
        b, sort_index = crowdingDist(a, index_return_list)

        sorted = []
        for x in sort_index:
            sorted.extend(x)

        self.architecture_list = self.sort_pop(self.architecture_list, sorted)
        return self.architecture_list[0:num_arch]


def get_MOBANANAS(experiment, search_space, evaluate_network, budget = 25, initial_samples = 20,
                  num_arch = 8, select_models = 4,function_evaluations = 100):

    # save models and dict so it can be picked up later on
    neural_predictor = Neural_Predictor(num_epochs = 80, num_ensamble_nets = 5)
    banana = BANANAS(neural_predictor, evaluate_network, experiment, search_space,
                     initial_samples, num_arch, budget, select_models, function_evaluations)
    banana.steps()
    return


