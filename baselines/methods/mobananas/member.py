import enum
import uuid
from copy import deepcopy

import ConfigSpace as CS
from ConfigSpace.hyperparameters import NumericalHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter

import numpy as np
from typing import Dict, Optional, Callable, Union
from ax import Experiment, GeneratorRun, Arm
from scipy.stats import truncnorm
from MOHPOBenchExperimentUtils.utils import adapt_configspace_configuration_to_ax_space

from loguru import logger


def normalize_parameters(params: Dict, configuration_space: CS.ConfigurationSpace, fidelity_name) -> Dict:
    normlized_parameters = {}
    for key in params.keys():
        if key in ['id', fidelity_name]:
            continue

        hp = configuration_space.get_hyperparameter(key)

        if isinstance(hp, CS.Constant):
            # Remove the constant parameters from the configuration: (do not add to train_data)
            continue

        # They apply a Min Max Scaling to the values.
        elif isinstance(params[key], bool):
            param = 1 if params[key] else 0

        elif isinstance(hp, NumericalHyperparameter):
            lower_lim, upper_lim = hp.lower, hp.upper
            param = (params[key] - lower_lim) / (upper_lim - lower_lim)

        # TODO: write stuff for non numerical data.
        elif isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
            if isinstance(hp.choices[0], str):
                param = hp.choices.index(params[key]) / len(hp.choices)
            else:
               choices = np.sort(hp.choices)
               lower_lim, upper_lim = choices[0], choices[-1]
               param = (params[key] - lower_lim) / (upper_lim - lower_lim)

        else:
            raise ValueError(f'Unsupported Parameter type: {key}: {type(params[key])}')

        normlized_parameters[key] = param
    return normlized_parameters


def inverse_normalize_configuration(normalized_params: Dict, configuration_space: CS.ConfigurationSpace) -> Dict:
    orig_parameters = {}
    for key in normalized_params.keys():

        hp = configuration_space.get_hyperparameter(key)
        param = normalized_params[param]
        # They apply a Min Max Scaling to the values.
        # if isinstance(params[key], bool):
        #     param = 1 if params[key] else 0

        if isinstance(hp, NumericalHyperparameter):
            lower_lim, upper_lim = hp.lower, hp.upper
            # param = (params[key] - lower_lim) / (upper_lim - lower_lim)
            param = (param * (upper_lim - lower_lim)) + lower_lim

        # TODO: write stuff for non numerical data.
        elif isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
            if isinstance(hp.choices[0], bool):
                param = param == 1

            elif isinstance(hp.choices[0], 'str'):
                index = param * len(hp.choices)
                index = np.rint(index).clip(min=0, max=len(hp.choices) - 1)
                param = hp.choices[index]
                # param = hp.choices.index(params[key]) / len(hp.choices)

            else:
                choices = np.sort(hp.choices)
                lower_lim, upper_lim = choices[0], choices[-1]
                param = (param * (upper_lim - lower_lim)) + lower_lim
                # param = (params[key] - lower_lim) / (upper_lim - lower_lim)

        else:
            raise ValueError(f'Unsupported Parameter type: {key}: {type(param)}')

        orig_parameters[key] = param
    return orig_parameters


class Mutation(enum.IntEnum):
    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation


class Member:
    """
    Class to handle member.
    """

    def __init__(self,
                 search_space: CS.ConfigurationSpace,
                 mutation: Mutation,
                 budget: Dict,
                 experiment: Experiment = None,
                 x_coordinate: Optional[Dict] = None) -> None:
        """
        Init
        :param search_space: search_space of the given problem
        :param x_coordinate: Initial coordinate of the member
        :param target_function: The target function that determines the fitness value
        :param mutation: hyperparameter that determines which mutation type use
        :budget number of epochs
        :param experiment: axi experiment to run
        """
        self._space = search_space
        self._budget = budget
        self._id = uuid.uuid4()
        self._x = search_space.sample_configuration().get_dictionary() if not x_coordinate else x_coordinate
        self._age = 0
        self._mutation = mutation
        self._x_changed = True
        self._fit = None
        self._experiment = experiment
        self._num_evals = 0

        # The authors only consider minimization problems. We have also max problems.
        self.min_objectives = [target.lower_is_better for target in self._experiment.optimization_config.objective.metrics]

    @property
    def fitness(self):
        if self._x_changed:  # Only if budget or architecture has changed we need to evaluate the fitness.
            self._x_changed = False

            params = deepcopy(self._x)
            data, metric_names = self._experiment.eval_configuration(
                configuration=params, fidelity={self._budget['name']: self._budget['limits'][1]}
            )

            result_list = [float(data.df[data.df['metric_name'] == obj]['mean']) for obj in metric_names]
            result_list = [obj if min_prob else -1 * obj for obj, min_prob in zip(result_list, self.min_objectives)]
            logger.debug(f'Returned objectives: {metric_names} - {result_list}')
            self._fit = result_list

        return self._fit  # evaluate or return save variable

    @property
    def x_coordinate(self):
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value):
        self._x_changed = True
        self._x = value

    @property
    def budget(self):
        return self._budget['limits'][1]

    @budget.setter
    def budget(self, value):
        self._x_changed = True
        self._budget['limits'][1] = value

    @property
    def id(self):
        return self._id

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    @staticmethod
    def __fill_missing_hp(cs, configuration) -> Dict:
        for key in cs.get_hyperparameter_names():
            if key not in configuration:
                missing_hp = cs.get_hyperparameter(key)
                configuration[key] = missing_hp.default_value
        return configuration

    def return_train_data(self):
        # TODO: Explain what happens here.
        #       The parameters are scaled?
        #       Binary Values are set to 0/1
        #       Only numerical values work?
        params = deepcopy(self.x_coordinate)

        params = self.__fill_missing_hp(self._space, params)

        # TODO: They apply here actually only a max scaling. hp = hp / max_value(hp)
        normalized_params = normalize_parameters(
            params=params, configuration_space=self._space, fidelity_name=self._budget['name']
        )
        normalized_params = [normalized_params[hp_name] for hp_name in self._space.get_hyperparameter_names()]
        return normalized_params

    def mutate(self):
        """
        Mutation to create a new offspring
        :return: new member who is based on this member
        """
        new_x = self.x_coordinate.copy()
        new_x = {k: v for k, v in new_x.items() if not isinstance(self._space.get_hyperparameter(k), CS.Constant)}

        if self._mutation == Mutation.GAUSSIAN:
            immutable_parameters = list(set([cond.parent.name for cond in self._space.get_conditions()]))
            keys = np.random.choice(list(new_x.keys()), min(len(new_x), 3), replace=False)

            for k in keys:
                # Immutable parameters are parameters that are parents in a condition
                hp = self._space.get_hyperparameter(str(k))

                if hp.name not in immutable_parameters:
                    try:
                        mean = new_x[hp.name]
                        upper = hp.upper
                        lower = hp.lower
                        sd = (upper - lower) / 3
                        X = self.get_truncated_normal(mean=mean, sd=sd, low=lower, upp=upper)
                        if isinstance(hp, CS.UniformIntegerHyperparameter):
                            new_x[hp.name] = int(X.rvs())
                        else:
                            new_x[hp.name] = X.rvs()
                    except:
                        new_x[hp.name] = hp.sample(self._space.random)

        elif self._mutation != Mutation.NONE:
            # We won't consider any other mutation types
            raise NotImplementedError

        child = Member(self._space, self._mutation, self._budget, self._experiment, new_x)

        self._age += 1
        return child
