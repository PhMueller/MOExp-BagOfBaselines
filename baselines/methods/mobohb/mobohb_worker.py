from baselines.methods.mobohb.hpbandster.core.worker import Worker
import ConfigSpace as CS
import numpy as np


class MOBOHBWorker(Worker):
    def __init__(self, experiment, fidelity: str, search_space, seed=0, **kwargs):
        super().__init__(**kwargs)

        self.experiment = experiment
        self.fidelity = fidelity
        self.search_space = search_space
        self.seed = seed

    def tchebycheff_norm(self, cost, rho=0.05):
        # Sample weights for all objectives.
        w = np.random.random_sample(2)
        w /= np.sum(w)

        w_f = w * cost
        max_k = np.max(w_f)
        rho_sum_wf = rho * np.sum(w_f)
        return max_k + rho_sum_wf

    def compute(self, config_id:int, config: CS.Configuration, budget:float, working_directory:str, *args, **kwargs) -> dict:

        fidelity = {self.fidelity['name']: budget}

        # Map numeric (cat.) hp back to their categories
        for hp, value in config.items():
            orig_hp = self.experiment.cs_search_space.get_hyperparameter(hp)
            if isinstance(orig_hp, CS.CategoricalHyperparameter):
                config[hp] = orig_hp.choices[value]

        data, metric_names = self.experiment.eval_configuration(configuration=config, fidelity=fidelity)
        result = tuple([float(data.df[data.df['metric_name'] == obj]['mean']) for obj in metric_names])
        return {'loss': result}
