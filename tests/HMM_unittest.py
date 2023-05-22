import unittest

import matplotlib.pyplot as plt

from HMM.HMM import *
import pandas as pd
import numpy as np
import seaborn
import pymc
import arviz as az

seaborn.set_style('whitegrid')
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy.testing as nptest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.initial_state_matrix = np.array([0.3, 0.7])
        self.transition_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
        self.distributions = [tfd.Normal(loc=0, scale=1),
                              tfd.Normal(loc=0.2, scale=2)]

        # Leemos los datos:
        data = pd.read_excel("../../GDP.xlsx");  # Cambiar el directorio
        data["Date"] = [int(tiempo) + (tiempo - int(tiempo)) * 2.5 for tiempo in data["Date"]]
        # data["GDP_LCHANGE"] = data["GDP"].diff()
        data.loc[0, "Change"] = 0
        self.observations = data["Change"].to_numpy()
        self.labels = data["NBER"].to_numpy()

    def _init_NormalHMM(self,scaling_algoritm =ScalingAlgorithm.division):
        self.hmm = NormalHMM(initial_state_matrix=self.initial_state_matrix,
                        transition_matrix=self.transition_matrix,
                        medias=[0,0.2],
                        std=[1,2],
                        observations=self.observations,
                        scaling_algorithm=scaling_algoritm)

    """
    def test_init_aic(self):
        self.hmm = NormalHMM(observations=self.observations)
    """


    def test_log_lik(self):
        self._init_NormalHMM()
        log_lik = self.hmm.log_likelihood(self.observations)
        self.hmm.scaling_algorithm = ScalingAlgorithm.division
        log_lik2 = self.hmm.log_likelihood(self.observations)
        print("log-lik value:", log_lik)
        self.assertAlmostEqual(log_lik, log_lik2,2)

    def test_init_with_n_states(self):
        hmm = NormalHMM(n_states=3,observations=[-1,1])
        self.assertTrue(np.all(hmm.initial_state_matrix==1/3))
        self.assertTrue(np.all(hmm.transition_matrix==1/3))
        self.assertTrue(np.array_equal([dist.loc for dist in hmm.distributions],[-1,0,1]))

    def test_label_train(self):
        self._init_NormalHMM()
        log_lik,f1 = self.hmm.train(self.observations,0.7,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.label,
                       beta=1 )
        _,f2 = self.hmm.train(self.observations,0.7,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.label,
                       beta=2)
        print(self.hmm.initial_state_matrix)
        print(self.hmm.transition_matrix)
        print([media.loc for media in self.hmm.distributions])
        print([media.scale for media in self.hmm.distributions])
        print(f1,f2)
        print(log_lik)

    def test_train(self):
        self._init_NormalHMM(scaling_algoritm=ScalingAlgorithm.logarithm)
        log_lik,f1 = self.hmm.train(self.observations,0.7,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.baum_welch,
                       beta=1 )
        _,f2 = self.hmm.train(self.observations,0.7,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.baum_welch,
                       beta=2)
        print(self.hmm.initial_state_matrix)
        print(self.hmm.transition_matrix)
        print([media.loc for media in self.hmm.distributions])
        print([media.scale for media in self.hmm.distributions])
        print(f1,f2)
        print("log_lik:", log_lik)
        self._init_NormalHMM(scaling_algoritm=ScalingAlgorithm.division)
        log_lik,f1 = self.hmm.train(self.observations,0.7,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.baum_welch,
                       beta=1 )
        _,f2 = self.hmm.train(self.observations,0.7,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.baum_welch,
                       beta=2)
        print(self.hmm.initial_state_matrix)
        print(self.hmm.transition_matrix)
        print([media.loc for media in self.hmm.distributions])
        print([media.scale for media in self.hmm.distributions])
        print(f1,f2)
        print("log_lik:", log_lik)

    def test_viterbi(self):
        self._init_NormalHMM()
        log_lik,f1 = self.hmm.train(self.observations,1,
                       labels=self.labels,
                       algorithm=TrainingAlgorithm.baum_welch,
                       beta=1 )
        vit = self.hmm.viterbi(self.observations,True)
        print(vit)

    def test_series_prediction(self):
        self._init_NormalHMM()
        dists = self.hmm.conditional_dist(self.observations)
        means = [dist.mean() for dist in dists]
        plt.figure(figsize=(10, 5))
        n_obs = len(self.observations)
        plt.rcParams.update({'font.size': 18})
        plt.xlabel("Tiempo")
        plt.ylabel("Variación del PIB")
        plt.plot([i for i in range(n_obs)],
                 self.observations)
        plt.plot([i for i in range(n_obs)],
                 means,
                 color="red")
        intervals = np.array([az.hdi(np.sort(dist.sample(10000)), hdi_prob=0.95) for dist in dists])
        plt.fill_between([i for i in range(n_obs)],
                         intervals[:, 0],
                         intervals[:, 1],
                         color="red", alpha=0.3
                         )
        plt.show()


    def test_forecast(self):
        self._init_NormalHMM()
        time_steps = 20
        dists = self.hmm.forecast_dist(time_steps)
        means = [dist.mean() for dist in dists]
        plt.figure(figsize=(10,5))
        n_obs = len(self.observations)
        plt.rcParams.update({'font.size': 18})
        plt.xlabel("Tiempo")
        plt.ylabel("Variación del PIB")
        plt.plot([i for i in range(n_obs)],
                 self.observations)
        plt.plot([i for i in range(n_obs,n_obs+time_steps)],
                 means,
                 color="red")
        intervals = np.array([az.hdi(np.sort(dist.sample(10000)),hdi_prob=0.95) for dist in dists])
        plt.fill_between([i for i in range(n_obs,n_obs+time_steps)],
                         intervals[:,0],
                         intervals[:,1],
                         color="red", alpha=0.3
                         )
        plt.show()
    def test_forecast_states(self):
        self._init_NormalHMM()
        states = self.hmm.forecast_states(3)
        print(states)




if __name__ == '__main__':
    unittest.main()
