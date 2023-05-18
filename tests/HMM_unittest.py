import unittest
from HMM.HMM import *
import pandas as pd
import numpy as np
import seaborn

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

    def _init_NormalHMM(self):
        self.hmm = NormalHMM(initial_state_matrix=self.initial_state_matrix,
                        transition_matrix=self.transition_matrix,
                        medias=[0,0.2],
                        std=[1,2],
                        observations=self.observations)
        """
        hmm2 = tfp.distributions.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(probs=self.initial_state_probs),
            transition_distribution=tfd.Categorical(probs=self.transition_probs),
            observation_distribution=observation_distribution,
            num_steps=len(self.observations),
            validate_args=True
        )
        # hmm2.posterior_mode(self.observations.astype(np.float32)).numpy()
        hmm2.log_prob(self.observations)
        """

    def test_init_aic(self):
        self.hmm = NormalHMM(observations=self.observations)


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

    def test_init_with_observations(self):
        hmm = NormalHMM(observations=self.observations)
    def test_train(self):
        self._init_NormalHMM()
        self.hmm.train(self.observations,0.8,labels=self.labels)

    def test_label_train(self):
        self._init_NormalHMM()
        self.hmm.train(self.observations,train_size=0.8,labels=self.labels,
                       algorithm = TrainingAlgorithm.label)

    def test_quantiles(self):
        self._init_NormalHMM()
        self.hmm.compute_quantiles(self.observations)
    def test_forecast(self):
        self._init_NormalHMM()
        self.hmm.forecast(3)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.observations)
        ax.show()




if __name__ == '__main__':
    unittest.main()
