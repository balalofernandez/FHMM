import unittest
from HMM import GHMM
import pandas as pd
import numpy as np
import seaborn
seaborn.set_style('whitegrid')
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy.testing as nptest
class MyTestCase(unittest.TestCase):
    def _create_HMM(self,initial_state_matrix, transition_matrix, distributions,n_obs):
        initial_state_probs = tf.Variable(initial_state_matrix, dtype=tf.float32)
        # Suma 1
        initial_state_probs = initial_state_probs / tf.reduce_sum(initial_state_probs)

        # Creamos la matriz de transición
        transition_probs = tf.Variable(transition_matrix, dtype=tf.float32)
        # Suma 1
        transition_probs = transition_probs / tf.reduce_sum(transition_probs, axis=1, keepdims=True)

        starting_loc = tf.Variable([distribution.loc for distribution in distributions], shape=(2,), dtype=tf.float32,
                                   name="medias")
        starting_scale = tf.Variable([distribution.scale for distribution in distributions], shape=(2,),
                                     dtype=tf.float32,
                                     name="varianzas")
        observation_distribution = tfd.Normal(loc=starting_loc, scale=starting_scale)
        # HiddenMarkovModel
        hmm = tfp.distributions.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(probs=initial_state_probs),
            transition_distribution=tfd.Categorical(probs=transition_probs),
            observation_distribution=observation_distribution,
            num_steps=n_obs,
            validate_args=True
        )
        return hmm

    def _init_ghmm(self):
        self.ghmm = GHMM(self.initial_state_matrix,
                            self.transition_matrix,
                            self.distributions,
                            self.observations)
    def setUp(self):
        self.initial_state_matrix = np.array([0.3,0.7])
        self.transition_matrix = np.array([[0.8,0.2],[0.2,0.8]])
        self.distributions = [tfd.Normal(loc=0, scale=1),
                  tfd.Normal(loc=0.2, scale=2)]

        #Leemos los datos:
        data = pd.read_excel("../../GDP.xlsx"); #Cambiar el directorio
        data["Date"] = [int(tiempo) + (tiempo - int(tiempo)) * 2.5 for tiempo in data["Date"]]
        # data["GDP_LCHANGE"] = data["GDP"].diff()
        data["Change"][0] = 0
        self.observations = data["Change"]
    def test_createClass(self):
        self.ghmm = GHMM(self.initial_state_matrix,
                            self.transition_matrix,
                            self.distributions,
                            self.observations)

    def test_forwardProbs(self):
        self._init_ghmm()
        alpha, scale, lik = self.ghmm.forward_probs()
        self.assertGreaterEqual(lik, -410)
    def test_backwardProbs(self):
        self._init_ghmm()
        alpha, scale, lik = self.ghmm.forward_probs()
        beta, start_state_val = self.ghmm.backward_probs(scale)
    def test_allProbs(self):
        self._init_ghmm()
        alpha, scale, lik = self.ghmm.forward_probs()
        beta, start_state_val = self.ghmm.backward_probs(scale)
        xi = self.ghmm.xi_probs(alpha,beta)
        self.ghmm.gamma_probs(xi)

    def test_baum_welch_normal(self):
        self._init_ghmm()
        self.ghmm.baum_welch_normal(verbose=True)

    def test_viterbi(self):
        self._init_ghmm()
        _,vit = self.ghmm.viterbi(self.observations)

    def test_log_baum_welch(self):
        self._init_ghmm()
        self.ghmm.log_baum_welch_normal(verbose=True)









if __name__ == '__main__':
    unittest.main()
