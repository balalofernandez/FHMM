from typing import List, Any
from HMM.utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style('whitegrid')
import torch
import tensorflow as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from matplotlib import pylab as plt
import scipy.stats as stats
import scipy.special as special
from enum import Enum
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class ScalingAlgorithm(Enum):
    logarithm = 1,
    division = 2
class TrainingAlgorithm(Enum):
    baum_welch = 1,
    label = 2
class DefaultInitialization(Enum):
    aic = 1,
    bic = 2


class HMM():
    def __init__(self, initial_state_matrix=None, transition_matrix=None, distributions=None):
        self.initial_state_matrix = initial_state_matrix

    @abstractmethod
    def conditional_dist(self,observations):
        raise Exception("Este método tiene que ser sobreescrito por una subclase.")

    @abstractmethod
    def forecast_dist(self, times:int):
        raise Exception("Este método tiene que ser sobreescrito por una subclase.")

    def forecast_values(self, times:int):
        if(self.observations is None):
            raise Exception("Se necesita introducir un vector de observaciones al crear el HMM")
        distribuciones = self.forecast_dist(times)
        return [distribuciones[t].mean() for t in range(times)]

    def forecast_states(self, times:int):
        if(self.observations is None):
            raise Exception("Se necesita introducir un vector de observaciones al crear el HMM")

        log_alfa, _ = self._forward_logprob()
        forecast_states = []
        state_probabilities = []
        last_alfa = log_alfa[:,-1]
        for t in range(times):
            last_alfa_aux = []
            for j in range(self.n_states):
                last_alfa_aux.append(
                    special.logsumexp([last_alfa[k]+self.log_transition[k,j] for k in range(self.n_states)]))
            last_alfa = last_alfa_aux[:]
            state_probabilities.append(np.exp(last_alfa-special.logsumexp(last_alfa)))
            forecast_states.append(np.argmax(last_alfa))
        return forecast_states,state_probabilities

    def _forward_probs(self):
        # number of states & observations
        n_states = len(self.initial_state_matrix)
        n_obs = len(self.observations)
        alpha = np.zeros((n_states, n_obs))
        scale = np.zeros(n_obs)
        # Calculate the initial forward probabilities
        matrix = np.array([dist.prob(self.observations[0]) for dist in self.distributions])
        res = np.multiply(self.initial_state_matrix, matrix)
        alpha[:, 0] = res
        scale[0] = sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / scale[0]
        # Compute the forward probabilities recursively
        for i in range(1, n_obs):
            for j in range(n_states):
                # alpha[t] = np.matmul(np.matmul(alpha[t-1], transition_matrix) , matrix)
                alpha_aux = [
                    alpha[k, i - 1] * self.distributions[j].prob(self.observations[i]) * \
                    self.transition_matrix[k, j] for
                    k in
                    range(n_states)]
                alpha[j, i] = sum(alpha_aux)
                scale[i] += alpha[j, i]
            alpha[:, i] = [alpha[k, i] / scale[i] for k in range(n_states)]

        lik = sum(alpha[:, -1])
        return alpha, scale, lik

    def _backward_probs(self, scale):
        # number of states & observations
        n_states = len(self.initial_state_matrix)
        n_obs = len(self.observations)
        beta = np.zeros((n_states, n_obs))
        # Calculate the initial backward probabilities
        beta[:, -1] = np.divide(np.ones(n_states), scale[-1])
        # Compute the backward probabilities recursively
        for i in range(2, n_obs + 1):
            for j in range(n_states):
                beta_aux = [beta[k, -i + 1] * self.distributions[k].prob(self.observations[-i + 1]) \
                            * self.transition_matrix[j, k]
                            for
                            k in range(n_states)]
                beta[j, -i] = sum(beta_aux)
            beta[:, -i] = np.divide(beta[:, -i], scale[-i])

        start_state = [beta[k, 0] * self.distributions[k].prob(self.observations[0]) for k in range(n_states)]
        start_state = np.multiply(start_state, self.initial_state_matrix)
        start_state_val = sum(start_state)
        return beta, start_state_val

    def _xi_probs(self, forward, backward):
        n_states = forward.shape[0]
        n_observations = forward.shape[1]
        xi = np.zeros((n_states, n_observations - 1, n_states))

        for t in range(n_observations - 1):
            for j in range(n_states):
                for k in range(n_states):
                    xi[j, t, k] = (forward[j, t] * backward[k, t + 1] * self.transition_matrix[j, k]
                                   * self.distributions[k].prob(self.observations[t + 1]))
        return xi

    def _gamma_probs(self, xi):
        n_states = xi.shape[0]
        gamma = np.zeros((n_states, xi.shape[1]))

        for t in range(xi.shape[1]):
            for j in range(n_states):
                gamma[j, t] = sum(xi[j, t, :])

        return gamma

    def _forward_logprob(self):
        # number of states & observations
        n_obs = len(self.observations)
        log_alpha = np.zeros((self.n_states, n_obs))
        scale = np.zeros(n_obs)
        # Calculate the initial forward probabilities
        matrix = np.array([dist.log_prob(self.observations[0]) for dist in self.distributions])
        for i, (x, y) in enumerate(zip(self.log_start, matrix)):
            if x == 0:
                log_alpha[i, 0] = 0
            else:
                log_alpha[i, 0] = x + y
        # Compute the forward probabilities recursively
        for i in range(1, n_obs):
            for j in range(self.n_states):
                log_alpha_aux = []
                for k in range(self.n_states):
                    if log_alpha[k, i - 1] == 0:
                        log_alpha_aux.append(0)
                    else:
                        log_alpha_aux.append(log_alpha[k, i - 1] + self.log_transition[k, j])

                log_alpha[j, i] = special.logsumexp(log_alpha_aux) + self.distributions[j].log_prob(
                    self.observations[i])

        lik = special.logsumexp(log_alpha[:, -1])
        return log_alpha, lik

    def _backward_logprob(self):
        # number of states & observations
        n_states = len(self.log_start)
        n_obs = len(self.observations)
        log_beta = np.zeros((n_states, n_obs))
        # Calculate the initial backward probabilities
        log_beta[:, -1] = np.zeros(shape=n_states)  # log([1,1])
        # Compute the backward probabilities recursively
        for i in range(2, n_obs + 1):
            for j in range(n_states):
                beta_aux = [log_beta[k, -i + 1] + self.distributions[k].log_prob(self.observations[-i + 1]) + \
                            self.log_transition[j, k] for k in range(n_states)]
                log_beta[j, -i] = special.logsumexp(beta_aux)

        start_state = [log_beta[k, 0] + self.distributions[k].log_prob(self.observations[0]) + \
                       self.log_start[k] for k in range(n_states)]
        start_state_val = special.logsumexp(start_state)
        return log_beta, start_state_val

    def _gamma_logprob(self, alf, beta):
        log_gamma = np.array(alf + beta)
        with np.errstate(under="ignore"):
            a_lse = special.logsumexp(log_gamma, 0, keepdims=True)
        log_gamma -= a_lse
        return log_gamma

    def _xi_logprob(self, log_forward, log_backward):
        n_states = log_forward.shape[0]
        n_observations = log_forward.shape[1]
        log_prob = special.logsumexp(log_forward[:, -1])  # lik
        xi = np.zeros((n_states, n_observations - 1, n_states))
        xi_sum = np.zeros((n_states, n_states))
        for t in range(n_observations - 1):
            for j in range(n_states):
                for k in range(n_states):
                    xi[j, t, k] = (log_forward[j, t]
                                   + self.log_transition[j, k]
                                   + self.distributions[k].log_prob(self.observations[t + 1])
                                   + log_backward[k, t + 1]
                                   - log_prob)
                    xi_sum[j, k] = np.logaddexp(xi_sum[j, k], xi[j, t, k])

        return xi, xi_sum

    def viterbi(self, observations: List[Any]=None, state_probs:bool=False):
        # Si nos introducen un vector de observaciones usaremos ese
        if observations is not None:
            self.observations = observations
        # Aqui guardamos la probabilidad máxima hasta ese punto
        vit = [{}]
        # El camino que hemos llevado hasta la sol
        # En lugar de quedarnos con el puntero, guardamos todo el camino
        path = {}
        n_states = len(self.log_start)
        n_obs = len(self.observations)
        # Initialize base cases (t == 0)
        matrix = np.array([dist.prob(self.observations[0]) for dist in self.distributions])
        for i in range(n_states):
            vit[0][i] = self.log_start[i] + matrix[i]
            path[i] = [i]

        # Run Viterbi for t > 0
        for t in range(1, n_obs):
            vit.append({})
            updated_path = {}
            for i in range(n_states):
                (prob, state) = max(
                    (
                    vit[t - 1][k] + self.log_transition[k][i] + self.distributions[i].log_prob(self.observations[t]), k) \
                    for k in range(n_states))
                vit[t][i] = prob
                updated_path[i] = path[state] + [i]
            # Nos quedamos con el mejor camino
            path = updated_path
        (prob, state) = max((vit[0][y], y) for y in range(n_states))
        if(state_probs):
            alf,_ = self._forward_logprob()
            beta,_ = self._backward_logprob()
            gamma = np.exp(self._gamma_logprob(alf,beta))
            return (np.exp(prob), path[state],gamma)
        return (np.exp(prob), path[state])

    def train(self, observations:  List[Any], train_size: float, labels: List[int] = None, iterations=30,
              verbose = False,
              algorithm=TrainingAlgorithm.baum_welch,beta=1):
        if (labels is not None and algorithm == TrainingAlgorithm.label):
            X_train, x_test, y_train, y_test = train_test_split(
                observations, labels, train_size=train_size,shuffle=False)
            log_lik = self.label_train(X_train, y_train)
        else:
            if (labels is not None and train_size != 1):
                X_train, x_test, y_train, y_test = train_test_split(
                    observations, labels, train_size=train_size,shuffle=False)
            elif(train_size<1.0 and train_size>0.0):
                X_train, _ = train_test_split(observations, train_size=train_size,shuffle=False)
            else:
                X_train = observations
                x_test = observations
                y_test = labels
            if (self.scaling_algorithm == ScalingAlgorithm.logarithm):
                log_lik = self.log_train_baum_welch(X_train, verbose=verbose, n_iter=iterations)
            elif (self.scaling_algorithm == ScalingAlgorithm.division):
                log_lik = self.train_baum_welch(X_train, verbose=verbose, n_iter=iterations)

        if (labels is not None):
            # Si hay etiquetas, validamos el entrenamiento con ellas
            # Si no, habrá que usar otros métodos
            decode = np.array(self.viterbi(x_test)[1])
            fbeta = compute_Fbeta_score(y_test, decode,beta)
            return (log_lik[-1],fbeta)

        return log_lik[-1],None

    @abstractmethod
    def train_baum_welch(self, observations:  List[Any] = None, n_iter=30, verbose=False) ->List[float]:
        raise Exception("Este método tiene que ser sobreescrito por una subclase.")

    @abstractmethod
    def log_train_baum_welch(self, observations:  List[Any] = None, n_iter=20, verbose=False) ->List[float]:
        raise Exception("Este método tiene que ser sobreescrito por una subclase.")

    @abstractmethod
    def label_train(self, observations: List[Any], labels:List[int]) -> List[float]:
        raise Exception("Este método tiene que ser sobreescrito por una subclase.")

    def AIC(self, log_lik:float, num_params:int) -> float:
        return -2*log_lik+2*num_params

    def BIC(self,log_lik:float, num_params:int, num_obs:int) -> float:
        return -2*log_lik + num_params * np.log(num_obs)

    def _initialize_parameters_aic(self, observations,max_params = 5):
        #Recibimos un modelo únicamente con observaciones, iremos aumentando los parámetros, ajustándolo y calculando el AIC
        best_model = {
            "best_aic" : float("inf")
        }
        for i in range(2,max_params+1):
            self.n_states = i
            self._initialize_parameters()
            log_lik = self.train(self.observations,1,verbose=False)
            aic_value = self.AIC(log_lik, self.n_states)
            print(i," parameters: ", aic_value)
            if(aic_value < best_model["best_aic"]):
                best_model = {
                    "best_aic" : aic_value,
                    "best_transition": self.transition_matrix,
                    "best_start":self.initial_state_matrix,
                    "best_distributions":self.distributions
                }
            else:
                break
        self.set_parameters(best_model)
        return best_model["best_aic"]

    def _initialize_parameters_bic(self, observations,max_params = 5):
        #Recibimos un modelo únicamente con observaciones, iremos aumentando los parámetros, ajustándolo y calculando el BIC
        best_model = {
            "best_bic" : float("inf")
        }
        for i in range(2,max_params+1):
            self.n_states = i
            self._initialize_parameters()
            log_lik = self.train(self.observations,1,verbose=False)
            bic_value = self.BIC(log_lik, self.n_states,len(self.observations))
            print(i," parameters: ", bic_value)
            if(bic_value < best_model["best_bic"]):
                best_model = {
                    "best_bic": bic_value,
                    "best_transition": self.transition_matrix,
                    "best_start":self.initial_state_matrix,
                    "best_distributions":self.distributions
                }
            else:
                break

        self.set_parameters(best_model)
        return best_model["best_bic"]

    def set_parameters(self, param_dict):
        if(param_dict["best_transition"] is not None):
            self.transition_matrix = param_dict["best_transition"]
            self.log_transition = np.log(param_dict["best_transition"])
        if(param_dict["best_start"] is not None):
            self.initial_state_matrix = param_dict["best_start"]
            self.log_start = np.log(param_dict["best_start"])
        if(param_dict["best_distributions"] is not None):
            self.distributions = param_dict["best_distributions"]

    def log_likelihood(self, observations:  List[Any] = None):
        if (observations is None and self.observations is None):
            raise Exception("No se ha introducido un vector de observaciones para calcular la log-verosimilud")
        elif (observations is not None):
            self.observations = observations

        if (self.scaling_algorithm == ScalingAlgorithm.logarithm):
            _, log_lik = self._forward_logprob()
        else:
            _, scale, _ = self._forward_probs()
            log_lik = np.sum(np.log(scale))
        return log_lik

    @abstractmethod
    def _initialize_parameters(self):
        raise Exception("Este método tiene que ser sobreescrito por una subclase.")


class NormalHMM(HMM):
    def __init__(self, initial_state_matrix: List[float] = None, transition_matrix: List[List[float]] = None,
                 medias: List[float] = None, std: List[float] = None,
                 n_states: int = 0, observations: List[float] = None,
                 scaling_algorithm: ScalingAlgorithm = ScalingAlgorithm.logarithm,
                 initialization = DefaultInitialization.bic):

        self.scaling_algorithm = scaling_algorithm
        self.observations = np.array(observations)
        if (n_states <= 0 and (
                initial_state_matrix is None or transition_matrix is None or medias is None is std is None)
                and observations is not None):
            # Inician únicamente con el vector de observaciones
            if(initialization == DefaultInitialization.aic):
                self._initialize_parameters_aic(observations)
            else:
                self._initialize_parameters_bic(observations)

        elif (not (initial_state_matrix is None or transition_matrix is None or medias is None or std is None)):
            self.n_states = len(initial_state_matrix)
            if (len(medias) != len(std)):
                raise Exception("Las longitudes de los vectores de medias y varianzas no coinciden, "
                                "por favor introduce vectores de igual tamaño")
            self.initial_state_matrix = np.array(initial_state_matrix)
            self.log_start = np.log(self.initial_state_matrix)
            self.transition_matrix = np.array(transition_matrix)
            self.log_transition = np.log(transition_matrix)
            self.distributions = [tfd.Normal(loc=loc, scale=scale)
                                  for loc, scale in zip(medias, std)]
        elif (n_states > 0 and observations is not None):
            self.n_states = n_states
            self._initialize_parameters()
        else:
            raise Exception("No hay parámetros suficientes, "
                            "por favor indica el número de estados o las matrices de probabilidades "
                            "(probabilidad inicial, transicion,) ")

    def _initialize_parameters(self):
        # Tenemos que inicializar los parámetros
        self.initial_state_matrix = np.full(self.n_states, 1 / self.n_states)
        self.log_start = np.log(self.initial_state_matrix)
        self.transition_matrix = np.full((self.n_states, self.n_states), 1 / self.n_states)
        self.log_transition = np.log(self.transition_matrix)
        # inicializamos las distribuciones de forma naive
        std_obs = self.observations.std()
        mean_obs = self.observations.mean()
        medias = np.linspace(mean_obs - std_obs, mean_obs + std_obs, self.n_states)
        self.distributions = [tfd.Normal(loc=loc, scale=std_obs)
                              for loc in medias]

    def train_baum_welch(self, observations: List[Any] = None, n_iter=30, verbose=False) ->List[float]:
        # Si nos introducen un vector de observaciones usaremos ese para entrenar el modelo
        if observations is not None:
            self.observations = observations
        log_lik = []
        n_obs = len(observations)
        for iteration in range(n_iter):
            if (verbose): print('\nIteration No: ', iteration + 1)

            # Calling probability functions to calculate all probabilities
            alf, scale, lik_alpha = self._forward_probs()
            beta, lik_beta = self._backward_probs(scale)
            log_lik.append(np.sum(np.log(scale)))
            xi = self._xi_probs(alf, beta)
            gamma = self._gamma_probs(xi)

            # La matriz a es la matriz de transición
            # La matriz b es la matriz de emisión
            a = np.zeros((self.n_states, self.n_states))

            # 'delta' matrix
            for j in range(self.n_states):
                self.initial_state_matrix[j] = gamma[j, 0]

            # 'a' matrix
            for j in range(self.n_states):
                for i in range(self.n_states):
                    denomenator_a = 0
                    for t in range(n_obs - 1):
                        a[j, i] = a[j, i] + xi[j, t, i]
                        denomenator_a += gamma[j, t]

                    denomenator_b = [xi[j, t_x, i_x] for t_x in range(n_obs - 1) for i_x in range(self.n_states)]
                    denomenator_b = sum(denomenator_b)

                    if (denomenator_a == 0):
                        a[j, i] = 0
                    else:
                        a[j, i] = a[j, i] / denomenator_a

            # 'b' matrix
            mu = np.zeros(self.n_states)
            sigma = np.zeros(self.n_states)

            # mu
            for i in range(self.n_states):
                num = 0
                den = 0
                for t in range(n_obs - 1):
                    num = num + (gamma[i, t] * self.observations[t])
                    den += gamma[i, t]
                mu[i] = num / den
            # sigma
            for i in range(self.n_states):
                num = 0
                den = 0
                for t in range(n_obs - 1):
                    num += gamma[i, t] * ((self.observations[t] - mu[i]) ** 2)
                    den += gamma[i, t]
                sigma[i] = np.sqrt(num / den)

            if (verbose): print('\nMatrix a:\n')
            if (verbose): print(np.matrix(a.round(decimals=4)))
            self.transition_matrix = a
            self.distributions = [tfd.Normal(loc=loc, scale=scale)
                                  for loc, scale in zip(mu, sigma)]
            if (verbose): print(self.distributions[0].loc, "\n",
                                self.distributions[0].scale, "\n",
                                self.distributions[1].loc, "\n",
                                self.distributions[1].scale)
            new_alf, new_scale, new_lik_alpha = self._forward_probs()
            new_log_lik = np.sum(np.log(new_scale))
            if (verbose): print('New log-verosim: ', new_log_lik)
            diff = np.abs(log_lik[iteration] - new_log_lik)
            if (verbose): print('Difference in forward probability: ', diff)

            if (diff < 0.0001):
                break
        return log_lik

    def log_train_baum_welch(self, observations: List[Any] = None, n_iter=20, verbose=False) ->List[float]:
        # Si nos introducen un vector de observaciones usaremos ese para entrenar el modelo
        if observations is not None:
            self.observations = observations
        n_obs = len(self.observations)
        log_lik = []
        for iter in range(n_iter):
            # Primero inicializo los valores que vamos a utilizar
            new_start = np.zeros(self.n_states)
            nobs = 0  # number of samples in data
            alf, lik = self._forward_logprob()
            log_lik.append(lik)
            beta, lik_b = self._backward_logprob()
            gamma = self._gamma_logprob(alf, beta)
            # Posteriors está al reves que hmmlearn
            posteriors = np.exp(gamma)
            nobs += 1
            new_start += posteriors[:, 0]
            log_xi, xi_sum = self._xi_logprob(alf, beta)
            log_gamma = np.array([special.logsumexp([log_xi[state, t_x, i_x] for t_x in range(n_obs - 1)
                                                     for i_x in range(self.n_states)])
                                  for state in range(self.n_states)])
            new_trans = np.exp(np.array([[xi_sum[i,j] - log_gamma[i] for j in range(self.n_states)] for i in range(self.n_states)]))
            obs_sqr = posteriors @ (self.observations ** 2)
            obs = posteriors @ self.observations
            post = posteriors.sum(axis=1)
            means = obs / post
            covars = np.sqrt(obs_sqr / post - (means ** 2))  # VAR = E[X^2]-E^[X]
            for x in new_start:
                if (x < 0.00001):
                    x = 1
            self.log_start = np.log(new_start)  # Ten cuidado cuando tiende a 0!
            self.log_transition = np.log(new_trans)
            self.distributions = [tfd.Normal(loc=loc, scale=scale)
                                  for loc, scale in zip(means, covars)]
            if (verbose == True): print("start:", new_start)
            if (verbose == True): print("transition:", new_trans)
            if (verbose == True): print("means:", means)
            if (verbose == True): print("covars:", covars)
            if (verbose == True): print("log-lik", log_lik[iter])
            diff = 1
            if (iter > 0):
                diff = np.abs(log_lik[iter] - log_lik[iter - 1])

            if (diff < 0.0001):
                break

        self.initial_state_matrix = np.exp(self.log_start)
        self.transition_matrix = np.exp(self.log_transition)
        return log_lik

    def label_train(self, observations: List[Any], labels:List[int]) -> List[float]:
        if observations is not None:
            self.observations = observations

        n_obs = len(self.observations)
        labels = np.array(labels)
        #Contamos las proporciones existentes de cada estado, obteniendo un vector shape(n_states,)
        self.initial_state_matrix = np.array([(labels == i).sum() /len(labels) for i in range(self.n_states)])

        #Vamos a contar las transiciones a otros estados
        shifted_labels = np.roll(labels, shift=-1)
        #En transition_matrix[i,j] metemos la probabilidad de saltar a j
        transition_matrix = np.zeros((self.n_states,self.n_states))
        for i in range(self.n_states):
            total_i_states = (labels == i).sum()
            for j in range(self.n_states):
                transition_matrix[i,j] = ((labels == i) & (shifted_labels == j)).sum()/total_i_states

        self.transition_matrix = transition_matrix
        medias = [np.mean(observations[(labels == i)]) for i in range(self.n_states)]
        std = [np.std(observations[(labels == i)]) for i in range(self.n_states)]
        self.distributions = [tfd.Normal(loc=loc, scale=scale)
                              for loc, scale in zip(medias, std)]
        model = {
            "best_transition": self.transition_matrix,
            "best_start":self.initial_state_matrix,
            "best_distributions":self.distributions
        }
        self.set_parameters(model)
        log_lik = self.log_likelihood()
        return [log_lik]

    def conditional_dist(self,observations):
        #Calculamos la distribución condicional
        self.observations = observations
        log_alfa, _ = self._forward_logprob()
        log_beta, _ = self._backward_logprob()

        n_obs = len(observations)
        conditional_distributions = []
        d = np.zeros((self.n_states, n_obs))
        d[:,0] = self.log_start + log_beta[:,0]
        for t in range(1,n_obs):
            for j in range(self.n_states):
                log_alfa_aux = []
                for k in range(self.n_states):
                    log_alfa_aux.append(log_alfa[k,t-1]+self.log_transition[k,j])
                d[j,t] = special.logsumexp(log_alfa_aux)+log_beta[j,t]

        #Calculamos el vector de pesos w
        w = np.exp([[d[i,t] - special.logsumexp(d[:,t]) for t in range(n_obs)] for i in range(self.n_states)])
        for t in range(n_obs):
            conditional_distributions.append(tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=w),
                components_distribution= tfd.Normal(
                    loc=[dist.loc for dist in self.distributions],
                    scale=[dist.scale for dist in self.distributions]
                )
            ))
        return conditional_distributions

    def forecast_dist(self, times:int):
        if(self.observations is None):
            raise Exception("Se necesita introducir un vector de observaciones al crear el HMM")

        log_alfa, _ = self._forward_logprob()
        forecast_distribution = []
        last_alfa = log_alfa[:,-1]
        for t in range(times):
            last_alfa_aux = []
            for j in range(self.n_states):
                last_alfa_aux.append(
                    special.logsumexp([last_alfa[k]+self.log_transition[k,j] for k in range(self.n_states)]))
            last_alfa = last_alfa_aux[:]
            w = np.exp([last_alfa[j]-special.logsumexp(last_alfa_aux) for j in range(self.n_states)]).astype(np.float32)
            medias = np.array([dist.loc for dist in self.distributions]).astype(np.float32)
            std = np.array([dist.scale for dist in self.distributions]).astype(np.float32)
            forecast_distribution.append(tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=w),
                components_distribution= tfd.Normal(
                    loc=tf.Variable(medias, dtype=tf.float32, name = "medias"),
                    scale=tf.Variable(std, dtype=tf.float32, name = "varianzas")
                ))
            )
        return forecast_distribution

class CategoricalHMM(HMM):
    def __init__(self, initial_state_matrix: List[float] = None, transition_matrix: List[List[float]] = None,
                 probabilities:List[List[float]]=None,
                 n_states: int = 0, observations:  List[Any] = None,
                 scaling_algorithm: ScalingAlgorithm = ScalingAlgorithm.logarithm,
                 initialization = DefaultInitialization.bic):

        self.scaling_algorithm = scaling_algorithm
        self.observations = np.array(observations)
        if (n_states <= 0 and (
                initial_state_matrix is None or transition_matrix is None)
                and observations is not None):
            # Inician únicamente con el vector de observaciones
            if(initialization == DefaultInitialization.aic):
                self._initialize_parameters_aic(observations)
            else:
                self._initialize_parameters_bic(observations)

        elif (not (initial_state_matrix is None or transition_matrix is None)):
            self.n_states = len(initial_state_matrix)
            self.initial_state_matrix = np.array(initial_state_matrix)
            self.log_start = np.log(self.initial_state_matrix)
            self.transition_matrix = np.array(transition_matrix)
            self.log_transition = np.log(transition_matrix)
            #Por último vamos a recibir un array de n_estados x n_sucesos posibles
            self.distributions = [tfd.Categorical(probs= p)
                                  for p in probabilities]
        elif (n_states > 0 and observations is not None):
            self.n_states = n_states
            self._initialize_parameters()
        else:
            raise Exception("No hay parámetros suficientes, "
                            "por favor indica el número de estados o las matrices de probabilidades "
                            "(probabilidad inicial, transicion,) ")

    def _initialize_parameters(self):
        # Tenemos que inicializar los parámetros
        self.initial_state_matrix = np.full(self.n_states, 1 / self.n_states)
        self.log_start = np.log(self.initial_state_matrix)
        self.transition_matrix = np.full((self.n_states, self.n_states), 1 / self.n_states)
        self.log_transition = np.log(self.transition_matrix)
        # Vamos a crear la matriz de n estados x n sucesos
        n_sucesos = len(set(self.observations))
        self.distributions = [tfd.Categorical(probs=np.full(n_sucesos,1/n_sucesos))
                              for i in range(self.n_states)]

    def train_baum_welch(self, observations:  List[Any] = None, n_iter=30, verbose=False) ->List[float]:
        raise NotImplementedError("Not Implemented")

    def log_train_baum_welch(self, observations:  List[Any] = None, n_iter=20, verbose=False) ->List[float]:
        raise NotImplementedError("Not Implemented")
    def label_train(self, observations, labels:List[int]) -> List[float]:
        raise NotImplementedError("Not Implemented")

    def conditional_dist(self,observations):
        raise NotImplementedError("Not Implemented")

    def forecast_dist(self, times:int):
        raise NotImplementedError("Not Implemented")





