import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.distributions import constraints
import seaborn;
from pomegranate import *

seaborn.set_style('whitegrid')
import torch
import tensorflow as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from matplotlib import pylab as plt
import scipy.stats as stats
import scipy.special as special


class HMM():
    def __init__(self, initial_state_matrix=None, transition_matrix=None, distributions=None):
        self.initial_state_matrix = initial_state_matrix


class GHMM(HMM):
    def __init__(self, initial_state_matrix, transition_matrix, distributions, observations):
        self.initial_state_matrix = initial_state_matrix
        self.log_start = np.log(initial_state_matrix)
        self.transition_matrix = transition_matrix
        self.log_transition = np.log(transition_matrix)
        self.num_states = len(initial_state_matrix)
        self.n_obs = len(observations)
        self.distributions = distributions
        self.observations = observations

    def forward_probs(self):
        # number of states & observations
        n_states = len(self.initial_state_matrix)
        n_obs = len(self.observations)
        alpha = np.zeros((n_states, n_obs))
        scale = np.zeros(n_obs)
        # Calculate the initial forward probabilities
        matrix = np.array([self.distributions[0].prob(self.observations.iat[0]),
                           self.distributions[1].prob(self.observations.iat[0])])
        res = np.multiply(self.initial_state_matrix, matrix)
        alpha[:, 0] = res
        scale[0] = sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / scale[0]
        # Compute the forward probabilities recursively
        for i in range(1, n_obs):
            for j in range(n_states):
                # alpha[t] = np.matmul(np.matmul(alpha[t-1], transition_matrix) , matrix)
                alpha_aux = [
                    alpha[k, i - 1] * self.distributions[j].prob(self.observations.iat[i]) * self.transition_matrix[
                        k, j] for
                    k in
                    range(n_states)]
                alpha[j, i] = sum(alpha_aux)
                scale[i] += alpha[j, i]
            alpha[:, i] = [alpha[k, i] / scale[i] for k in range(n_states)]

        lik = sum(alpha[:, -1])
        return alpha, scale, lik

    def backward_probs(self, scale):
        # number of states & observations
        n_states = len(self.initial_state_matrix)
        n_obs = len(self.observations)
        beta = np.zeros((n_states, n_obs))
        # Calculate the initial backward probabilities
        beta[:, -1] = np.divide([1, 1], scale[-1])
        # Compute the backward probabilities recursively
        for i in range(2, n_obs + 1):
            for j in range(n_states):
                beta_aux = [beta[k, -i + 1] * self.distributions[k].prob(self.observations.iat[-i + 1]) \
                            * self.transition_matrix[j, k]
                            for
                            k in range(n_states)]
                beta[j, -i] = sum(beta_aux)
            beta[:, -i] = np.divide(beta[:, -i], scale[-i])

        start_state = [beta[k, 0] * self.distributions[k].prob(self.observations.iat[0]) for k in range(n_states)]
        start_state = np.multiply(start_state, self.initial_state_matrix)
        start_state_val = sum(start_state)
        return beta, start_state_val

    def xi_probs(self, forward, backward):
        n_states = forward.shape[0]
        n_observations = forward.shape[1]
        xi = np.zeros((n_states, n_observations - 1, n_states))

        for t in range(n_observations - 1):
            for j in range(n_states):
                for k in range(n_states):
                    xi[j, t, k] = (forward[j, t] * backward[k, t + 1] * self.transition_matrix[j, k]
                                   * self.distributions[k].prob(self.observations.iat[t + 1]))
        return xi

    def gamma_probs(self, xi):
        n_states = xi.shape[0]
        gamma = np.zeros((n_states, xi.shape[1]))

        for t in range(xi.shape[1]):
            for j in range(n_states):
                gamma[j, t] = sum(xi[j, t, :])

        return gamma

    def baum_welch_normal(self, observations=None, n_iter=30, verbose=False):
        # Si nos introducen un vector de observaciones usaremos ese para entrenar el modelo
        if observations is not None:
            self.observations = observations
        log_verosim = []
        for iteration in range(n_iter):
            if (verbose): print('\nIteration No: ', iteration + 1)

            # Calling probability functions to calculate all probabilities
            alf, scale, lik_alpha = self.forward_probs()
            beta, lik_beta = self.backward_probs(scale)
            log_verosim.append(np.sum(np.log(scale)))
            xi = self.xi_probs(alf, beta)
            gamma = self.gamma_probs(xi)

            # La matriz a es la matriz de transición
            # La matriz b es la matriz de emisión
            a = np.zeros((self.num_states, self.num_states))

            # 'delta' matrix
            for j in range(self.num_states):
                self.initial_state_matrix[j] = gamma[j, 0]

            # 'a' matrix
            for j in range(self.num_states):
                for i in range(self.num_states):
                    denomenator_a = 0
                    for t in range(self.n_obs - 1):
                        a[j, i] = a[j, i] + xi[j, t, i]
                        denomenator_a += gamma[j, t]

                    denomenator_b = [xi[j, t_x, i_x] for t_x in range(self.n_obs - 1) for i_x in range(self.num_states)]
                    denomenator_b = sum(denomenator_b)

                    if (denomenator_a == 0):
                        a[j, i] = 0
                    else:
                        a[j, i] = a[j, i] / denomenator_a

            # 'b' matrix
            mu = np.zeros(self.num_states)
            sigma = np.zeros(self.num_states)

            # mu
            for i in range(self.num_states):
                num = 0
                den = 0
                for t in range(self.n_obs - 1):
                    num = num + (gamma[i, t] * self.observations.iat[t])
                    den += gamma[i, t]
                mu[i] = num / den
            # sigma
            for i in range(self.num_states):
                num = 0
                den = 0
                for t in range(self.n_obs - 1):
                    num += gamma[i, t] * ((self.observations.iat[t] - mu[i]) ** 2)
                    den += gamma[i, t]
                sigma[i] = np.sqrt(num / den)

            if (verbose): print('\nMatrix a:\n')
            if (verbose): print(np.matrix(a.round(decimals=4)))
            self.transition_matrix = a
            self.distributions[0] = tfd.Normal(loc=mu[0], scale=sigma[0])
            self.distributions[1] = tfd.Normal(loc=mu[1], scale=sigma[1])
            if (verbose): print(self.distributions[0].loc, "\n",
                                self.distributions[0].scale, "\n",
                                self.distributions[1].loc, "\n",
                                self.distributions[1].scale)
            new_alf, new_scale, new_lik_alpha = self.forward_probs()
            new_log_verosim = np.sum(np.log(new_scale))
            if (verbose): print('New log-verosim: ', new_log_verosim)
            diff = np.abs(log_verosim[iteration] - new_log_verosim)
            if (verbose): print('Difference in forward probability: ', diff)

            if (diff < 0.0001):
                break
        return log_verosim, self.initial_state_matrix, self.transition_matrix, self.distributions

    def viterbi(self, observations=None):
        # Si nos introducen un vector de observaciones usaremos ese
        if observations is not None:
            self.observations = observations
        log_start = np.log(self.initial_state_matrix)
        log_trans = np.log(self.transition_matrix)
        # Aqui guardamos la probabilidad máxima hasta ese punto
        vit = [{}]
        # El camino que hemos llevado hasta la sol
        # En lugar de quedarnos con el puntero, guardamos todo el camino
        path = {}
        n_states = len(log_start)
        n_obs = len(self.observations)
        # Initialize base cases (t == 0)
        matrix = [self.distributions[0].log_prob(self.observations.iat[0]),
                  self.distributions[1].log_prob(self.observations.iat[0])]
        for i in range(n_states):
            vit[0][i] = log_start[i] + matrix[i]
            path[i] = [i]

        # Run Viterbi for t > 0
        for t in range(1, n_obs):
            vit.append({})
            updated_path = {}
            for i in range(n_states):
                (prob, state) = max(
                    (vit[t - 1][k] + log_trans[k][i] + self.distributions[i].log_prob(self.observations.iat[t]), k) \
                    for k in range(n_states))
                vit[t][i] = prob
                updated_path[i] = path[state] + [i]
            # Nos quedamos con el mejor camino
            path = updated_path
        (prob, state) = max((vit[0][y], y) for y in range(n_states))
        return (prob, path[state])

    def forward_logprob(self):
        # number of states & observations
        n_states = len(self.log_start)
        n_obs = len(self.observations)
        log_alpha = np.zeros((n_states, n_obs))
        scale = np.zeros(n_obs)
        # Calculate the initial forward probabilities
        matrix = [self.distributions[0].log_prob(self.observations.iat[0]),
                  self.distributions[1].log_prob(self.observations.iat[0])]
        for i, (x, y) in enumerate(zip(self.log_start, matrix)):
            if x == 0:
                log_alpha[i, 0] = 0
            else:
                log_alpha[i, 0] = x + y
        # Compute the forward probabilities recursively
        for i in range(1, n_obs):
            for j in range(n_states):
                log_alpha_aux = []
                for k in range(n_states):
                    if log_alpha[k, i - 1] == 0:
                        log_alpha_aux.append(0)
                    else:
                        log_alpha_aux.append(log_alpha[k, i - 1] + self.log_transition[k, j])

                log_alpha[j, i] = special.logsumexp(log_alpha_aux) + \
                                  self.distributions[j].log_prob(self.observations.iat[i])

        lik = special.logsumexp(log_alpha[:, -1])
        return log_alpha, lik

    def backward_logprob(self):
        # number of states & observations
        n_states = len(self.log_start)
        n_obs = len(self.observations)
        log_beta = np.zeros((n_states, n_obs))
        # Calculate the initial backward probabilities
        log_beta[:, -1] = [0, 0]  # log([1,1])
        # Compute the backward probabilities recursively
        for i in range(2, n_obs + 1):
            for j in range(n_states):
                beta_aux = [log_beta[k, -i + 1] + self.distributions[k].log_prob(self.observations.iat[-i + 1]) + \
                            self.log_transition[j, k] for k in range(n_states)]
                log_beta[j, -i] = special.logsumexp(beta_aux)

        start_state = [log_beta[k, 0] + self.distributions[k].log_prob(self.observations.iat[0]) + \
                       self.log_start[k] for k in range(n_states)]
        start_state_val = special.logsumexp(start_state)
        return log_beta, start_state_val

    def gamma_logprob(self, alf, beta):
        log_gamma = alf + beta
        with np.errstate(under="ignore"):
            a_lse = special.logsumexp(log_gamma, 0, keepdims=True)
        log_gamma -= a_lse
        return log_gamma

    def log_xi_sum(self, log_forward, log_backward):
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
                                   + self.distributions[k].log_prob(self.observations.iat[t + 1])
                                   + log_backward[k, t + 1]
                                   - log_prob)
                    xi_sum[j, k] = np.logaddexp(xi_sum[j, k], xi[j, t, k])

        return xi, xi_sum

    def log_baum_welch_normal(self, observations=None, n_iter=20, verbose=False):
        # Si nos introducen un vector de observaciones usaremos ese para entrenar el modelo
        if observations is not None:
            self.observations = observations
        n_states = len(self.log_start)
        n_obs = len(self.observations)
        log_lik = []
        for iter in range(n_iter):
            # Primero inicializo los valores que vamos a utilizar
            new_start = [0, 0]
            new_trans = [[0, 0], [0, 0]]
            post = np.zeros(n_states)
            nobs = 0  # number of samples in data
            obs = np.zeros(n_states)
            obs_sqr = np.zeros(n_states)
            alf, lik = self.forward_logprob()
            log_lik.append(lik)
            beta, lik_b = self.backward_logprob()
            gamma = self.gamma_logprob(alf, beta)
            # Posteriors está al reves que hmmlearn
            posteriors = np.exp(gamma)
            nobs += 1
            new_start += posteriors[:, 0]
            log_xi, xi_sum = self.log_xi_sum(alf, beta)
            log_gamma0 = special.logsumexp([log_xi[0, t_x, i_x] for t_x in range(n_obs - 1) for i_x in range(n_states)])
            log_gamma1 = special.logsumexp([log_xi[1, t_x, i_x] for t_x in range(n_obs - 1) for i_x in range(n_states)])
            xi00 = special.logsumexp([log_xi[0, x_t, 0] for x_t in range(n_obs - 1)])
            xi01 = special.logsumexp([log_xi[0, x_t, 1] for x_t in range(n_obs - 1)])
            xi10 = special.logsumexp([log_xi[1, x_t, 0] for x_t in range(n_obs - 1)])
            xi11 = special.logsumexp([log_xi[1, x_t, 1] for x_t in range(n_obs - 1)])
            t00 = np.exp(xi00 - log_gamma0)
            t01 = np.exp(xi01 - log_gamma0)
            t10 = np.exp(xi10 - log_gamma1)
            t11 = np.exp(xi11 - log_gamma1)
            new_trans = np.array([[t00, t01], [t10, t11]])

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
            prior_1 = tfd.Normal(loc=means[0], scale=covars[0])
            prior_2 = tfd.Normal(loc=means[1], scale=covars[1])
            self.distributions = [prior_1, prior_2]
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
        return log_lik, new_start, new_trans, self.distributions
