import math

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

data = pd.read_excel("./GDP.xlsx");
data["Date"] = [int(tiempo) + (tiempo - int(tiempo)) * 2.5 for tiempo in data["Date"]]
# data["GDP_LCHANGE"] = data["GDP"].diff()
data["Change"][0] = 0
print(data.head(5))


def compute_Fbeta_score(true_values, predicted_results, beta=1):
    true_values = test["NBER"]

    # Calculate number of true positives, false positives, true negatives, and false negatives
    true_positives = np.sum(np.logical_and(predicted_results == 1, true_values == 1))
    false_positives = np.sum(np.logical_and(predicted_results == 1, true_values == 0))
    true_negatives = np.sum(np.logical_and(predicted_results == 0, true_values == 0))
    false_negatives = np.sum(np.logical_and(predicted_results == 0, true_values == 1))

    # Calculate accuracy, false positive rate, and false negative rate
    accuracy = (true_positives + true_negatives) / len(true_values)
    false_positive_rate = false_positives / (false_positives + true_negatives)
    false_negative_rate = false_negatives / (false_negatives + true_positives)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate Fbeta score
    if precision + recall > 0:
        fbeta_score = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
    else:
        fbeta_score = 0

    return fbeta_score


def create_HMM(initial_state_matrix, transition_matrix, distributions):
    initial_state_probs = tf.Variable(initial_state_matrix, dtype=tf.float32)
    # Suma 1
    initial_state_probs = initial_state_probs / tf.reduce_sum(initial_state_probs)

    # Creamos la matriz de transición
    transition_probs = tf.Variable(transition_matrix, dtype=tf.float32)
    # Suma 1
    transition_probs = transition_probs / tf.reduce_sum(transition_probs, axis=1, keepdims=True)

    # Inicializamos las probabilidades de emisión
    # Supongamos que los cambios se modelan como una distribución Normal
    media_recesion = train.loc[train["NBER"] == 1, "Change"].mean()
    media_no_recesion = train.loc[train["NBER"] == 0, "Change"].mean()
    std_recesion = train.loc[train["NBER"] == 1, "Change"].std(ddof=0)
    std_no_recesion = train.loc[train["NBER"] == 0, "Change"].std(ddof=0)

    starting_loc = tf.Variable([distribution.loc for distribution in distributions], shape=(2,), dtype=tf.float32,
                               name="medias")
    starting_scale = tf.Variable([distribution.scale for distribution in distributions], shape=(2,), dtype=tf.float32,
                                 name="varianzas")
    observation_distribution = tfd.Normal(loc=starting_loc, scale=starting_scale)
    # HiddenMarkovModel
    hmm = tfp.distributions.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(probs=initial_state_probs),
        transition_distribution=tfd.Categorical(probs=transition_probs),
        observation_distribution=observation_distribution,
        num_steps=len(train),
        validate_args=True
    )
    return hmm


train = data
test = data

nber_1 = (train["NBER"] == 1).sum()
nber_0 = (train["NBER"] == 0).sum()

# P(nber = 1 | 0)  y P(nber = 0| 1)
cond_0 = ((train['NBER'] == 1) & (train['NBER'].shift() == 0)).sum()
cond_1 = ((train['NBER'] == 0) & (train['NBER'].shift() == 1)).sum()

# Llamemos Pij a la probabilidad de ir del estado i al j, entonces la matriz de transición es:
p01 = cond_0 / nber_0
p00 = 1 - p01
p10 = cond_1 / nber_1
p11 = 1 - p10
# Tenemos 2 estados, recesión o no recesión
num_states = 2
batch_size = 1

# Vamos a iniciar con las proporciones de transición acorde a los datos
proporcion_rec = (train["NBER"] == 1).sum() / len(train["NBER"])

# Inicializamos las probabilidades iniciales
initial_state_matrix = np.array([1 - proporcion_rec, proporcion_rec])
initial_state_probs = tf.Variable(initial_state_matrix, dtype=tf.float32)
# Suma 1
initial_state_probs = initial_state_probs / tf.reduce_sum(initial_state_probs)

# Creamos la matriz de transición
transition_matrix = np.array([[p00, p01], [p10, p11]])
transition_probs = tf.Variable(transition_matrix, dtype=tf.float32)
# Suma 1
transition_probs = transition_probs / tf.reduce_sum(transition_probs, axis=1, keepdims=True)

# Inicializamos las probabilidades de emisión
# Supongamos que los cambios se modelan como una distribución Normal
media_recesion = train.loc[train["NBER"] == 1, "Change"].mean()
media_no_recesion = train.loc[train["NBER"] == 0, "Change"].mean()
std_recesion = train.loc[train["NBER"] == 1, "Change"].std(ddof=0)
std_no_recesion = train.loc[train["NBER"] == 0, "Change"].std(ddof=0)

prior_1 = tfd.Normal(loc=media_no_recesion, scale=std_no_recesion)
prior_2 = tfd.Normal(loc=media_recesion, scale=std_recesion)

starting_loc = tf.Variable([media_no_recesion, media_recesion], shape=(2,), dtype=tf.float32, name="medias")
starting_scale = tf.Variable([std_no_recesion, std_recesion], shape=(2,), dtype=tf.float32, name="varianzas")
observation_distribution = tfd.Normal(loc=starting_loc, scale=starting_scale)
# HiddenMarkovModel
hmm = tfp.distributions.HiddenMarkovModel(
    initial_distribution=tfd.Categorical(probs=initial_state_probs),
    transition_distribution=tfd.Categorical(probs=transition_probs),
    observation_distribution=observation_distribution,
    num_steps=len(train),
    validate_args=True
)
# COMIENZO POMEGRANATE
dists = [NormalDistribution(media_no_recesion, std_no_recesion), NormalDistribution(media_recesion, std_recesion)]

model = HiddenMarkovModel.from_matrix(transition_matrix, dists, initial_state_matrix)
model.bake()
print(model.fit([train["Change"]]))

# FIN POMEGRANATE

print("ini", initial_state_matrix, "\n trans", transition_matrix, "\n priors", [prior_1, prior_2])


def forward_probabilities(initial_state_matrix, transition_matrix, distributions, observations):
    # number of states & observations
    n_states = len(initial_state_matrix)
    n_obs = len(observations)
    alpha = np.zeros((n_states, n_obs))
    scale = np.zeros(n_obs)
    # Calculate the initial forward probabilities
    matrix = np.array([distributions[0].prob(observations[0]), distributions[1].prob(observations[0])])
    res = np.multiply(initial_state_matrix, matrix)
    alpha[:, 0] = res
    scale[0] = sum(alpha[:, 0])
    alpha[:, 0] = alpha[:, 0] / scale[0]
    # Compute the forward probabilities recursively
    for i in range(1, n_obs):
        for j in range(n_states):
            # alpha[t] = np.matmul(np.matmul(alpha[t-1], transition_matrix) , matrix)
            alpha_aux = [alpha[k, i - 1] * distributions[j].prob(observations[i]) * transition_matrix[k, j] for k in
                         range(n_states)]
            alpha[j, i] = sum(alpha_aux)
            scale[i] += alpha[j, i]
        alpha[:, i] = [alpha[k, i] / scale[i] for k in range(n_states)]

    lik = sum(alpha[:, -1])
    return alpha, scale, lik


def backward_probabilities(scale, initial_state_matrix, transition_matrix, distributions, observations):
    # number of states & observations
    n_states = len(initial_state_matrix)
    n_obs = len(observations)
    beta = np.zeros((n_states, n_obs))
    # Calculate the initial backward probabilities
    beta[:, -1] = np.divide([1, 1], scale[-1])
    # Compute the backward probabilities recursively
    for i in range(2, n_obs + 1):
        for j in range(n_states):
            beta_aux = [beta[k, -i + 1] * distributions[k].prob(observations.iat[-i + 1]) * transition_matrix[j, k] for
                        k in range(n_states)]
            beta[j, -i] = sum(beta_aux)
        beta[:, -i] = np.divide(beta[:, -i], scale[-i])

    start_state = [beta[k, 0] * distributions[k].prob(observations.iat[0]) for k in range(n_states)]
    start_state = np.multiply(start_state, initial_state_matrix)
    start_state_val = sum(start_state)
    return beta, start_state_val


# Probabilidad de transitar de un estado j a k en un tiempo t
def xi_probabilities(forward, backward, transition_matrix, distributions, observations):
    n_states = forward.shape[0]
    n_observations = forward.shape[1]
    xi = np.zeros((n_states, n_observations - 1, n_states))

    for t in range(n_observations - 1):
        for j in range(n_states):
            for k in range(n_states):
                xi[j, t, k] = (forward[j, t] * backward[k, t + 1] * transition_matrix[j, k]
                               * distributions[k].prob(observations.iat[t + 1]))
    return xi


"""
#Probabilidad de estar en el estado i en un tiempo t
def gamma_probabilities(forward, backward, scale):

    n_states = forward.shape[0]
    n_observations = forward.shape[1]
    gamma = np.zeros((n_states, n_observations))

    for i in range(n_observations):
        for j in range(n_states):
          #print(forward[j, i] , backward[j, i] , log_likelihood)
          #gamma[j, i] = np.exp(forward[j, i] + backward[j, i] - log_likelihood)
          gamma[j, i] = forward[j, i] * backward[j, i] / scale[i]

    return gamma
"""


# Probabilidad de estar en el estado i en un tiempo t
def gamma_probabilities(xi):
    n_states = xi.shape[0]
    gamma = np.zeros((n_states, xi.shape[1]))

    for t in range(xi.shape[1]):
        for j in range(n_states):
            gamma[j, t] = sum(xi[j, t, :])

    return gamma


train = data
test = data

n_states = len(initial_state_matrix)
n_obs = len(train["Change"])


def baum_welch_normal(initial_state_matrix, transition_matrix, distributions, verbose=False):
    log_verosim = []
    for iteration in range(30):
        if (verbose): print('\nIteration No: ', iteration + 1)

        # Calling probability functions to calculate all probabilities
        alf, scale, lik_alpha = forward_probabilities(initial_state_matrix, transition_matrix, distributions,
                                                      train["Change"])
        beta, lik_beta = backward_probabilities(scale, initial_state_matrix, transition_matrix, distributions,
                                                train["Change"])
        log_verosim.append(- np.sum(np.log(scale)))
        xi = xi_probabilities(alf, beta, transition_matrix, distributions, train["Change"])
        gamma = gamma_probabilities(xi)

        # La matriz a es la matriz de transición
        # La matriz b es la matriz de emisión
        a = np.zeros((n_states, n_states))

        # 'delta' matrix
        for j in range(n_states):
            initial_state_matrix[j] = gamma[j, 0] / np.sum(
                gamma[:, 0])  # Revisar que Gamma no es una probabilidad aqui!

        # 'a' matrix
        for j in range(n_states):
            for i in range(n_states):
                denomenator_a = 0
                for t in range(n_obs - 1):
                    a[j, i] = a[j, i] + xi[j, t, i]
                    denomenator_a += gamma[j,t]

                denomenator_b = [xi[j, t_x, i_x] for t_x in range(n_obs - 1) for i_x in range(n_states)]
                denomenator_b = sum(denomenator_b)

                if (denomenator_a == 0):
                    a[j, i] = 0
                else:
                    a[j, i] = a[j, i] / denomenator_a

        # 'b' matrix
        mu = np.zeros(n_states)
        sigma = np.zeros(n_states)

        # mu
        for i in range(n_states):
            num = 0
            den = 0
            for t in range(n_obs - 1):
                num = num + (gamma[i, t] * train["Change"][t])
                den += gamma[i, t]
            mu[i] = num / den
        # sigma
        for i in range(n_states):
            num = 0
            den = 0
            for t in range(n_obs - 1):
                num = gamma[i, t] * ((train["Change"][t] - mu[i]) ** 2)
                den = gamma[i, t]
            sigma[i] = np.sqrt(num / den)

        if (verbose): print('\nMatrix a:\n')
        if (verbose): print(np.matrix(a.round(decimals=4)))
        transition_matrix = a
        distributions[0] = tfd.Normal(loc=mu[0], scale=sigma[0])
        distributions[1] = tfd.Normal(loc=mu[1], scale=sigma[1])
        if (verbose): print(distributions[0].loc, "\n", distributions[0].scale, "\n", distributions[1].loc, "\n",
                            distributions[1].scale)
        new_alf, new_scale, new_lik_alpha = forward_probabilities(initial_state_matrix, transition_matrix,
                                                                  distributions, train["Change"])
        new_log_verosim = - np.sum(np.log(new_scale))
        better_new_log = create_HMM(initial_state_matrix, transition_matrix, distributions).log_prob(
            train["Change"]).numpy()
        if (verbose): print('New log-verosim: ', -new_log_verosim)
        diff = np.abs(log_verosim[iteration] - new_log_verosim)
        if (verbose): print('Difference in forward probability: ', diff)

        if (diff < 0.0000001):
            break
    return log_verosim, initial_state_matrix, transition_matrix, [prior_1, prior_2]


log_verosim, initial_state_matrix, transition_matrix, [prior_1, prior_2] = baum_welch_normal(initial_state_matrix,
                                                                                             transition_matrix,
                                                                                             [prior_1, prior_2],
                                                                                             verbose=False)
plt.plot(log_verosim)
plt.show()
hmm = create_HMM(initial_state_matrix, transition_matrix, [prior_1, prior_2])
predicted_results = hmm.posterior_mode(test["Change"].astype(np.float32)).numpy()
print("F1-score:", compute_Fbeta_score(test["NBER"], predicted_results))
print("F2-score:", compute_Fbeta_score(test["NBER"], predicted_results, beta=2))

print("ini", initial_state_matrix, "\n trans", transition_matrix, "\n priors", [prior_1, prior_2])



