import numpy as np
def compute_Fbeta_score(true_values, predicted_results, beta = 1):
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
      fbeta_score = (1+beta**2) * (precision * recall) / ((beta**2) * precision + recall)
  else:
      fbeta_score = 0

  return fbeta_score