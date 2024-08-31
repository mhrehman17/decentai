from decentai.losses.loss_calculator import LossCalculator
import numpy as np

class CategoricalCrossEntropyLoss(LossCalculator):
    def calculate_loss(self, true_labels, predicted_scores):
        predicted_labels = np.argmax(predicted_scores, axis=0)
        loss = 0.0
        for i in range(len(true_labels)):
            label = true_labels[i]
            score = predicted_scores[i, predicted_labels[i]]
            scores = predicted_scores[i, :]
            scores -= score
            exp_scores = np.exp(scores)
            sum_exp_scores = np.sum(exp_scores)
            loss += -score / sum_exp_scores
        return loss