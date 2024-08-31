from decentai.losses.loss_calculator import LossCalculator
import numpy as np

class MeanSquaredErrorLoss(LossCalculator):
    def calculate_loss(self, true_labels, predicted_labels):
        return np.mean((true_labels - predicted_labels) ** 2)