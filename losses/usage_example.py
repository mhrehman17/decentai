from decentai.losses.loss_factory import LossFactory

import numpy as np

loss_factory = LossFactory()
loss_calculator = loss_factory.create_loss_calculator("categorical_cross_entropy")
true_labels = np.array([0, 1, 0])  # Replace placeholder with actual values
predicted_labels = np.array([1, 2, 0])  # Replace placeholder with actual values

loss = loss_calculator.calculate_loss(true_labels, predicted_labels)