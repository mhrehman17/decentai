from decentai.losses.categorical_cross_entropy_loss import CategoricalCrossEntropyLoss
from decentai.losses.mean_squared_error_loss import MeanSquaredErrorLoss

class LossFactory:
    def __init__(self):
        pass

    def create_loss_calculator(self, loss_type: str):
        if loss_type == "categorical_cross_entropy":
            return CategoricalCrossEntropyLoss()
        elif loss_type == "mean_squared_error":
            return MeanSquaredErrorLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")