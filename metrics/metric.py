class Metric:
    def __init__(self):
        self = self

    def calculate(self, true_labels, predicted_labels):
        raise NotImplementedError("Subclasses must implement this method")
