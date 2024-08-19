from decentai.metrics.metric import Metric
from decentai.metrics.precision_metric import PrecisionMetric
from decentai.metrics.recall_metric import RecallMetric

class F1Metric(Metric):
    def calculate(self, true_labels, predicted_labels):
        precision = PrecisionMetric().calculate(true_labels, predicted_labels)
        recall = RecallMetric().calculate(true_labels, predicted_labels)
        return 2 * (precision * recall) / (precision + recall)
