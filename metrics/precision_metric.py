from decentai.metrics.metric import Metric

class PrecisionMetric(Metric):
    def calculate(self, true_labels, predicted_labels):
        correct_predictions = 0
        total_positive_instances = sum(true_labels)
        for i in range(len(true_labels)):
            if true_labels[i] == predicted_labels[i]:
                correct_predictions += 1
        return correct_predictions / total_positive_instances if total_positive_instances > 0 else 0

