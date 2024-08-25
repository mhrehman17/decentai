# Import the MetricFactory from decentai's metric_factory module.
from decentai.metrics.metric_factory import MetricFactory

# Define true labels and predicted labels for classification task.
true_labels1 = [0, 0, 1, 1]
predicted_labels1 = [0, 1, 1, 1]
true_labels2 = [0, 0, 1, 1, 0, 0]
predicted_labels2 = [0, 1, 1, 1, 1 , 0]
true_labels3 = [0, 0, 1, 1, 1, 1, 1]
predicted_labels3 = [0, 1, 1, 1, 0, 1, 0]
# Create an instance of the MetricFactory class to generate metrics.
metric_factory = MetricFactory()

# Retrieve precision metric from the factory using its name.
precision_metric = metric_factory.get_metric("precision")

# Retrieve recall metric from the factory using its name.
recall_metric = metric_factory.get_metric("recall")

# Retrieve F1 score (F1) metric from the factory using its name.
f1_metric = metric_factory.get_metric("f1")

# Calculate and print precision, recall, and F1 score metrics for the classification task.
print(precision_metric.calculate(true_labels1, predicted_labels1))
print(recall_metric.calculate(true_labels2, predicted_labels2))
print(f1_metric.calculate(true_labels3, predicted_labels3))
