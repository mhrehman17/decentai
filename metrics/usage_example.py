from decentai.metrics.metric_factory import MetricFactory

true_labels = [0, 0, 1, 1]
predicted_labels = [0, 1, 1, 1]

metric_factory = MetricFactory()
precision_metric = metric_factory.get_metric("precision")
recall_metric = metric_factory.get_metric("recall")
f1_metric = metric_factory.get_metric("f1")

print(precision_metric.calculate(true_labels, predicted_labels))
print(recall_metric.calculate(true_labels, predicted_labels))
print(f1_metric.calculate(true_labels, predicted_labels))
