# Import necessary modules from decentai library
from decentai.metrics.metric import Metric  # This module contains the base class for all metrics.
from decentai.metrics.precision_metric import PrecisionMetric  # This metric calculates the precision of a model.
from decentai.metrics.recall_metric import RecallMetric  # This metric calculates the recall of a model.
from decentai.metrics.f1_metric import F1Metric  # This metric calculates the F1 score (harmonic mean of precision and recall) of a model.

# Define a factory class to create metrics based on their names
class MetricFactory:
    """
    A class that creates instances of specific metric classes based on the provided metric name.
    """
    @staticmethod
    def get_metric(metric_name):
        """
        Method to create and return an instance of the specified metric class.

        Args:
            metric_name (str): The name of the metric. It can be either 'precision', 'recall' or 'f1'.

        Returns:
            Metric: An instance of the requested metric class.
        """

        if metric_name.lower() == "precision":
            """
            Check if the provided metric name is 'precision'. If it is, return an instance of PrecisionMetric.
            """
            return PrecisionMetric

        elif metric_name.lower() == 'recall':
            """
            Check if the provided metric name is 'recall'. If it is, return an instance of RecallMetric.
            """
            return RecallMetric

        elif metric_name.lower() == 'f1':
            """
            Check if the provided metric name is 'f1'. If it is, return an instance of F1Metric.
            """
            return F1Metric

        else:
            """
            Raise a ValueError if the provided metric name is not recognized. This could be due to an unsupported or unknown metric name.
            """
            raise ValueError(f"Unsupported agent: {metric_name}")

# The above code can now be used to create instances of specific metric classes based on their names.