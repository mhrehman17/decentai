import unittest
from decentai.data_pipeline.pipeline import MyDataPipeline
from decentai.data_pipeline.non_iid_datasets.non_iid_dataset import NonIIDDatasets
from decentai.data_pipeline.storage.distributed_database import DistributedDatabase
from decentai.data_pipeline.partition.data_partitioning import DataPartitioning
from decentai.data_pipeline.replication.data_replication import DataReplication
from decentai.data_pipeline.analytics.real_time_analytics import RealTimeAnalytics
import asyncio

import numpy as np

class TestDataPipeline(unittest.TestCase):

    def test_data_pipeline(self):
        num_samples = 1000
        num_features = 784
        dataset_name = 'mnist'
        non_iid_ds = NonIIDDatasets(dataset_name)
        train_X, test_X, train_y, test_y = non_iid_ds.generate_non_iid_data(num_samples, num_features)

        data_pipeline = MyDataPipeline()

        steps = [
            {'source': 'non_iid_dataset', 'args': {'dataset_name': dataset_name}},
         ]

async def main():
    my_data_pipeline = MyDataPipeline()
    await my_data_pipeline.run(steps)

if __name__ == '__main__':
    asyncio.run(unittest.main())
