import asyncio
from abc import ABC, abstractmethod

class StreamingData(ABC):
    @abstractmethod
    async def stream_data(self):
        pass

class KafkaStreamingData(StreamingData):
    def __init__(self, bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers

    async def stream_data(self):
        # Establish a connection to the Kafka broker
        kafka_consumer = await asyncio.create_task(kafka.Consumer.frombootstrap_servers(self.bootstrap_servers))

        while True:
            # Consume data from the topic
            message = await kafka_consumer.get_message()

            # Process the incoming data
            print(f"Received data: {message.value}")

            # Acknowledge the message to mark it as processed
            await kafka_consumer.commit(message.offset + 1)