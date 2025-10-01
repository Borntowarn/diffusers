"""
Подмодуль wrappers для интеграции с внешними сервисами.

Содержит обертки для взаимодействия с различными внешними сервисами:
- TritonWrapper: Интеграция с Triton Inference Server
- RabbitWrapper: Интеграция с RabbitMQ
- KafkaWrapper: Интеграция с Apache Kafka
"""

from .rabbit_wrapper import RabbitWrapper
from .triton_wrapper import TritonWrapper
from .kafka_wrapper import KafkaWrapper