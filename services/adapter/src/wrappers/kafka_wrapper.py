"""
Обертка для взаимодействия с Apache Kafka.

Этот модуль предоставляет интерфейс для работы с Kafka топиками,
включая создание топиков, публикацию и прослушивание сообщений.
Поддерживает конфигурацию через ini файлы и переменные окружения.
"""

import json
from dataclasses import dataclass, field
import configparser
import json
import time
import os
import traceback
from typing import Union, Optional, Any, Callable, List, Dict

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from kafka.structs import TopicPartition
import sys

from .. import logger


@dataclass
class KafkaAnswer:
    """Структура ответа для Kafka сообщений.
    
    Содержит результаты обработки данных с временными метками
    и автоматически генерирует JSON представление.
    
    Attributes:
        time (Optional[float]): Время обработки в секундах
        inputs (Optional[dict]): Входные данные для обработки
        result (Optional[dict]): Результаты обработки
        json (str): JSON представление ответа (генерируется автоматически)
    """
    time: Optional[float]
    inputs: Optional[dict]
    result: Optional[dict]
    json: str = field(init=False)

    def __post_init__(self):
        """Инициализация JSON представления ответа.
        
        Создает JSON строку с результатами обработки и временем выполнения.
        """
        self.json = json.dumps(
            {"inputs": self.inputs, "process_time": self.time, "result": self.result},
            ensure_ascii=False,
        )


class KafkaWrapper:
    """Обертка для работы с Apache Kafka топиками.
    
    Предоставляет удобный интерфейс для подключения к Kafka,
    создания топиков, публикации и прослушивания сообщений.
    Поддерживает конфигурацию через ini файлы и переменные окружения.
    
    ВНИМАНИЕ: НАХОДИТСЯ В БЕТА РЕЖИМЕ
    
    Поддерживаемые переменные окружения:
    - KAFKA_URL: URL Kafka кластера
    - OUTPUT_TOPIC: Имя выходного топика
    - INPUT_TOPIC: Имя входного топика
    - GROUP_ID: ID группы консьюмеров
    - NUM_REPLICAS: Количество реплик для топиков
    
    Attributes:
        bootstrap_servers (str): URL Kafka кластера
        input_topic (str): Имя входного топика
        output_topic (str): Имя выходного топика
        num_replicas (int): Количество реплик для топиков
        consumer: Kafka консьюмер
        producer: Kafka продюсер
    """

    def __init__(
        self,
        config_path: str = None,
        service_name: str = None,
        config: Dict = {},
        bootstrap_servers: Optional[str] = None,
        input_topic: Optional[str] = None,
        input_partitions: Optional[List[int]] = None,
        output_topic: Optional[str] = None,
        output_partition: Optional[int] = None,
        swap_topics: bool = False,
        num_replicas: Optional[int] = None,
        consumer_kwargs: Dict = {},
        publisher_kwargs: Dict = {},
    ) -> None:
        """Инициализация обертки Kafka.
        
        Конфигурация может быть задана тремя способами (в порядке приоритета):
        1. Аргументами инициализации класса
        2. ini файлом конфигурации с указанием сервиса
        3. Переменными окружения
        
        Args:
            config_path (str, optional): Путь до ini файла конфигурации
            service_name (str, optional): Наименование сервиса в ini файле
            config (dict, optional): Словарь конфигурации
            bootstrap_servers (str, optional): URL Kafka кластера
            input_topic (str, optional): Имя входного топика
            input_partitions (List[int], optional): Партиции для подписки
            output_topic (str, optional): Имя выходного топика
            output_partition (int, optional): Партиция для отправки сообщений
            swap_topics (bool, optional): Поменять местами входной и выходной топики
            num_replicas (int, optional): Количество реплик для топиков
            consumer_kwargs (Dict): Дополнительные аргументы для KafkaConsumer
            publisher_kwargs (Dict): Дополнительные аргументы для KafkaProducer
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.num_replicas = num_replicas
        self.config = config
        self.consumer_kwargs = consumer_kwargs
        self.publisher_kwargs = publisher_kwargs
        self.output_partition = output_partition

        if config_path and service_name and os.path.exists(config_path):
            self.config = configparser.ConfigParser()
            self.config.read(config_path)
            self.config = self.config[service_name]
        self._load_config()

        if swap_topics:
            self.input_topic, self.output_topic = self.output_topic, self.input_topic

        if self.input_topic:
            self._create_topic(self.input_topic)
            self.consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=json.loads,
                **self.consumer_kwargs,
            )
            if input_partitions and isinstance(input_partitions, list):
                self.consumer.assign(
                    [TopicPartition(self.input_topic, i) for i in input_partitions]
                )
            else:
                self.consumer.subscribe([self.input_topic])
            logger.info(f"Input topic {self.input_topic} has been connected")
        else:
            self.consumer = None

        if self.output_topic:
            self._create_topic(self.output_topic)
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: x.encode("utf-8"),
                **self.publisher_kwargs,
            )
            logger.info(f"Output topic {self.output_topic} has been connected")
        else:
            self.producer = None

    def _process_item(self, pipeline, **payload) -> tuple[dict, float]:
        """Обработка элемента через pipeline.
        
        Args:
            pipeline: Функция для обработки данных
            **payload: Данные для обработки
            
        Returns:
            tuple: Кортеж из результата обработки и времени выполнения
        """
        try:
            logger.info("Start processing an item")
            start_time = time.time()
            result = pipeline(**payload)
            process_time = time.time() - start_time
            logger.info(f"Item has been processed in {process_time}s")
        except Exception as e:
            result = None
            process_time = None
            logger.error(f"{traceback.format_exc()}")
        return result, process_time

    def _load_config(self):
        """Загрузка конфигурации из переменных окружения и словаря конфига.
        
        Поддерживает переменные окружения: KAFKA_URL, INPUT_TOPIC, OUTPUT_TOPIC,
        NUM_REPLICAS, GROUP_ID
        """
        if not self.bootstrap_servers:
            self.bootstrap_servers = self.config.get(
                "KAFKA_URL", os.environ.get("KAFKA_URL", None)
            )

        if not self.input_topic:
            self.input_topic = self.config.get(
                "INPUT_TOPIC", os.environ.get("INPUT_TOPIC", None)
            )

        if not self.output_topic:
            self.output_topic = self.config.get(
                "OUTPUT_TOPIC", os.environ.get("OUTPUT_TOPIC", None)
            )

        if not self.num_replicas:
            self.num_replicas = int(
                self.config.get("NUM_REPLICAS", os.environ.get("NUM_REPLICAS", 1))
            )

        self.consumer_kwargs["group_id"] = self.consumer_kwargs.get(
            "group_id", self.config.get("GROUP_ID", os.environ.get("GROUP_ID", None))
        )
        self.consumer_kwargs["auto_offset_reset"] = self.consumer_kwargs.get(
            "auto_offset_reset", "latest"
        )

        logger.info("Config has been loaded")

    def _create_answer(self, time, payload: dict, result: Optional[dict]) -> None:
        """Создание объекта ответа для Kafka.
        
        Args:
            time (float): Время обработки
            payload (dict): Входные данные
            result (Optional[dict]): Результаты обработки
            
        Returns:
            KafkaAnswer: Объект ответа
        """
        return KafkaAnswer(time, payload, result)

    def _create_topic(self, name):
        """Создание топика в Kafka.
        
        Args:
            name (str): Имя топика для создания
        """
        admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        if name not in admin_client.list_topics():
            try:
                topic = NewTopic(
                    name=name, num_partitions=self.num_replicas, replication_factor=1
                )
                admin_client.create_topics(new_topics=[topic], validate_only=False)
                logger.success(f"Topic {name} has been created")
            except TopicAlreadyExistsError as e:
                logger.warning(f"Topic {name} already exists")

    def publish(
        self,
        data: Union[list[dict], dict],
        time: Optional[float] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Публикация сообщений в Kafka топик.
        
        Args:
            data (Union[list[dict], dict]): Данные для отправки в топик
            time (float, optional): Время обработки
            payload (dict, optional): Входные данные
            
        Raises:
            AssertionError: Если продюсер не инициализирован
        """
        assert self.producer, "There is producer needed"

        if not isinstance(data, list):
            data = [data]

        for item in data:
            if payload:
                answer = self._create_answer(time, payload, item).json
            else:
                answer = json.dumps(item)
            self.producer.send(
                self.output_topic, answer, partition=self.output_partition
            )
            self.producer.flush()
            logger.debug(f"Publish msg to {self.output_topic}")

    def listen(
        self,
        num=-1,
        pipeline: Optional[Callable] = None,
        consumer_timeout_ms: float = float("inf"),
    ) -> None:
        """Прослушивание сообщений из Kafka топика.
        
        Если указан pipeline, то при получении сообщения его содержимое
        передается в функцию для обработки.
        
        Args:
            num (int, optional): Количество сообщений для обработки. -1 для бесконечного прослушивания
            pipeline (Callable, optional): Функция для обработки сообщений
            consumer_timeout_ms (float, optional): Таймаут консьюмера в миллисекундах
            
        Returns:
            list: Список обработанных сообщений (если pipeline не указан)
            
        Raises:
            AssertionError: Если консьюмер не инициализирован
        """
        assert self.consumer, "There is consumer needed"

        if pipeline:
            logger.info(f"Consumer gets pipeline: {pipeline.__class__.__name__}")

        self.consumer.config["consumer_timeout_ms"] = consumer_timeout_ms

        payloads = []
        logger.info(f"Start consuming on {self.input_topic}")
        for n, msg in enumerate(self.consumer):
            try:
                if self.consumer_kwargs["group_id"]:
                    self.consumer.commit()
                logger.debug(f"Got and commited message")
                payload = msg.value

                if pipeline:
                    result, time = self._process_item(pipeline, **payload)
                    if self.producer:
                        self.publish(result, time, payload)
                else:
                    payloads.append(payload)
                    if n + 1 == num:
                        return payloads
            except Exception as e:
                if e == KeyboardInterrupt:
                    self.connection.close()
                    sys.exit()
                logger.error(f"{traceback.format_exc()}")
        return payloads
