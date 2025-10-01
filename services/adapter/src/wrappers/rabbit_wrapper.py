"""
Обертка для взаимодействия с RabbitMQ.

Этот модуль предоставляет интерфейс для работы с очередями RabbitMQ,
включая подключение, создание топиков, публикацию и прослушивание сообщений.
Поддерживает автоматическое переподключение и обработку ошибок.
"""

import json
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import *

import amqp
import yaml

from .. import logger


@dataclass
class RabbitAnswer:
    """Структура ответа для RabbitMQ сообщений.
    
    Содержит результаты обработки данных с временными метками
    и автоматически генерирует JSON представление.
    
    Attributes:
        time (float | None): Время обработки в секундах
        inputs (dict | None): Входные данные для обработки
        result (dict | None): Результаты обработки
        json (str): JSON представление ответа (генерируется автоматически)
    """
    time: float | None
    inputs: dict | None
    result: dict | None
    json: str = field(init=False)

    def __post_init__(self):
        """Инициализация JSON представления ответа.
        
        Создает JSON строку с результатами обработки, временем выполнения
        и текущим временем в формате ISO.
        """
        self.json = json.dumps(
            {
                "inputs": self.inputs,
                "process_time": self.time,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": self.result,
            },
            ensure_ascii=False,
        )


class RabbitWrapper:
    """Обертка для работы с RabbitMQ очередями.
    
    Предоставляет удобный интерфейс для подключения к RabbitMQ,
    создания топиков, публикации и прослушивания сообщений.
    Поддерживает автоматическое переподключение при сбоях.
    
    Attributes:
        url (str): URL подключения к RabbitMQ
        input_topic (str): Имя входной очереди
        output_topic (str): Имя выходной очереди
        config (dict): Конфигурация подключения
        connection: Соединение с RabbitMQ
        channel: Канал для работы с очередями
    """

    def __init__(
        self,
        config: dict = {},
        url: str | None = None,
        input_topic: str | None = None,
        output_topic: str | None = None,
        swap_topics: bool = False,
    ) -> None:
        """Инициализация обертки RabbitMQ.
        
        Args:
            config (dict): Словарь конфигурации подключения
            url (str, optional): URL подключения к RabbitMQ
            input_topic (str, optional): Имя входной очереди
            output_topic (str, optional): Имя выходной очереди
            swap_topics (bool, optional): Поменять местами входную и выходную очереди
        """
        self.url = url
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.config = config

        self._load_config()

        if swap_topics:
            self.input_topic, self.output_topic = self.output_topic, self.input_topic

        self._connect()

        if self.input_topic:
            self._create_topic(self.input_topic)
            logger.info(f"Input topic {self.input_topic} has been connected")

        if self.output_topic:
            self._create_topic(self.output_topic)
            logger.info(f"Output topic {self.output_topic} has been connected")

    def _load_config(self):
        """Загрузка конфигурации из переменных окружения и словаря конфига.
        
        Парсит URL подключения и извлекает параметры подключения.
        Поддерживает переменные окружения: RABBIT_URL, INPUT_TOPIC, OUTPUT_TOPIC
        """
        if not self.url:
            self.url = os.environ.get("RABBIT_URL", self.config.get("RABBIT_URL", None))

        self.host, self.virtual_host = self.url.split("@")[1].split("/")
        _, self.username, self.password = self.url.split("@")[0].split(":")
        self.username = self.username.replace("//", "")

        if len(self.virtual_host) == 0:
            self.virtual_host = "/"

        if not self.input_topic:
            self.input_topic = os.environ.get(
                "INPUT_TOPIC", self.config.get("INPUT_TOPIC", None)
            )

        if not self.output_topic:
            self.output_topic = os.environ.get(
                "OUTPUT_TOPIC", self.config.get("OUTPUT_TOPIC", None)
            )

        logger.info("Config has been loaded")

    def _create_topic(self, topic_name):
        """Создание очереди в RabbitMQ.
        
        Args:
            topic_name (str): Имя очереди для создания
        """
        self.channel.queue_declare(
            queue=topic_name,
            durable=True,
            exclusive=False,
            auto_delete=False,
            arguments={"x-queue-type=classic": "classic"},
        )
        logger.debug(f"Topic {topic_name} has been created")

    def _create_answer(self, time, payload: dict, result: dict | None) -> None:
        """Создание объекта ответа для RabbitMQ.
        
        Args:
            time (float): Время обработки
            payload (dict): Входные данные
            result (dict | None): Результаты обработки
            
        Returns:
            RabbitAnswer: Объект ответа
        """
        return RabbitAnswer(time, payload, result)

    def _connect(self) -> Any:
        """Подключение к RabbitMQ серверу.
        
        Выполняет попытки подключения с интервалом 5 секунд
        до успешного установления соединения.
        
        Raises:
            Exception: При критических ошибках подключения
        """
        tries = 0
        while True:
            try:
                tries += 1
                logger.info(f"Trying to connect at {tries} time")
                self.connection = amqp.Connection(
                    host=self.host,
                    userid=self.username,
                    password=self.password,
                    virtual_host=self.virtual_host,
                )

                self.connection.connect()
                self.channel = self.connection.channel()
                logger.info("Connection successful")
                break
            except Exception as e:
                logger.warning(f"Connection failed. Waiting for a 5 seconds...")
                time.sleep(5)

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

    def publish(
        self, data: list[dict] | dict, time=None, payload=None, output_topic=None
    ) -> None:
        """Публикация сообщений в RabbitMQ очередь.
        
        Args:
            data (list[dict] | dict): Данные для публикации
            time (float, optional): Время обработки
            payload (dict, optional): Входные данные
            output_topic (str, optional): Имя выходной очереди
            
        Raises:
            AssertionError: Если не указана выходная очередь
        """
        if not output_topic:
            output_topic = self.output_topic

        assert output_topic, "There is output topic needed"

        if not isinstance(data, list):
            data = [data]

        for item in data:
            try:
                if payload:
                    answer = self._create_answer(time, payload, item).json
                else:
                    answer = json.dumps(item)
                msg = amqp.basic_message.Message(body=answer)
                self.channel.basic_publish(msg, exchange="", routing_key=output_topic)
                logger.info(f"Publish msg to {output_topic}")
            except Exception as e:
                logger.error(f"Message publishing failed: {traceback.format_exc()}")

    def listen(
        self, num=-1, pipeline: Callable | None = None, ack: bool = False
    ) -> None:
        """Прослушивание сообщений из RabbitMQ очереди.
        
        Args:
            num (int, optional): Количество сообщений для обработки. -1 для бесконечного прослушивания
            pipeline (Callable, optional): Функция для обработки сообщений
            ack (bool, optional): Подтверждать ли получение сообщений
            
        Returns:
            list: Список обработанных сообщений (если pipeline не указан)
            
        Raises:
            AssertionError: Если не указана входная очередь
        """
        assert self.input_topic, "There is input topic needed"

        if pipeline:
            ack = True
            logger.info(f"Consumer gets pipeline: {pipeline.__class__.__name__}")

        n = 0
        payloads = []
        logger.info(f"Start consuming on {self.input_topic}")
        while True:
            try:
                message = self.channel.basic_get(queue=self.input_topic)
                if message:
                    logger.debug(f"Got message")
                    if ack:
                        self.channel.basic_ack(delivery_tag=message.delivery_tag)
                        logger.debug(f"Acked on message")
                    payload = json.loads(message.body)

                    if pipeline:
                        result, time = self._process_item(pipeline, **payload)
                        self.publish(result, time, payload)
                    else:
                        payloads.append(payload)
                        if n + 1 == num:
                            return payloads
                    n += 1
                elif not pipeline:
                    return payloads
            except Exception as e:
                self.connection.close()
                logger.error(f"{traceback.format_exc()}")
