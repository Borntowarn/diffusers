"""
Асинхронная обертка для взаимодействия с RabbitMQ.

Этот модуль предоставляет асинхронный интерфейс для работы с RabbitMQ
в контексте веб-приложения. Поддерживает публикацию сообщений,
прослушивание очередей и получение результатов обработки.
"""

import asyncio
import json
import os
import time
from typing import Optional

import aio_pika
from aio_pika.abc import AbstractRobustConnection
from loguru import logger

class RabbitWrapper:
    """Асинхронная обертка для работы с RabbitMQ.
    
    Предоставляет удобный интерфейс для асинхронной работы с RabbitMQ
    в контексте веб-приложения. Поддерживает публикацию сообщений,
    прослушивание очередей и кэширование результатов.
    
    Attributes:
        url (str): URL подключения к RabbitMQ
        input_topic (str): Имя входной очереди
        output_topic (str): Имя выходной очереди
        config (dict): Конфигурация подключения
        results (dict): Кэш результатов обработки по session_id
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        url: Optional[str] = None,
        input_topic: Optional[str] = None,
        output_topic: Optional[str] = None,
    ) -> None:
        """Инициализация асинхронной обертки RabbitMQ.
        
        Args:
            config (Optional[dict]): Словарь конфигурации подключения
            url (Optional[str]): URL подключения к RabbitMQ
            input_topic (Optional[str]): Имя входной очереди
            output_topic (Optional[str]): Имя выходной очереди
        """
        self.url = url
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.config = config or {}
        self.results = {}

        self._load_config()

    def _load_config(self) -> None:
        """Загрузка конфигурации из переменных окружения и словаря конфига.
        
        Поддерживает переменные окружения: RABBIT_URL, INPUT_TOPIC, OUTPUT_TOPIC
        """
        if not self.url:
            self.url = os.environ.get("RABBIT_URL", self.config.get("RABBIT_URL", None))
        if not self.input_topic:
            self.input_topic = os.environ.get("INPUT_TOPIC", self.config.get("INPUT_TOPIC", None))
        if not self.output_topic:
            self.output_topic = os.environ.get("OUTPUT_TOPIC", self.config.get("OUTPUT_TOPIC", None))
        logger.info("Config has been loaded")

    async def connect(self) -> AbstractRobustConnection | None:
        """Установление подключения к RabbitMQ серверу.
        
        Выполняет попытки подключения с интервалом 5 секунд
        до успешного установления соединения.
        
        Returns:
            AbstractRobustConnection | None: Соединение с RabbitMQ или None при ошибке
        """
        tries = 0
        while True:
            tries += 1
            try:
                connection = await aio_pika.connect_robust(self.url)
                logger.info(f"Connection to RabbitMQ established")
                return connection
            except Exception as e:
                logger.info(f"Connection failed. Waiting for a 5 seconds...")
                time.sleep(5)

    async def publish(self, msg: dict, queue_name: str):
        """Публикация сообщения в RabbitMQ очередь.
        
        Args:
            msg (dict): Сообщение для публикации
            queue_name (str): Имя очереди для публикации
        """
        connection = await self.connect()
        async with connection:
            channel = await connection.channel()
            _ = await channel.declare_queue(queue_name, durable=True)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(msg).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=queue_name,
            )

    async def get_result(self, session_id):
        """Получение результата обработки по session_id.
        
        Ожидает появления результата в кэше и возвращает его.
        
        Args:
            session_id (str): Идентификатор сессии
            
        Returns:
            dict: Результат обработки или None
        """
        while session_id not in self.results:
            await asyncio.sleep(0.5)
        logger.info(f"Returning result for session_id: {session_id}")
        return self.results.pop(session_id)

    async def consume(self, listen_queue_name: str):
        """Прослушивание очереди RabbitMQ для получения результатов.
        
        Args:
            listen_queue_name (str): Имя очереди для прослушивания
        """
        connection = await self.connect()
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)
            listen_queue = await channel.declare_queue(
                listen_queue_name, durable=True, auto_delete=False
            )
            async with listen_queue.iterator() as queue_iter:
                async for message in queue_iter:
                    msg = json.loads(message.body)
                    await message.ack()
                    result = msg.get("result")
                    session_id = msg.get("inputs").get("session_id")
                    logger.info(f"Received message for session_id: {session_id}")

                    if not result:
                        logger.info(f"Result is None for session_id: {session_id}")
                        self.results[session_id] = None
                    else:
                        logger.info(f"Writing result for session_id: {session_id}")
                        self.results[session_id] = {
                            "message": result["message"],
                            "df_desease": result["df_desease"],
                            "warnings": result["warnings"],
                        }
