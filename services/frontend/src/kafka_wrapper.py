"""
Асинхронная обертка для взаимодействия с Apache Kafka.

Этот модуль предоставляет асинхронный интерфейс для работы с Kafka
в контексте веб-приложения. Поддерживает публикацию сообщений,
прослушивание топиков и получение результатов обработки.
"""

import asyncio
import json
import os
import tempfile
from typing import Optional

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from loguru import logger

class KafkaWrapper:
    """Асинхронная обертка для работы с Apache Kafka.
    
    Предоставляет удобный интерфейс для асинхронной работы с Kafka
    в контексте веб-приложения. Поддерживает публикацию сообщений,
    прослушивание топиков и кэширование результатов.
    
    Attributes:
        bootstrap_servers (str): URL Kafka кластера
        config (dict): Конфигурация подключения
        results (dict): Кэш результатов обработки по session_id
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        bootstrap_servers: Optional[str] = None,
    ) -> None:
        """Инициализация асинхронной обертки Kafka.
        
        Args:
            config (Optional[dict]): Словарь конфигурации подключения
            bootstrap_servers (Optional[str]): URL Kafka кластера
        """
        self.bootstrap_servers = bootstrap_servers
        self.config = config
        self.results = {}

        if self.config:
            self._load_env_from_config()
        self._load_env_from_os()

    def _load_env_from_os(self):
        """Загрузка конфигурации из переменных окружения.
        
        Поддерживает переменную окружения: KAFKA_BOOTSTRAP_SERVERS
        """
        if not self.bootstrap_servers:
            self.bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        logger.info("Envs are loaded")

    def _load_env_from_config(self):
        """Загрузка конфигурации из словаря в переменные окружения.
        
        Устанавливает переменную окружения KAFKA_BOOTSTRAP_SERVERS
        из конфигурации.
        """
        os.environ["KAFKA_BOOTSTRAP_SERVERS"] = self.config.get("KAFKA_BOOTSTRAP_SERVERS")

    async def publish(self, msg: dict, topic_name: str):
        """Публикация сообщения в Kafka топик.
        
        Args:
            msg (dict): Сообщение для публикации
            topic_name (str): Имя топика для публикации
        """
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await producer.start()
        try:
            await producer.send_and_wait(topic_name, msg)
            logger.info(f"Message sent to topic {topic_name}")
        finally:
            await producer.stop()

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
        return self.results.pop(session_id)

    async def consume(self, topic_name: str, group_id: str = "default-group"):
        """Прослушивание Kafka топика для получения результатов.
        
        Args:
            topic_name (str): Имя топика для прослушивания
            group_id (str): ID группы консьюмеров
        """
        consumer = AIOKafkaConsumer(
            topic_name,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True,
        )
        await consumer.start()
        try:
            async for message in consumer:
                msg = message.value
                result = msg.get("result")
                session_id = msg.get("inputs", {}).get("session_id")

                if not result:
                    self.results[session_id] = None
                else:
                    report_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf", prefix="/tmp/gradio"
                    )
                    report_file.write(decode_str(result["report"]))
                    self.results[session_id] = {
                        "image": result["image"],
                        "message": result["message"],
                        "report": report_file.name,
                    }
                logger.info(f"Processed message for session_id: {session_id}")
        finally:
            await consumer.stop()
