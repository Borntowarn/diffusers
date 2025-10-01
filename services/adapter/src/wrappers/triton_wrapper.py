"""
Обертка для взаимодействия с Triton Inference Server.

Этот модуль предоставляет интерфейс для работы с моделями машинного обучения,
развернутыми на Triton Inference Server. Поддерживает загрузку моделей,
выполнение инференса и управление жизненным циклом моделей.
"""

import json
import os
import traceback
from typing import *

import numpy as np
import tritonclient.grpc as grpcclient

from .. import logger


class TritonWrapper:
    """Обертка для взаимодействия с Triton Inference Server.
    
    Предоставляет удобный интерфейс для работы с моделями машинного обучения,
    развернутыми на Triton Inference Server. Поддерживает автоматическую
    загрузку моделей, выполнение инференса и управление жизненным циклом.
    
    Attributes:
        config (dict): Конфигурация подключения к Triton
        url (str): URL Triton сервера
        model_name (str): Имя модели для инференса
        model_version (str): Версия модели
        input_names (list): Имена входных тензоров
        output_names (list): Имена выходных тензоров
        input_dtype (list): Типы данных входных тензоров
        input_shape (list): Формы входных тензоров
    """
    
    def __init__(self, config: dict = {}):
        """Инициализация обертки Triton.
        
        Args:
            config (dict): Конфигурация подключения к Triton серверу.
                          Может содержать настройки URL, имени модели, версии,
                          входных/выходных имен и типов данных.
        """
        self.config = config
        self._load_config()
        self.model_version = "1" if not getattr(self, "model_version", None) else self.model_version
        logger.info("Config has been loaded")
        self._load_model()

    def _as_list(self, value: Any) -> list[str]:
        """Преобразование значения в список строк.
        
        Поддерживает различные форматы входных данных:
        - None -> пустой список
        - Список -> возвращает как есть
        - Строка JSON массива -> парсит и возвращает
        - Строка с разделителями -> разбивает по запятым
        - Другое -> преобразует в строку и возвращает в списке
        
        Args:
            value (Any): Значение для преобразования
            
        Returns:
            list[str]: Список строк
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            v = value.strip()
            try:
                if v.startswith("[") and v.endswith("]"):
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [v]
            except Exception:
                pass
            return [s.strip() for s in v.split(",") if s.strip()]
        return [str(value)]

    def _load_config(self) -> None:
        """Загрузка конфигурации из переменных окружения и словаря конфига.
        
        Приоритет: переменные окружения > конфиг > значения по умолчанию
        Поддерживает следующие переменные окружения:
        - TRITON_URL: URL Triton сервера
        - TRITON_MODEL_NAME: Имя модели
        - TRITON_MODEL_VERSION: Версия модели
        - TRITON_INPUT_NAMES: Имена входных тензоров
        - TRITON_OUTPUT_NAMES: Имена выходных тензоров
        - TRITON_INPUT_DTYPE: Типы данных входных тензоров
        - TRITON_INPUT_SHAPE: Формы входных тензоров
        """
        # URL and model identifiers
        self.url = os.environ.get("TRITON_URL", self.config.get("url", None))
        self.model_name = os.environ.get("TRITON_MODEL_NAME", self.config.get("model_name", None))
        self.model_version = os.environ.get("TRITON_MODEL_VERSION", self.config.get("model_version", None))

        # IO names and dtypes
        env_input_names = os.environ.get("TRITON_INPUT_NAMES")
        env_output_names = os.environ.get("TRITON_OUTPUT_NAMES")
        env_input_dtype = os.environ.get("TRITON_INPUT_DTYPE")

        self.input_names = self._as_list(env_input_names) or self._as_list(self.config.get("input_names", None))
        self.output_names = self._as_list(env_output_names) or self._as_list(self.config.get("output_names", None))
        self.input_dtype = self._as_list(env_input_dtype) or self._as_list(self.config.get("input_dtype", None))

        # Optional: input shapes (not used directly here, but keep parity)      
        env_input_shape = os.environ.get("TRITON_INPUT_SHAPE")
        self.input_shape = self._as_list(env_input_shape) or self.config.get("input_shape", None)

    def _postprocess(self, inference_res):
        """Постобработка результатов инференса.
        
        Args:
            inference_res: Результат инференса от Triton клиента
            
        Returns:
            list: Список numpy массивов с результатами
        """
        return [inference_res.as_numpy(out_name) for out_name in self.output_names]

    def __call__(self, *inp: np.array) -> Any:
        """Вызов инференса через оператор вызова.
        
        Args:
            *inp: Входные numpy массивы
            
        Returns:
            Any: Результаты инференса
        """
        return self.inference(*inp)
    
    def _load_model(self):
        """Загрузка модели на Triton сервере.
        
        Note:
            Модель должна быть уже развернута на сервере
        """
        with grpcclient.InferenceServerClient(url=self.url, verbose=False) as client:
            client.load_model(self.model_name)
        logger.info("Model has been loaded inside Triton successfully")
    
    def unload_model(self):
        """Выгрузка модели с Triton сервера.
        
        Note:
            Освобождает ресурсы сервера, занятые моделью
        """
        with grpcclient.InferenceServerClient(url=self.url, verbose=False) as client:
            client.unload_model(self.model_name)
        logger.info("Model has been unloaded inside Triton successfully")

    def inference(self, *inp: np.array):
        """Выполнение инференса на входных данных.
        
        Args:
            *inp: Входные numpy массивы для инференса
            
        Returns:
            list: Список numpy массивов с результатами инференса
            
        Raises:
            Exception: При ошибках подключения к серверу или инференса
        """
        inputs = []

        with grpcclient.InferenceServerClient(url=self.url, verbose=False) as client:
            for i, name, dt in zip(inp, self.input_names, self.input_dtype):
                inputs.append(grpcclient.InferInput(name, i.shape, dt))
                inputs[-1].set_data_from_numpy(i)

            result = client.infer(
                self.model_name, model_version=self.model_version, inputs=inputs
            )
            return self._postprocess(result)

    def __del__(self):
        """Деструктор - вызывается при удалении объекта.
        
        Автоматически выгружает модель с сервера при уничтожении объекта.
        """
        try:
            self.unload_model()
        except Exception as e:
            logger.error(f"Error unloading model: {traceback.format_exc()}")
            pass
