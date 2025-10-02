"""
Модуль адаптера для обработки медицинских изображений грудной клетки.

Этот модуль предоставляет функциональность для:
- Предобработки DICOM и NIFTI изображений
- Инференса с использованием моделей машинного обучения через Triton
- Постобработки результатов анализа
- Интеграции с системами очередей (RabbitMQ)

Основные компоненты:
- Model: Основной класс для обработки медицинских изображений
- ExaminationType: Перечисление типов исследований
- Интеграция с Triton Inference Server для ML инференса
- Поддержка RabbitMQ для асинхронной обработки
"""

import os
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytz
import torch
import torchio as tio
import yaml
from loguru import logger

from .src.message import KafkaMessage
from .src.utils import resize_array
from .src.wrappers import RabbitWrapper, TritonWrapper
from .src.utils import InferencePreprocessor
from .src.exceptions import MissingRequiredTagsException

logger.add(f"{__file__.split('/')[-1].split('.')[0]}.log", rotation="50 MB")
tz = pytz.timezone("Europe/Moscow")

warnings.filterwarnings(
    "ignore", message="Invalid value for VR UI", category=UserWarning
)


class ExaminationType(Enum):
    """Перечисление типов медицинских исследований.
    
    Определяет поддерживаемые форматы медицинских изображений
    для обработки в системе анализа патологий грудной клетки.
    """
    DICOM = "DICOM_FOLDER"  # DICOM формат (стандарт медицинской визуализации)
    NIFTI = "NIFTI"         # NIFTI формат (Neuroimaging Informatics Technology Initiative)


class Model:
    """Основной класс для обработки медицинских изображений грудной клетки.
    
    Этот класс объединяет функциональность предобработки, инференса и постобработки
    для анализа патологий в медицинских изображениях. Использует несколько моделей
    машинного обучения через Triton Inference Server для комплексного анализа.
    
    Attributes:
        vit (TritonWrapper): Модель Vision Transformer для извлечения признаков
        multilabel (TritonWrapper): Модель для мультиклассовой классификации патологий
        binary (TritonWrapper): Модель для бинарной классификации наличия патологии
        binary_threshold (float): Пороговое значение для бинарной классификации
        deseases2thresholds (dict): Пороговые значения для различных патологий
        name2index (dict): Маппинг названий патологий на индексы
        severity_order (list): Порядок патологий по степени опасности
        config (dict): Конфигурация модели
    """
    
    def __init__(self, config: dict) -> None:
        """Инициализация модели с конфигурацией.
        
        Args:
            config (dict): Словарь конфигурации, содержащий настройки для:
                - vit: Конфигурация Vision Transformer модели
                - multilabel: Конфигурация мультиклассовой модели
                - binary: Конфигурация бинарной модели
                - binary_threshold: Пороговое значение для бинарной классификации
                - deseases2thresholds: Пороговые значения для патологий
                - severity_order: Порядок патологий по опасности
        """
        self.vit = TritonWrapper(config["vit_onnx"])
        self.multilabel = TritonWrapper(config["multilabel_onnx"])
        self.binary = TritonWrapper(config["binary_onnx"])
        self.bpr = TritonWrapper(config["bpr"])

        self.binary_threshold = config["binary_threshold"]

        self.deseases2thresholds = config["deseases2thresholds"]
        self.name2index = {name: i for i, name in enumerate(list(self.deseases2thresholds.keys()))}
        self.severity_order = config["severity_order"]

        self.config = config
        logger.success("Adapter has been initialized")

    @staticmethod
    def get_datetime():
        """Получить текущее время в московском часовом поясе.
        
        Returns:
            datetime: Текущее время в часовом поясе Europe/Moscow
        """
        return datetime.now(tz)
    
    @staticmethod
    def datetime_to_str(datetime_obj):
        """Преобразовать объект datetime в строку в ISO формате.
        
        Args:
            datetime_obj (datetime): Объект datetime для преобразования
            
        Returns:
            str: Строка в формате "YYYY-MM-DDTHH:MM:SS.ffffff+HHMM"
        """
        return datetime_obj.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    def preprocess(self, data_path):
        """Предобработка медицинских изображений для анализа.
        
        Выполняет полный цикл предобработки медицинских изображений:
        1. Определение типа исследования (DICOM/NIFTI)
        2. Извлечение архивов при необходимости
        3. Чтение и валидация DICOM серий
        4. Выбор оптимальной серии (минимальная толщина среза и окно контраста)
        5. Нормализация и ресайзинг изображения
        6. Приведение к стандартному формату для инференса
        
        Args:
            data_path (str): Путь к файлу или директории с медицинским исследованием
                           Поддерживаются форматы: .dcm, .nii, .zip, .tar, .tgz, .tar.gz
            
        Returns:
            tuple: Кортеж из трех элементов:
                - np.ndarray: Предобработанный тензор изображения (1, 240, 480, 480)
                - str: Study Instance UID исследования
                - str: Series Instance UID серии
                
        Raises:
            Exception: При ошибках чтения файлов или обработки изображений
        """

        study_id = 'Отсутствует'
        series_id = 'Отсутствует'
        data_path = Path(data_path)
        processor = InferencePreprocessor(self.bpr)

        study_type = (
            ExaminationType.NIFTI
            if ".nii" in Path(data_path).suffixes
            else ExaminationType.DICOM
        )

        # temp_dir = TemporaryDirectory()
        with TemporaryDirectory() as archive_temp_dir, TemporaryDirectory() as thinnest_series_dir:
            if study_type is ExaminationType.DICOM:
                logger.info(f"Обработка DICOM исследования")
                if data_path.is_file() and str(data_path).endswith(
                    (".zip", ".tar", ".tgz", ".tar.gz")
                ):
                    data_path = processor.extract_archive_to_dir(data_path, archive_temp_dir)

                dicom_series = processor.get_valid_series(data_path)
                if len(dicom_series.keys()) >= 1:
                    thinnest_series_paths = (
                        processor.get_thinnest_slice_series_with_windows_paths_(dicom_series)
                    )
                    processor.copy_files_to_directory(thinnest_series_paths, thinnest_series_dir)
                    data_path = thinnest_series_dir
                study_id, series_id = processor.get_study_id_and_series_id(data_path)
            else:
                logger.info(f"Обработка NIFTI исследования")

            try:
                logger.info(f"Чтение финальной серии из {data_path}")
                image = tio.ScalarImage(data_path)

                if image.data.dtype is not torch.float64:
                    affine = image.affine
                    tensor_float64 = image.data.to(torch.float64)
                    image = tio.ScalarImage(tensor=tensor_float64, affine=affine)
            except Exception:
                raise MissingRequiredTagsException()

        ### ---(SLOPE / INTERCEPT) CALCULATION---

        # if study_type is DICOM:
        #     image_reader = sitk.ImageSeriesReader()
        #     file_reader = sitk.ImageFileReader()
        #     file_names = image_reader.GetGDCMSeriesFileNames(data_path)
        #     first_file_path = file_names[0]
        #     file_reader.SetFileName(first_file_path)
        #     file_reader.ReadImageInformation()

        #     intercept = float(file_reader.GetMetaData("0028|1052"))
        #     slope = float(file_reader.GetMetaData("0028|1053"))
        # else:
        #     slope = 1
        #     intercept = 0

        to_orientation = tio.ToOrientation("SLP")
        image = to_orientation(image)
        affine = image.affine

        current = image.spacing
        img_data = image.data.squeeze(0)

        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        # img_data = slope * img_data + intercept

        img_data = 1 * img_data + 0
        hu_min, hu_max = -1000, 1000
        # print(img_data.min(), img_data.max())
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = img_data.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(img_data, current, target)
        img_data = img_data[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        img_data = (((img_data) / 1000)).astype(np.float32)

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480, 480, 240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        # Pad
        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
            value=-1,
        )

        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return {
            "tensor": tensor.numpy(),
            "study_id": study_id,
            "series_id": series_id,
            "warnings": set(processor.warnings)
        }

    def forward(self, image: np.ndarray):
        """Выполнение инференса на предобработанном изображении.
        
        Использует каскад из трех моделей для комплексного анализа:
        1. Vision Transformer для извлечения признаков
        2. Мультиклассовая модель для определения типов патологий
        3. Бинарная модель для определения наличия патологии
        
        Args:
            image (np.ndarray): Предобработанное изображение в формате (240, 480, 480)
            
        Returns:
            dict: Словарь с результатами инференса:
                - embedding: Признаки, извлеченные Vision Transformer
                - multilabel: Вероятности для каждого типа патологии
                - binary: Вероятность наличия патологии
        """
        embedding = self.vit(image[None])[0]
        embedding = torch.nn.functional.normalize(torch.tensor(embedding), dim=-1).numpy()
        
        # Получаем сырые выходы моделей
        raw_multilabel = self.multilabel(embedding)[0]
        raw_binary = self.binary(embedding)[0]
        
        # Применяем сигмоиду к выходам
        result_multilabel = torch.nn.functional.sigmoid(torch.tensor(raw_multilabel)).tolist()
        result_binary = torch.nn.functional.softmax(torch.tensor(raw_binary), dim=-1).tolist()

        return {
            "embedding": embedding,
            "multilabel": result_multilabel,
            "binary": result_binary,
        }

    def postprocess(self, result: dict):
        """Постобработка результатов инференса для получения финального диагноза.
        
        Анализирует результаты моделей и определяет:
        1. Наличие патологии на основе бинарной модели
        2. Наиболее опасный тип патологии по порядку приоритета
        3. Вероятность наличия патологии
        
        Args:
            result (dict): Результаты инференса, содержащий:
                - multilabel: Вероятности для каждого типа патологии
                - binary: Вероятность наличия патологии
                
        Returns:
            tuple: Кортеж из трех элементов:
                - bool: Наличие патологии (True/False)
                - float: Вероятность наличия патологии
                - str: Название наиболее опасной патологии (пустая строка если патологии нет)
        """
        result_multilabel = result["multilabel"]
        result_binary = result["binary"]

        binary_desease_probability = result_binary[0][-1]
        is_desease = binary_desease_probability > self.binary_threshold
        logger.info(f"Binary desease probability: {binary_desease_probability}")
        logger.info(f"Is desease: {is_desease}")

        logger.info(f"Result multilabel: {result_multilabel}")

        most_dangerous_pathology_type = ''
        if is_desease:
            most_dangerous_pathology_type = 'Невозможно определить патологию'
            for desease_name in self.severity_order:
                desease_index = self.name2index[desease_name]
                desease_probability = result_multilabel[0][desease_index]
                if desease_probability > self.deseases2thresholds[desease_name]:
                    most_dangerous_pathology_type = desease_name
                    break

        logger.info(f"Most dangerous pathology type: {most_dangerous_pathology_type}")
        return is_desease, binary_desease_probability, most_dangerous_pathology_type

    def __call__(self, file_path: str, study_name: str, session_id: str, **kwargs):
        """Основной метод обработки медицинского исследования.
        
        Выполняет полный цикл обработки медицинского изображения:
        1. Предобработка изображения
        2. Инференс с использованием ML моделей
        3. Постобработка результатов
        4. Формирование результата в стандартном формате
        5. Очистка временных файлов
        
        Args:
            file_path (str): Путь к файлу медицинского исследования
            study_name (str): Название исследования для логирования
            session_id (str): Идентификатор сессии для отслеживания
            **kwargs: Дополнительные аргументы (не используются)
            
        Returns:
            dict: Результат обработки, содержащий:
                - message: KafkaMessage с результатами анализа
                - df_desease: Список с данными о патологии
                - session_id: Идентификатор сессии
                
        Note:
            В случае ошибки возвращается структура с пустыми полями
            и статусом "Failed" в df_desease
        """
        logger.info(f"[SESSION_ID: {session_id}] | Study name: {study_name} | Processing file {file_path}")

        try:
            start_time_preprocess = self.get_datetime()
            result = self.preprocess(file_path)
            end_time_preprocess = self.get_datetime()

            tensor = result["tensor"]
            study_id = result["study_id"]
            series_id = result["series_id"]
            warnings = result["warnings"]

            start_time_detect = self.get_datetime()
            result = self.forward(tensor)
            end_time_detect = self.get_datetime()

            start_time_postprocess = self.get_datetime()
            is_desease, desease_probability, most_dangerous_pathology_type = (
                self.postprocess(result)
            )
            end_time_postprocess = self.get_datetime()

            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[SESSION_ID: {session_id}] | Study name: {study_name} | Removed file {file_path}")

            message = asdict(
                KafkaMessage(
                    studyIUID=study_id,
                    seriesIUID=series_id,
                    pathologyFlag=is_desease,
                    confidenceLevel=round(desease_probability, 4),
                    most_dangerous_pathology_type=most_dangerous_pathology_type,
                    preprocessStartDT=self.datetime_to_str(start_time_preprocess),
                    preprocessEndDT=self.datetime_to_str(end_time_preprocess),
                    processStartDT=self.datetime_to_str(start_time_detect),
                    processEndDT=self.datetime_to_str(end_time_detect),
                    postprocessStartDT=self.datetime_to_str(start_time_postprocess),
                    postprocessEndDT=self.datetime_to_str(end_time_postprocess),
                )
            )

            result = {
                "message": message,
                "df_desease": [
                    {
                        "path_to_study": study_name,
                        "study_uid": study_id,
                        "series_uid": series_id,
                        "probability_of_pathology": round(desease_probability, 4),
                        "pathology": int(is_desease),
                        "processing_status": "Success",
                        "time_of_processing": round((end_time_postprocess - start_time_preprocess).total_seconds(), 4),
                        "most_dangerous_pathology_type": most_dangerous_pathology_type,
                    }
                ],
                "warnings": '' if len(warnings) == 0 else (study_name + '\n\t' + '\n\t'.join(warnings)),
                "session_id": session_id,
            }
        except Exception as e:
            logger.error(f"[SESSION_ID: {session_id}] | Study name: {study_name} | Error processing file {file_path}: {e}")
            logger.error(traceback.format_exc())
            result = {
                "message": None,
                "df_desease": [
                    {
                        "path_to_study": study_name,
                        "study_uid": '',
                        "series_uid": '',
                        "probability_of_pathology": '',
                        "pathology": '',
                        "processing_status": "Failure",
                        "time_of_processing": '',
                        "most_dangerous_pathology_type": '',
                    }
                ],
                "warnings": study_name + '\n\t' + str(e),
                "session_id": session_id,
            }
            return result

        logger.info(f"[SESSION_ID: {session_id}] | Study name: {study_name} | Processed file {file_path}")
        return result


if __name__ == "__main__":
    """Точка входа в приложение адаптера.
    
    Инициализирует модель и запускает сервер для прослушивания очереди RabbitMQ.
    Обрабатывает медицинские исследования в режиме реального времени.
    
    Конфигурация загружается из файла, указанного в переменной окружения CONFIG_PATH
    или из "../configs/config.yaml" по умолчанию.
    
    Обработка прерывается по Ctrl+C с корректным завершением работы.
    """
    config_path = os.getenv("CONFIG_PATH", "../configs/config.yaml")
    config = yaml.safe_load(Path(config_path).read_text())["adapter"]

    model = Model(config)
    # print(model.bpr.inference(np.random.rand(1, 240, 480, 480).astype(np.float32), np.array([[0.75, 0.75, 1.5]]).astype(np.float32)))

    try:
        broker = RabbitWrapper(config=config["rabbit"])
        broker.listen(pipeline=model)
    except KeyboardInterrupt:
        del model
        logger.info("Завершение работы...")
