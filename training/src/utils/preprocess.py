import shutil
import tarfile
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pydicom
import SimpleITK as sitk
import torch
from bpreg.inference.inference_model import InferenceModel
from pydicom import dcmread
from src import BPR_WEIGHTS_DIR, logger
from src.utils.exceptions import *

# Параметра окна для контроля констрастности исследования
WINDOW_CENTER_RANGE = (30, 50)
WINDOW_WIDTH_RANGE = (300, 500)


warnings.filterwarnings(
    "ignore", message="Invalid value for VR UI", category=UserWarning
)


def is_leaf_dir(path):
    """Папка последнего уровня: все дочерние элементы — только файлы"""
    return all(child.is_file() for child in Path(path).iterdir())


class InferencePreprocessor:
    def __init__(self):
        self.warnings = []

    def check_series_body_part(self, series):
        """Функция применяет модель BodyPartReg для определения области тела
        на полученной серии

        Args:
            series (_type_): _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Определение области тела на серии")
        files_paths = [Path(dcm.filename) for dcm in series]

        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            with TemporaryDirectory() as archive_temp_dir:
                self.copy_files_to_directory(files_paths, archive_temp_dir)

                model = InferenceModel(str(BPR_WEIGHTS_DIR) + "/")

                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(archive_temp_dir)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                dicom_spacings = image.GetSpacing()
                dicom_array = sitk.GetArrayFromImage(image)

                bodypart_metadata = model.npy2json(
                    X_=dicom_array,
                    output_path="",  # No data need to be saved in json file
                    pixel_spacings=dicom_spacings,
                    axis_ordering=(2, 1, 0),  # Base sitk.GetArrayFromImage ordering
                )
                tag = bodypart_metadata["body part examined tag"]
        except Exception as e:
            logger.error(f"Ошибка при определении области тела на серии: {e}")
            tag = ""

        logger.info(f"Область тела на серии: {tag}")
        del model
        torch.cuda.empty_cache()
        return tag

    def extract_archive_to_dir(self, archive_path, extract_dir):
        """
        Функция для извлечения архива (zip или tar) в указанную папку.

        Args:
            archive_path (str or Path): путь к архиву
            extract_dir (str or Path): путь к папке, куда извлекать

        Returns:
            Path: путь к папке с извлечёнными файлами
        """
        logger.info(f"Распаковка архива {archive_path} в {extract_dir}")
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if str(archive_path).endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif str(archive_path).endswith((".tar", ".tgz", ".tar.gz")):
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                raise IncorrectArchiveTypeException()
        except Exception:
            raise ArchiveExtractionException()

        logger.info(f"Архив {archive_path} распакован в {extract_dir}")
        return extract_dir

    def get_study_id_and_series_id(self, data_path):
        """Извлечение StudyInstanceUID и SeriesInstanceUID из DICOM файлов.

        Сканирует директорию с DICOM файлами и извлекает уникальные идентификаторы
        исследования и серии из первого доступного файла.

        Args:
            data_path (str): Путь к директории с DICOM файлами

        Returns:
            tuple: Кортеж из двух строк:
                - str: StudyInstanceUID исследования
                - str: SeriesInstanceUID серии

        Note:
            Если идентификаторы не найдены, возвращаются пустые строки
        """
        study_id = "Отсутствует"
        series_id = "Отсутствует"
        data_path = Path(data_path)
        for file_path in data_path.iterdir():
            if file_path.is_file():
                try:
                    dcm = dcmread(file_path, stop_before_pixels=True)
                    study_id = getattr(dcm, "StudyInstanceUID", "Отсутствует")
                    series_id = getattr(dcm, "SeriesInstanceUID", "Отсутствует")
                    if study_id == "Отсутствует" or series_id == "Отсутствует":
                        logger.warning(
                            f"В файле {file_path} не найден StudyInstanceUID или SeriesInstanceUID"
                        )
                        continue
                    logger.info(
                        f"Найден StudyInstanceUID: {study_id} и SeriesInstanceUID: {series_id} в файле {file_path}"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Ошибка при чтении файла {file_path}: {e}")
                    continue

        if study_id == "Отсутствует":
            self.warnings.append(f"В файлах не найден StudyInstanceUID")

        if series_id == "Отсутствует":
            self.warnings.append(f"В файлах не найден SeriesInstanceUID")

        return study_id, series_id

    def get_valid_series(self, directory):
        """Функция для поиска серий в указанной директории DICOM исследования

        Args:
            directory (str или Path): путь к исследованию

        Returns:
            defaultdict: словарь с сериями
        """
        # Словарь для группировки КТ-серий
        logger.info(f"Чтение КТ-серий из {directory}")
        series = defaultdict(list)
        series_modality_dict = {}
        series_body_part_dict = {}

        directory = Path(directory)

        # Сканируем директорию с помощью pathlib
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    # Читаем DICOM-файл
                    dcm = pydicom.dcmread(str(file_path))

                    modality = getattr(dcm, "Modality", "")
                    body_part = getattr(dcm, "BodyPartExamined", "")
                    series_uid = getattr(dcm, "SeriesInstanceUID", None)

                    if series_uid:
                        if modality.upper() == "CT":
                            series_modality_dict[series_uid] = modality
                        if body_part.upper() == "CHEST":
                            series_body_part_dict[series_uid] = body_part

                        series[series_uid].append(dcm)
                except Exception as e:
                    logger.warning(f"Ошибка чтения файла {file_path}: {e}")
                    self.warnings.append(f"Часть файлов не прочитана")

        series_to_remove = []
        for series_uid, files in series.items():
            if series_modality_dict.get(series_uid, "").upper() == "CT" and (
                series_body_part_dict.get(series_uid, "").upper() == "CHEST"
                or self.check_series_body_part(files) == "CHEST"
            ):
                series_modality_dict[series_uid] = "CT"
                series_body_part_dict[series_uid] = "CHEST"
                series[series_uid] = sorted(
                    files,
                    key=lambda x: (
                        x.ImagePositionPatient[2]
                        if hasattr(x, "ImagePositionPatient")
                        else 0
                    ),
                )
            else:
                self.warnings.append(
                    f"Серия {series_uid} не валидна. Модальность: {series_modality_dict.get(series_uid, 'Пусто')}, Часть тела: {series_body_part_dict.get(series_uid, 'Пусто')}"
                )
                series_to_remove.append(series_uid)

        logger.info(f"series_modality_dict: {series_modality_dict}")
        logger.info(f"series_body_part_dict: {series_body_part_dict}")

        for series_uid in series_to_remove:
            series.pop(series_uid)

        if not series:
            if "CT" not in list(series_modality_dict.values()) and "CHEST" in list(
                series_body_part_dict.values()
            ):
                raise InvalidDicomFormatException()
            if "CHEST" not in list(series_body_part_dict.values()) and "CT" in list(
                series_modality_dict.values()
            ):
                raise NotChestBodyPartException()
            raise EmptyDicomSeriesException()

        
        logger.info(
            f"Прочитаны КТ-серии из {directory}; найдено {len(series)} серий КТ ОГК"
        )
        return series

    def copy_files_to_directory(self, file_list, destination_directory):
        """
        Создает папку и копирует в нее все файлы из указанного списка.

        Args:
            file_list (list): Список строк с полными путями к файлам для копирования.
            destination_directory (str): Полный путь к папке, куда будут скопированы файлы.
        """
        try:
            logger.info(f"Копирование файлов в {destination_directory}")
            Path(destination_directory).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Ошибка при создании папки: {e}")
            return

        for file_path in file_list:
            if file_path.exists():
                try:
                    shutil.copy(file_path, destination_directory)
                except shutil.Error as e:
                    logger.error(f"Ошибка при копировании файла '{file_path}': {e}")
            else:
                logger.error(f"Файл '{file_path}' не найден и не был скопирован.")

        logger.info(f"Файлы скопированы в {destination_directory}")

    def get_thinnest_slice_series_with_windows_paths_(
        self, series, wc_range=WINDOW_CENTER_RANGE, ww_range=WINDOW_WIDTH_RANGE
    ):
        """Модификация функции get_thinnest_slice_series_paths, позволяющая проводить
        дополнительный отбор исследований по уровню контрастности и яркости.

        Args:
            series (defaultdict): Серии полученные из директории
            wc_range (tuple, optional): Диапазон значений DICOM тега WindowCenter. Defaults to (30, 50).
            ww_range (tuple, optional): Диапазон значений DICOM тега WindowWidth. Defaults to (300, 500).

        Returns:
            list: Список путей ко всем .dcm файлам серии
        """
        # Находим серию с минимальной толщиной среза и подходящими WindowCenter/WindowWidth
        logger.info(
            f"Поиск серии с минимальной толщиной среза с учётом оконных параметров"
        )
        min_slice_thickness = float("inf")
        candidate_series = []

        # Собираем серии с минимальной толщиной среза
        for series_uid, files in series.items():
            if files:  # Проверяем, что серия не пуста
                first_dcm = files[0]
                if hasattr(first_dcm, "SliceThickness"):
                    slice_thickness = float(first_dcm.SliceThickness)
                    if slice_thickness < min_slice_thickness:
                        min_slice_thickness = slice_thickness
                        candidate_series = [(series_uid, files)]
                    elif slice_thickness == min_slice_thickness:
                        candidate_series.append((series_uid, files))

        # Если есть кандидаты, фильтруем по WindowCenter и WindowWidth
        thinnest_series_uid = None
        thinnest_series_files = []
        wc_min, wc_max = wc_range
        ww_min, ww_max = ww_range

        for series_uid, files in candidate_series:
            first_dcm = files[0]
            # Проверяем WindowCenter и WindowWidth
            if hasattr(first_dcm, "WindowCenter") and hasattr(first_dcm, "WindowWidth"):
                window_center = float(
                    first_dcm.WindowCenter[0]
                    if isinstance(first_dcm.WindowCenter, pydicom.multival.MultiValue)
                    else first_dcm.WindowCenter
                )
                window_width = float(
                    first_dcm.WindowWidth[0]
                    if isinstance(first_dcm.WindowWidth, pydicom.multival.MultiValue)
                    else first_dcm.WindowWidth
                )
                if (wc_min <= window_center <= wc_max) and (
                    ww_min <= window_width <= ww_max
                ):
                    thinnest_series_uid = series_uid
                    thinnest_series_files = files
                    break  # Берем первую подходящую серию

        # Если не найдено серий с подходящими WindowCenter/WindowWidth, берем первую с минимальной толщиной
        if not thinnest_series_uid and candidate_series:
            thinnest_series_uid, thinnest_series_files = candidate_series[0]

        logger.info(f"Определена минимальная толщина среза: {min_slice_thickness} мм")

        # Возвращаем список путей к файлам выбранной серии
        if thinnest_series_uid:
            if hasattr(thinnest_series_files[0], "WindowCenter") and hasattr(
                thinnest_series_files[0], "WindowWidth"
            ):
                wc = thinnest_series_files[0].WindowCenter
                ww = thinnest_series_files[0].WindowWidth
                logger.info(f"Параметры окна — WindowCenter: {wc}, WindowWidth: {ww}")
            final_series = thinnest_series_files
        else:
            logger.warning(
                "Не найдено КТ-серий с атрибутом SliceThickness. Берем первую серию"
            )
            final_series = list(series.values())[0]
        
        slope = getattr(final_series[0], "RescaleSlope", 1)
        intercept = getattr(final_series[0], "RescaleIntercept", 0)

        if not hasattr(final_series[0], "RescaleSlope"):
            self.warnings.append(f"Не найден атрибут RescaleSlope. Установлено значение 1")
        else:
            logger.info(f"Найден атрибут RescaleSlope: {slope}")

        if not hasattr(final_series[0], "RescaleIntercept"):
            self.warnings.append(f"Не найден атрибут RescaleIntercept. Установлено значение 0")
        else:
            logger.info(f"Найден атрибут RescaleIntercept: {intercept}")

        self.check_for_missing_and_duplicate_values(final_series)
        return [Path(dcm.filename) for dcm in final_series]

    def check_for_missing_and_duplicate_values(self, sequence: tuple | list) -> bool:
        """
        Проверяет, является ли последовательность чисел строго возрастающей
        и непрерывной (без пропусков).

        Условия:
        1. Все числа в последовательности должны быть уникальными (без повторов).
        2. Разница между каждым соседним элементом должна быть строго равна 1.

        Args:
            sequence: Кортеж или список чисел.

        Returns:
            True, если последовательность соответствует обоим условиям, False иначе.
        """
        logger.info(f"Проверка на пропуски и дубликаты в последовательности")
        sequence = sorted([getattr(x, "InstanceNumber", 0) for x in sequence])

        missed_slices_percentage = round(
            (1 - len(set(sequence)) / max(sequence)) * 100, 2
        )

        if len(sequence) != len(set(sequence)):
            self.warnings.append(f"Последовательность содержит дубликаты")

        # 3. Проверка на пропуски и строгий порядок (Условие 2)
        # Используем функцию zip для итерации по парам (текущее, следующее)
        for current, next_val in zip(sequence[:-1], sequence[1:]):
            if next_val - current != 1:
                self.warnings.append(f"Последовательность содержит пропуски")

        if missed_slices_percentage > 0:
            self.warnings.append(
                f"Процент пропущенных срезов: {missed_slices_percentage}%"
            )


class TrainingPreprocessor:

    @staticmethod
    def extract_archive_to_dir(archive_path, extract_dir):
        """
        Функция для извлечения архива (zip или tar) в указанную папку.

        Args:
            archive_path (str or Path): путь к архиву
            extract_dir (str or Path): путь к папке, куда извлекать

        Returns:
            Path: путь к папке с извлечёнными файлами
        """
        logger.info(f"Распаковка архива {archive_path} в {extract_dir}")
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Если архив содержит папку, а нам нужно, чтобы все файлы оказались непосредственно в extract_dir,
        # то после распаковки переносим содержимое из вложенной папки в extract_dir.
        def _move_contents_to_dir(src_dir, dst_dir):
            for item in Path(src_dir).iterdir():
                target = Path(dst_dir) / item.name
                if item.is_dir():
                    shutil.move(str(item), str(target))
                else:
                    shutil.move(str(item), str(dst_dir))

        if str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        elif str(archive_path).endswith((".tar", ".tgz", ".tar.gz")):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Неподдерживаемый формат архива: {archive_path}")

        # Проверяем, не появилась ли единственная вложенная папка после распаковки
        extracted_items = list(Path(extract_dir).iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            inner_dir = extracted_items[0]
            _move_contents_to_dir(inner_dir, extract_dir)
            inner_dir.rmdir()

        logger.info(f"Архив {archive_path} распакован в {extract_dir}")
        return extract_dir

    @staticmethod
    def get_study_id_and_series_id(data_path):
        """Извлечение StudyInstanceUID и SeriesInstanceUID из DICOM файлов.

        Сканирует директорию с DICOM файлами и извлекает уникальные идентификаторы
        исследования и серии из первого доступного файла.

        Args:
            data_path (str): Путь к директории с DICOM файлами

        Returns:
            tuple: Кортеж из двух строк:
                - str: StudyInstanceUID исследования
                - str: SeriesInstanceUID серии

        Note:
            Если идентификаторы не найдены, возвращаются пустые строки
        """
        study_id = ""
        series_id = ""
        data_path = Path(data_path)
        for file_path in data_path.iterdir():
            if file_path.is_file():
                try:
                    dcm = dcmread(file_path, stop_before_pixels=True)
                    study_id = getattr(dcm, "StudyInstanceUID", "")
                    series_id = getattr(dcm, "SeriesInstanceUID", "")
                    if study_id == "" or series_id == "":
                        logger.warning(
                            f"В файле {file_path} не найден StudyInstanceUID или SeriesInstanceUID"
                        )
                        continue
                    logger.info(
                        f"Найден StudyInstanceUID: {study_id} и SeriesInstanceUID: {series_id} в файле {file_path}"
                    )
                    break
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла {file_path}: {e}")
                    continue
        return study_id, series_id

    @staticmethod
    def read_ct_series(directory):
        """Функция для поиска серий в указанной директории DICOM исследования

        Args:
            directory (str или Path): путь к исследованию

        Returns:
            defaultdict: словарь с сериями
        """
        # Словарь для группировки КТ-серий
        logger.info(f"Чтение КТ-серий из {directory}")
        series = defaultdict(list)
        TARGET_BODY_PART = "CHEST"

        directory = Path(directory)

        # Сканируем директорию с помощью pathlib
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    # Читаем DICOM-файл
                    dcm = pydicom.dcmread(str(file_path))
                    # Проверяем, что это КТ
                    if hasattr(dcm, "Modality") and dcm.Modality == "CT":
                        series_uid = dcm.SeriesInstanceUID
                        series[series_uid].append(dcm)
                except Exception as e:
                    logger.error(f"Ошибка чтения файла {file_path}: {e}")

        # Сортируем файлы в каждой серии по позиции среза (ImagePositionPatient[2])
        for series_uid, files in series.items():
            series[series_uid] = sorted(
                files,
                key=lambda x: (
                    x.ImagePositionPatient[2]
                    if hasattr(x, "ImagePositionPatient")
                    else 0
                ),
            )

        logger.info(f"Прочитаны КТ-серии из {directory}; найдено {len(series)} серий")
        return series

    @staticmethod
    def copy_files_to_directory(file_list, destination_directory):
        """
        Создает папку и копирует в нее все файлы из указанного списка.

        Args:
            file_list (list): Список строк с полными путями к файлам для копирования.
            destination_directory (str): Полный путь к папке, куда будут скопированы файлы.
        """
        try:
            logger.info(f"Копирование файлов в {destination_directory}")
            Path(destination_directory).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Ошибка при создании папки: {e}")
            return

        for file_path in file_list:
            if file_path.exists():
                try:
                    shutil.copy(file_path, destination_directory)
                except shutil.Error as e:
                    logger.error(f"Ошибка при копировании файла '{file_path}': {e}")
            else:
                logger.error(f"Файл '{file_path}' не найден и не был скопирован.")

        logger.info(f"Файлы скопированы в {destination_directory}")

    @staticmethod
    def get_thinnest_slice_series_paths(series):
        """Функция для поиска серии с миннимальной толщиной среза
        (при равных выбирается первое вхождение)

        Args:
            series (defaultdict): Серии полученные из директории

        Returns:
            list: Список путей ко всем .dcm файлам серии
        """
        logger.info(f"Поиск серии с минимальной толщиной среза")
        min_slice_thickness = float("inf")
        thinnest_series_uid = None
        thinnest_series_files = []

        for series_uid, files in series.items():
            if files:  # Проверяем, что серия не пуста
                first_dcm = files[0]
                if hasattr(first_dcm, "SliceThickness"):
                    slice_thickness = float(first_dcm.SliceThickness)
                    if slice_thickness < min_slice_thickness:
                        min_slice_thickness = slice_thickness
                        thinnest_series_uid = series_uid
                        thinnest_series_files = files

        # Возвращаем список путей к файлам выбранной серии
        if thinnest_series_uid:
            logger.info(
                f"Серия с минимальной толщиной среза ({min_slice_thickness} мм): {thinnest_series_uid}"
            )
            return [Path(dcm.filename) for dcm in thinnest_series_files]
        else:
            logger.error("Не найдено КТ-серий с атрибутом SliceThickness")
            return []

    @staticmethod
    def get_thinnest_slice_series_with_windows_paths_(
        series, wc_range=WINDOW_CENTER_RANGE, ww_range=WINDOW_WIDTH_RANGE
    ):
        """Модификация функции get_thinnest_slice_series_paths, позволяющая проводить
        дополнительный отбор исследований по уровню контрастности и яркости.

        Args:
            series (defaultdict): Серии полученные из директории
            wc_range (tuple, optional): Диапазон значений DICOM тега WindowCenter. Defaults to (30, 50).
            ww_range (tuple, optional): Диапазон значений DICOM тега WindowWidth. Defaults to (300, 500).

        Returns:
            list: Список путей ко всем .dcm файлам серии
        """
        # Находим серию с минимальной толщиной среза и подходящими WindowCenter/WindowWidth
        logger.info(
            f"Поиск серии с минимальной толщиной среза с учётом оконных параметров"
        )
        min_slice_thickness = float("inf")
        candidate_series = []

        # Собираем серии с минимальной толщиной среза
        for series_uid, files in series.items():
            if files:  # Проверяем, что серия не пуста
                first_dcm = files[0]
                if hasattr(first_dcm, "SliceThickness"):
                    slice_thickness = float(first_dcm.SliceThickness)
                    if slice_thickness < min_slice_thickness:
                        min_slice_thickness = slice_thickness
                        candidate_series = [(series_uid, files)]
                    elif slice_thickness == min_slice_thickness:
                        candidate_series.append((series_uid, files))

        # Если есть кандидаты, фильтруем по WindowCenter и WindowWidth
        thinnest_series_uid = None
        thinnest_series_files = []
        wc_min, wc_max = wc_range
        ww_min, ww_max = ww_range

        for series_uid, files in candidate_series:
            first_dcm = files[0]
            # Проверяем WindowCenter и WindowWidth
            if hasattr(first_dcm, "WindowCenter") and hasattr(first_dcm, "WindowWidth"):
                window_center = float(
                    first_dcm.WindowCenter[0]
                    if isinstance(first_dcm.WindowCenter, pydicom.multival.MultiValue)
                    else first_dcm.WindowCenter
                )
                window_width = float(
                    first_dcm.WindowWidth[0]
                    if isinstance(first_dcm.WindowWidth, pydicom.multival.MultiValue)
                    else first_dcm.WindowWidth
                )
                if (wc_min <= window_center <= wc_max) and (
                    ww_min <= window_width <= ww_max
                ):
                    thinnest_series_uid = series_uid
                    thinnest_series_files = files
                    break  # Берем первую подходящую серию

        # Если не найдено серий с подходящими WindowCenter/WindowWidth, берем первую с минимальной толщиной
        if not thinnest_series_uid and candidate_series:
            thinnest_series_uid, thinnest_series_files = candidate_series[0]

        logger.info(f"Определена минимальная толщина среза: {min_slice_thickness} мм")

        # Возвращаем список путей к файлам выбранной серии
        if thinnest_series_uid:
            if hasattr(thinnest_series_files[0], "WindowCenter") and hasattr(
                thinnest_series_files[0], "WindowWidth"
            ):
                wc = thinnest_series_files[0].WindowCenter
                ww = thinnest_series_files[0].WindowWidth
                logger.info(f"Параметры окна — WindowCenter: {wc}, WindowWidth: {ww}")
            return [Path(dcm.filename) for dcm in thinnest_series_files]
        else:
            logger.warning(
                "Не найдено КТ-серий с атрибутом SliceThickness. Берем первую серию"
            )
            return [Path(dcm.filename) for dcm in list(series.values())[0]]
