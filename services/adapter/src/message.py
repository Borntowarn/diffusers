"""
Модели данных для сообщений системы анализа медицинских изображений.

Этот модуль содержит структуры данных для передачи результатов
анализа медицинских изображений между компонентами системы.
"""

from dataclasses import dataclass


@dataclass
class KafkaMessage:
    """Структура сообщения с результатами анализа медицинского исследования.
    
    Содержит полную информацию о результатах обработки медицинского изображения,
    включая метаданные исследования, результаты анализа патологий и временные метки
    всех этапов обработки.
    
    Attributes:
        studyIUID (str): Уникальный идентификатор исследования (Study Instance UID)
        seriesIUID (str): Уникальный идентификатор серии (Series Instance UID)
        pathologyFlag (bool): Флаг наличия патологии в исследовании
        confidenceLevel (float): Уровень уверенности в наличии патологии (0.0-1.0)
        most_dangerous_pathology_type (str): Тип наиболее опасной выявленной патологии
        preprocessStartDT (str): Время начала предобработки в ISO формате
        preprocessEndDT (str): Время окончания предобработки в ISO формате
        processStartDT (str): Время начала инференса в ISO формате
        processEndDT (str): Время окончания инференса в ISO формате
        postprocessStartDT (str): Время начала постобработки в ISO формате
        postprocessEndDT (str): Время окончания постобработки в ISO формате
        
    Note:
        Временные метки должны быть в формате ISO 8601: YYYY-MM-DDTHH:MM:SS.ffffff+HHMM
    """
    studyIUID: str  # Уникальный номер исследования
    seriesIUID: str  # Уникальный номер дополнительной серии от ИИ
    pathologyFlag: bool  # Наличие патологии в исследовании
    confidenceLevel: float  # Вероятность наличия патологии во всем исследовании в целом
    most_dangerous_pathology_type: str  # Тип наиболее опасной патологии

    # YYYY-MM-DDThh:mm:ss.sss+hhmm
    preprocessStartDT: str  # Время начала предобработки исследования
    preprocessEndDT: str  # Время окончания предобработки исследования
    processStartDT: str  # Время начала обработки исследования на стороне ИИ сервиса
    processEndDT: str  # Время окончания обработки исследования на стороне ИИ сервиса
    postprocessStartDT: str  # Время начала постпроцессинга исследования
    postprocessEndDT: str  # Время окончания постпроцессинга исследования