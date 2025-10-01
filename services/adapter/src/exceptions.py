class ArchiveExtractionException(Exception):
    """Ошибка чтения архива"""
    def __init__(self, *args):
        super().__init__("Ошибка извлечения файлов из архива. Битый файл")

class IncorrectArchiveTypeException(Exception):
    """Ошибка некорректного типа архива"""
    def __init__(self, *args):
        super().__init__("Некорректный формат архива")


class EmptyDicomSeriesException(Exception):
    """Ошибка отсутствия валидной серии в исследовании"""
    def __init__(self, *args):
        super().__init__("Не удалось найти валидную серию для обработки")


class InvalidDicomFormatException(Exception):
    """Ошибка некорреткного формата DICOM исследования"""
    def __init__(self, *args):
        super().__init__("Некорретный формат DICOM исследования (не КТ)")


class NotChestBodyPartException(Exception):
    """Ошибка отсутствие тега BodyPartExamined"""
    def __init__(self, *args):
        super().__init__("Область исследования не является ОГК")


class NotChestBodyPartException(Exception):
    """Ошибка отсутствие тега BodyPartExamined"""
    def __init__(self, *args):
        super().__init__("Исследование не является грудной клеткой")

class MissingRequiredTagsException(Exception):
    """Ошибка отсутствие обязательных тегов"""
    def __init__(self, *args):
        tags = [
            "(7FE0, 0010): Pixel Data",
            "(0028, 0100): Bits Allocated",
            "(0028, 0011): Columns",
            "(0028, 0010): Rows",
        ]
        super().__init__("Отсутствуют обязательные теги: " + ", ".join(tags))