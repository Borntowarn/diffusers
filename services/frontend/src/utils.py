"""
Утилиты для работы с файлами в веб-интерфейсе.

Этот модуль содержит вспомогательные функции для кодирования
и декодирования файлов в base64 формате для передачи через
веб-интерфейс.
"""

import base64

def encode_file(file: str):
    """Кодирование файла в base64 строку.
    
    Args:
        file (str): Путь к файлу для кодирования
        
    Returns:
        str: Base64 закодированная строка
    """
    str_ = base64.b64encode(open(file, "rb").read())
    return str_.decode("utf-8")

def decode_file(file_bytes: str):
    """Декодирование base64 строки в байты.
    
    Args:
        file_bytes (str): Base64 закодированная строка
        
    Returns:
        bytes: Декодированные байты
    """
    return base64.b64decode(file_bytes)

