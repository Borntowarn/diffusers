import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

def find_training_root(file_path) -> Path | None:
    """
    Находит путь к родительской папке 'training' начиная с file_path.
    Возвращает абсолютный путь к папке 'training', либо None, если не найдено.
    """
    path = Path(file_path).resolve()
    for parent in [path] + list(path.parents):
        if parent.name == 'training':
            return parent
    return None

# Определяем корневую папку проекта (training)
ROOT_DIR = find_training_root(__file__)

# Путь к .env всегда ищем в корне training
ENV_PATH = ROOT_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

# Папка для данных всегда training/data
CACHE_DIR = ROOT_DIR / '.cache'

HF_TOKEN = os.getenv('HF_TOKEN')

REPO_CT_RATE_ID = 'ibrahimhamamci/CT-RATE'

DATA_DIR = ROOT_DIR / 'data' 
CT_RATE_DIR = ROOT_DIR / 'data' / 'CT-RATE'

WEIGHTS_DIR = ROOT_DIR / 'weights'
CT_RATE_WEIGHTS_DIR = WEIGHTS_DIR / 'CT-RATE'
VISTA3D_WEIGHTS_DIR = WEIGHTS_DIR / 'Vista3D'
BPR_WEIGHTS_DIR = WEIGHTS_DIR / 'bpr'
