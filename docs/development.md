# Установка и системные требования

### 1. Поддерживаемые платформы
- Linux
- WSL2
- Windows
- Требуется GPU NVIDIA для ускорения

### 2. Предварительные требования
- Docker
- NVIDIA драйвер на хосте
- NVIDIA Container Toolkit
- Наличие GPU с памятью:
    - не менее 8Гб для запуска inference (см [гайд](./inference.md))
    - не менее 12Гб для запуска сервисов (см [гайд](./start services.md)). Памяти занимает больше, потому что она выделяется сразу на максимальный batch_size в конфигурации моделей

### 3. Порядок действи для полноценного окружения для R&D

1. Проверьте доступность GPU:
```bash
nvidia-smi
```

2. Клонирование репозитория:
```bash
git clone https://github.com/Borntowarn/diffusers
cd diffusers
```

3. Для локальных экспериментов без контейнеров можно установить зависимости Python:
```bash
conda create --prefix ./venv python=3.11
conda activate ./venv
pip install -r requirements.txt
```

ℹ️ Рекомендуется отдельное окружение `conda`/`venv`, чтобы избежать конфликтов зависимостей.

4. Далее необходимо установить дополнительные модули как зависимости для запуска моделей:
```bash
pip install -e training/CT-CLIP/CT_CLIP
pip install -e training/CT-CLIP/transformer_maskgit
```

5. Если вы хотите запускать ноутбуки для различных тестов, то необходимо установить ipykernel в venv:
```bash
pip install ipykernel
```

6. Ваше основное окружение для проекта готово, теперь можно приступать к работе.

7. Если вы хотите запускать обучение MLP модели, то необходимо установить дополнительный `requirements.txt` в директории `training/classification`:
```bash
pip install -r requirements.txt
```

8. Для загрузки исходных весов скачайте папку `initial_weights.zip` с исходными моделями и весами в директорию `training/weights` из [Яндекс.Диска](https://disk.yandex.ru/d/nq0x0-Ivx93VJw). Подробнее в [гайде](./models.md#3-для-разработки)

