<p align="center">
  <strong>ЛИДЕРЫ ЦИФРОВОЙ ТРАНСФОРМАЦИИ 2025</strong>
</p>

---

<p align="center">
  СЕРВИС ДЛЯ ВЫЯВЛЕНИЯ КОМПЬЮТЕРНЫХ ТОМОГРАФИЙ ОРГАНОВ ГРУДНОЙ КЛЕТКИ БЕЗ ПАТОЛОГИЙ
</p>

---

<img alt="Титул ридми" src="assets/titul.jpg">

## *Состав команды "Диффузоры"*
1. Козлов Михаил (@borntowarn) - MLE/MLOps
2. Карпов Даниил (@Free4ky) - MLE/MLOps

## Оглавление
1. [Задание](#1)
2. [Решение](#2)
3. [Результат разработки](#3)
4. [Уникальность решения](#4)
5. [Стек](#5)
6. [Развертывание и тестирование](#6)
7. [Структура проекта](#7)
8. [Ссылки](#8)

## <a name="1">Задание</a>

Разработать программное обеспечение на основе
искусственного интеллекта для автоматической классификации КТ исследований ОГК на два класса: «без патологии» и «с патологией».

## <a name="2">Решение</a>

<p align="center">
<img  alt="" src="assets/architecture_scheme.jpg">
</p>
<p align="center">
<img  alt="" src="assets/binary_model.jpg">
</p>
<p align="center">
<img  alt="" src="assets/multilabel_arch.jpg">
</p>

Система поднимает NVIDIA Triton Inference Server с моделями из `model_repository_running`, использует `RabbitMQ` для очередей задач и адаптер, который маршрутизирует запросы к моделям. Для ручного тестирования доступен веб-интерфейс на `Gradio` (порт `7860`). Поддерживается пакетная и офлайн-обработка архивов с исследованиями в формате `zip`.

Ключевые режимы работы:
- Локальный одиночный инференс через скрипты.
- Полный микросервисный стек через Docker Compose.

## <a name="3">Результат разработки</a>
- Запуск Triton с готовым `model_repository_running`.
- Адаптер для gRPC-взаимодействия и очередей `RabbitMQ`.
- Веб-интерфейс для интерактивных испытаний.
- Скрипты для пакетного офлайн-инференса по папке архивов.

## <a name="4">Уникальность решения</a>
- Фокус на производительности GPU за счет Triton (динамическое батчирование, эффективная утилизация).
- Единый стек для локального R&D и продакшен-подобного окружения.
- Простая смена/добавление моделей через `model_repository_running`.

## <a name="5">Стек</a>
`Python`, `NVIDIA Triton`, `Docker/Compose`, `RabbitMQ`, `Gradio`, `ONNX/TensorRT`

## <a name="6">Развертывание и тестирование</a>

### Предварительные требования
- Linux (проверено в WSL2), `Docker 24+`, `Docker Compose v2`.
- NVIDIA драйвер и `NVIDIA Container Toolkit`.

Проверка GPU:
```bash
nvidia-smi
```

### Быстрый старт (готовые контейнеры)
1) Опционально установите `LOGURU_LEVEL`.
2) Запустите стек:
```bash
bash ./start.sh
```
3) Откройте интерфейс: `http://localhost:7860`
4) Остановите стек:
```bash
bash ./stop.sh
```

Открытые порты:
- Triton: `8000` (HTTP), `8001` (gRPC), `8002` (Metrics)
- RabbitMQ: `5672` (AMQP), `15672` (Management UI)
- Frontend (Gradio): `7860`

### Локальный инференс по архивам
Подготовьте входную папку:
```
YOUR_INPUT_FOLDER_WITH_ZIPS/
  study_id_1.zip
  study_id_2.zip
  ...
  final_archive.zip
```

Запуск из Docker Hub:
```bash
bash ./inference.remote.sh
```

Локальная сборка и запуск:
```bash
bash ./inference.local.sh
```

Переменные в скриптах:
- `YOUR_INPUT_FOLDER_WITH_ZIPS` — абсолютный путь к данным
- `YOUR_OUTPUT_FOLDER` — папка результатов (по умолчанию `./output`)

### Локальная сборка образов и запуск стека
```bash
docker compose -f compose.local.yaml --profile base up -d --build
```

Остановка и очистка:
```bash
docker compose -f compose.local.yaml --profile base down
```

Проверка логов:
```bash
docker compose -f compose.local.yaml logs -f tritonserver | cat
docker compose -f compose.local.yaml logs -f adapter | cat
docker compose -f compose.local.yaml logs -f frontend | cat
```

Основные переменные окружения:
- `TRITON_URL` — например, `tritonserver:8001`
- `RABBIT_URL` — `amqp://user:pass@rabbitmq:5672/`
- `INPUT_TOPIC` / `OUTPUT_TOPIC`
- `CONFIG_PATH` — путь к `./configs/config.yaml`

## <a name="7">Структура проекта</a>

```
.                                   # Корневая директория проекта
├── configs                         # Папка с конфигурационными файлами
│   └── config.yaml                 # Главный YAML-файл с настройками для сервисов
├── convertation                    # Модули и скрипты для конвертации моделей
│   ├── head                        # Конвертация моделей "head" (головы нейросети)
│   │   ├── convert2trt.sh          # Bash-скрипт для преобразования модели в формат TensorRT
│   │   └── torch2onnx.py           # Python-скрипт для конвертации PyTorch-модели в ONNX
│   ├── vit                         # Конвертация моделей Vision Transformer (ViT)
│   │   ├── transformer_maskgit     # Папка с файлами и кодом для модели MaskGIT
│   │   ├── convert2trt.sh          # Bash-скрипт для конвертации ViT в TensorRT
│   │   ├── test_onnx.py            # Тестирование работоспособности ONNX-модели ViT
│   │   └── torch2onnx.py           # Конвертация ViT из PyTorch в ONNX
│   ├── Dockerfile                  # Dockerfile для контейнеризации процесса конвертации
│   ├── README.md                   # Документация по использованию конвертации
│   ├── compose.yaml                # Docker Compose файл для запуска конвертации в контейнере
│   └── run.sh                      # Скрипт для запуска контейнера конвертации
├── model_repository_running        # Папка с готовыми моделями для Triton Inference Server (тут должны быть распакованные модели для локальной сборки)
├── services                        # Каталог с микросервисами проекта
│   ├── adapter                     # Сервис-адаптер для взаимодействия между компонентами
│   │   ├── src                     # Исходный код адаптера
│   │   ├── Dockerfile              # Dockerfile для сборки контейнера адаптера
│   │   ├── __init__.py             # Инициализация Python-пакета адаптера
│   │   ├── __main__.py             # Главная точка входа адаптера (запуск сервиса)
│   │   └── requirements.txt        # Список Python-зависимостей адаптера
│   └── frontend                    # Сервис пользовательского интерфейса (фронтенд)
│       ├── src                     # Исходный код фронтенда
│       ├── Dockerfile              # Dockerfile для сборки контейнера фронтенда
│       ├── __init__.py             # Инициализация Python-пакета фронтенда
│       ├── __main__.py             # Главная точка входа фронтенда (запуск сервиса)
│       └── requirements.txt        # Список Python-зависимостей фронтенда
├── storage                         # Общий том/данные для сервисов (монтируется в контейнеры)
├── training                        # Папка для обучения, инференса и всевозможных тестов моделей и данных. README внутри
├── visualisation                   # Визуализация эмбеддингов в UI
│   ├── Dockerfile                  # Образ для окружения визуализации
│   └── viz_embeddings_fiftyone.py  # Скрипт визуализации эмбеддингов (FiftyOne)
├── Dockerfile.inference.yaml       # Образ одиночного инференса (docker run)
├── Dockerfile.triton.yaml          # Образ Triton с установкой зависимостей модели BPR
├── README.md                       # Основной файл с документацией по проекту
├── compose.local.yaml              # Docker Compose для локального билда и запуска всех сервисов (локальная сборка образов)
├── compose.remote.yaml             # Docker Compose для запуска всех сервисов из готовых образов Docker Hub
├── inference.local.sh              # Локальная сборка и запуск инференса
├── inference.remote.sh             # Запуск инференса из Docker Hub (подготовленный образ, предпочтительнее)
├── requirements.txt                # Питон-зависимости проекта для локальных запусков
├── start.local.sh                  # Локальная сборка образов и запуск сервисов группы base (tritonserver, rabbitmq, adapter, frontend)
├── start.remote.sh                 # Загрузка подготовленных образов и запуск сервисов группы base (tritonserver, rabbitmq, adapter, frontend)
├── stop.local.sh                   # Остановка сервисов группы base запущенных локально (tritonserver, rabbitmq, adapter, frontend)
└── stop.remote.sh                  # Остановка сервисов группы base запущенных из Docker Hub (tritonserver, rabbitmq, adapter, frontend)
```            

Ключевые файлы:
- `compose.local.yaml` — сервисы `tritonserver`, `rabbitmq`, `adapter`, `frontend` (профили: `base`, `utils`).
- `Dockerfile.triton.yaml` — сборка Triton и копирование `model_repository_running`.
- `Dockerfile.inference.yaml` — одиночный инференс для папки архивов.
- `services/adapter/Dockerfile` — адаптер (`TRITON_URL`, `RABBIT_URL`, `CONFIG_PATH`).
- `services/frontend/Dockerfile` — фронтенд (`RABBIT_URL`, `CONFIG_PATH`).

## <a name="8">Ссылки</a>
- RabbitMQ UI: `http://localhost:15672` (`guest/guest`)
- Frontend (Gradio): `http://localhost:7860`
