# Запуск стека для развертывания сервиса

Этот раздел охватывает каким образом производится локальная сборка и запуск сервисов для запуска решения.

Чтобы запустить решение без сборки смотрите [гайд по быстрому старту](./quick_start.md#2-быстрый-старт-сервисов-из-заранее-подготовленных-образов)

# 1. Локальная сборка образов и запуск сервисов

Для того, чтобы запустить сервисы локально, необходимо
1. Подготовить репозиторий с моделями и весами для запуска TritonServer [гайду](./models.md#2-для-локального-развертывания-triton).

⚠️ Структура `model_repository_running` должна совпадать с образцом ниже, иначе Triton не поднимет модели.

Финальная структура должна быть следующей:

```
...
├── convertation
├── model_repository_running
│   ├── binary_onnx
│   │   ├── 1
│   │   │   └── model.onnx
│   │   └── config.pbtxt
│   ├── bpr_model
│   │   ├── 1
│   │   │   ├── bpreg
│   │   │   ├── model.py
│   │   │   ├── requirements.txt
│   │   │   └── weights
│   │   └── config.pbtxt
│   ├── multilabel_onnx
│   │   ├── 1
│   │   │   └── model.onnx
│   │   └── config.pbtxt
│   └── vit_onnx
│       ├── 1
│       │   └── model.onnx
│       └── config.pbtxt
├── services
...
```

2. Запустить `start.local.sh` для локальной сборки образов и запуска сервисов

Это запустит процесс подгрузки образа RabbitMQ, а так же сборку сервисов TritonServer, Adapter и Frontend.


3. Чтобы остановить сервисы, необходимо запустить `stop.local.sh`:
```bash
sh ./stop.local.sh
```

Основные переменные окружения:
- `TRITON_URL` — grpc ручка, `tritonserver:8001`
- `RABBIT_URL` — amqp ручка, `amqp://guest:guest@rabbitmq:5672/`
- `INPUT_TOPIC` / `OUTPUT_TOPIC` - топики для обмена сообщениями между компонентами
- `CONFIG_PATH` — путь к `./configs/config.yaml` для `frontend` и `adapter`

# 2. Запуск сервисов без контейнеров

Для того, чтобы запустить сервисы без контейнеров, необходимо:
1. Подготовить окружение в соответствии с [гайдом](./development.md)
2. Подготовить репозиторий с моделями для запуска TritonServer по [гайду](./models.md#2-для-локального-развертывания-triton).
3. Запустить сервисы TritonServer и RabbitMQ:
```bash
docker compose -f compose.local.yaml --profile utils up -d
```
4. Перейти в директорию `services` и запустить сервисы Adapter и Frontend:
```bash
cd services
python -m adapter
python -m frontend
```