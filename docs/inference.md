## Запуск стека и локальный инференс

Этот раздел охватывает:
- Полный стек через Docker Compose
- Локальный пакетный инференс по папке с архивами
- Переменные окружения и логи

### Полный стек: Docker Compose

Готовые образы:
```bash
bash ./start.remote.sh
```

Локальная сборка:
```bash
bash ./start.local.sh
```

Остановка:
```bash
bash ./stop.remote.sh   # если запускали remote
bash ./stop.local.sh    # если запускали local
```

Проверка логов по сервисам:
```bash
docker compose -f compose.local.yaml logs -f tritonserver | cat
docker compose -f compose.local.yaml logs -f adapter | cat
docker compose -f compose.local.yaml logs -f frontend | cat
```

Основные переменные окружения (используются в Dockerfile/compose и сервисах):
- `TRITON_URL` — например `tritonserver:8001`
- `RABBIT_URL` — например `amqp://user:pass@rabbitmq:5672/`
- `INPUT_TOPIC` / `OUTPUT_TOPIC`
- `CONFIG_PATH` — путь к `./configs/config.yaml`

### Локальный инференс по архивам

Ожидается входная директория с zip-архивами исследований:
```
YOUR_INPUT_FOLDER_WITH_ZIPS/
  study_id_1.zip
  study_id_2.zip
  ...
  final_archive.zip
```

Запуск на готовых образах:
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

