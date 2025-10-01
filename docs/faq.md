## FAQ и устранение неполадок

### UI не открывается на http://localhost:7860
- Проверьте, что стек запущен (`docker ps` показывает контейнеры)
- Посмотрите логи фронтенда: `docker compose -f compose.local.yaml logs -f frontend | cat`

### Triton не видит модели
- Убедитесь, что директория `model_repository_running/` смонтирована и содержит корректную структуру
- Проверьте `config.pbtxt` и наличие папки версии `1/`
- Посмотрите логи: `docker compose -f compose.local.yaml logs -f tritonserver | cat`

### Нет GPU внутри контейнера
- Убедитесь, что установлен NVIDIA Container Toolkit и демон Docker перезапущен
- Проверьте запуск с `--gpus all` или профиль/compose с GPU

### Ошибки подключения к RabbitMQ
- Проверьте `RABBIT_URL`
- Откройте `http://localhost:15672` и удостоверьтесь, что брокер активен

### Где задаются пути и параметры
- Конфигурация — `./configs/config.yaml`
- Переменные окружения — см. [run.md](./run.md)

### Как выполнить офлайн-инференс по архивам
- Используйте `inference.remote.sh` или `inference.local.sh`
- Подготовьте `YOUR_INPUT_FOLDER_WITH_ZIPS` с `.zip` исследованиями


