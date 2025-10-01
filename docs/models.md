## Модели и веса: загрузка, обновление, конвертация

В проекте модели разворачиваются через NVIDIA Triton Inference Server. Готовые артефакты (ONNX или TensorRT plan) кладутся в `model_repository_running/`.

### Где хранить модели
- `model_repository_running/` — распакованные модели, готовые к запуску Triton
- `storage/` — можно использовать для хранения промежуточных артефактов и весов тренинга

### Загрузка готовых моделей
Если образы приходят с уже включёнными моделями, ничего делать не нужно. Если вы хотите подменить модели:
1. Остановите стек
2. Замените содержимое `model_repository_running/<model_name>` на новую версию
3. Запустите стек снова

### Добавление новой модели
1. Подготовьте `1/` с `model.onnx` или `model.plan`
2. Создайте корректный `config.pbtxt`
3. Поместите директорию модели в `model_repository_running/`
4. Перезапустите Triton/стек

### Конвертация PyTorch → ONNX → TensorRT
См. подробности в [training.md](./training.md). Коротко:
```bash
# Экспорт в ONNX
python convertation/<model_family>/torch2onnx.py --weights best.pth --output model.onnx

# Проверка ONNX
python convertation/<model_family>/test_onnx.py --onnx model.onnx

# Конвертация в TensorRT внутри контейнера
cd convertation && sh run.sh
cd <model_family> && sh convert2trt.sh /abs/path/model.onnx /abs/path/model.plan
```

### Версионирование моделей
Triton поддерживает версионирование через поддиректории `1/`, `2/`, `3/` и т. д. Вы можете хранить несколько версий и указывать активную через конфиг или путь.

### Рекомендации по `config.pbtxt`
- Укажите `platform` в зависимости от артефакта: `onnxruntime_onnx` или `tensorrt_plan`
- Настройте `max_batch_size` и `dynamic_batching` под ваш сценарий
- Для GPU добавьте `instance_group [ { kind: KIND_GPU } ]`


