## Обучение и дообучение моделей

Документ описывает подходы к обучению/дообучению и конвертации моделей для последующего инференса в Triton.

### Выбор режима
- Обучение «с нуля» — если у вас есть размеченный датасет и архитектуры
- Дообучение (fine-tuning) — если есть предобученные веса
- Только конвертация — если веса уже получены, но их нужно подготовить под Triton

### Структура и ключевые директории
- `training/` — код/скрипты для обучения (ознакомьтесь с содержимым вашей ветки/модуля)
- `convertation/` — инструменты для экспорта в ONNX/TensorRT и проверок
- `model_repository_running/` — финальная выкладка моделей для Triton (распакованные модели с `config.pbtxt`)

### Базовые требования окружения для обучения
Рекомендуется отдельное окружение Python с CUDA и PyTorch. Пример:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# при необходимости установите совместимые версии torch/torchvision/torchaudio с CUDA
```

### Общий шаблон обучения (примерный)
Псевдокод процесса обучения может выглядеть так:
```python
# псевдокод — замените на ваш реальный тренинг-скрипт
from pathlib import Path

def train_model(config_path: str, output_dir: str):
    # 1) загрузить конфиг, датасеты, разделить train/val
    # 2) инициализировать модель и оптимизатор
    # 3) цикл эпох: обучение, валидация, сохранение лучших весов
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # save weights to output_dir

if __name__ == "__main__":
    train_model("./configs/config.yaml", "./storage/training_runs/run_001")
```

Сохраните лучшие веса, журнал обучения и метрики в `./storage/training_runs/<run_id>`.

### Экспорт в ONNX
В каталоге `convertation/` для разных архитектур есть примеры `torch2onnx.py`. Общая идея:
```bash
cd convertation/<model_family>
python torch2onnx.py \
  --weights /abs/path/to/best.pth \
  --output /abs/path/to/model.onnx \
  --img-size 224 224 # пример, зависит от модели
```

Проверьте корректность ONNX:
```bash
python test_onnx.py --onnx /abs/path/to/model.onnx
```

### Конвертация в TensorRT
Можно использовать контейнер из `convertation/run.sh`:
```bash
cd convertation
sh run.sh
# внутри контейнера для конкретной модели:
cd <model_family>
sh convert2trt.sh /abs/path/to/model.onnx /abs/path/to/model.plan
```

### Подготовка репозитория моделей Triton
Структура для одной модели в `model_repository_running/`:
```
model_repository_running/
  my_model/
    1/
      model.onnx | model.plan
    config.pbtxt
```

Минимальный `config.pbtxt` (пример, замените на ваш):
```text
name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  { name: "input", data_type: TYPE_FP32, dims: [3, 224, 224] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [1000] }
]
```

Для TensorRT:
```text
name: "my_trt_model"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  { name: "input", data_type: TYPE_FP32, dims: [3, 224, 224] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [1000] }
]
instance_group [ { kind: KIND_GPU } ]
dynamic_batching { preferred_batch_size: [4, 8] }
```

После выкладки модели перезапустите сервис Triton или пересоберите образ (см. quickstart/run).


