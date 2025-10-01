# Модели и веса: загрузка, обновление, конвертация

В проекте модели разворачиваются через NVIDIA Triton Inference Server. Готовые артефакты (ONNX или TensorRT plan) кладутся в `model_repository_running/`.

### 1. Заранее подготовленные образы 
Готовые образы уже содержат все необходимые модели и могут быть использованы для запуска Triton и всех сервисов без дополнительных действий.

Аналогично и инференс - заранее подготовленный образ уже содержит все необходимые веса для простого запуска инференса.

ℹ️ Этот путь удобен для быстрой проверки без ручной подготовки артефактов.


### 2. Для локального развертывания Triton

1. Скачайте `model_repository_running.zip` из [Яндекс.Диска](https://disk.yandex.ru/d/nq0x0-Ivx93VJw)

2. Распакуйте архив в директорию чтобы в корне проекта появилась директория `model_repository_running` со следующей структурой:

⚠️ Сохраняйте точную структуру каталогов и имена файлов — Triton валидирует их на старте.
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

3. Ваш репозиторий моделей готов к запуску Triton.

### 3. Скачивание исходных моделей
Необходимо скачать папку `weights.zip` с исходными моделями и весами в директорию `training/weights` из [Яндекс.Диска](https://disk.yandex.ru/d/nq0x0-Ivx93VJw) чтобы получилась следующая структура:
```
training
│   └── weights
│       ├── CT-RATE
│       │   ├── ClassifierHead_LiPro_V2.pt
│       │   ├── ProjectionVIT_Base_V2.pt
│       │   ├── ProjectionVIT_LiPro_V2.pt
│       │   ├── ProjectionVIT_VocabFine_V2.pt
│       │   ├── model_binary.pth
│       │   ├── model_multilabel.pth
│       │   └── models
│       ├── bpr
│       │   ├── config.json
│       │   ├── inference-settings.json
│       │   ├── model.pt
│       │   └── reference.xlsx
│       └── faiss_database
│           ├── all
│           └── train
```


### 4. Конвертация PyTorch → ONNX
1. Скачайте исходные `.pt` модели в директорию `training/weights` (по умолчанию) ([по гайду](#3-скачивание-исходных-моделей))
2. Перейдите в директорию `convertation`:
```bash
cd convertation
```
3. Выберите модель для конвертации в ONNX:
```bash
cd vit
ИЛИ
cd head
```
4. Запустите скрипт для конвертации в ONNX:
```bash
python torch2onnx.py
```

### 5. Конвертация ONNX → TensorRT
1. Скачайте исходные `.pt` модели в директорию `training/weights` (по умолчанию) ([по гайду](#3-скачивание-исходных-моделей))
2. Перейдите в директорию `convertation`:
```bash
cd convertation
```
3. Запустите контейнер для конвертации, который содержит утилиту `trtexec`:
```bash
sh run.sh
```
4. Внутри контейнера для конкретной модели:
```bash
cd vit
ИЛИ
cd head
```
4. Запустите скрипт для конвертации в TensorRT:
```bash
sh convert2trt.sh
```