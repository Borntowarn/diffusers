# Установка и системные требования

### Поддерживаемые платформы
- Linux
- WSL2
- Windows
- Требуется GPU NVIDIA для ускорения

### Предварительные требования
- Docker
- NVIDIA драйвер на хосте
- NVIDIA Container Toolkit
- Наличие GPU с памятью:
    - не менее 8Гб для запуска inference (см [гайд](./inference.md))
    - не менее 12Гб для запуска сервисов (см [гайд](./start services.md)). Памяти занимает больше, потому что она выделяется сразу на максимальный batch_size в конфигурации моделей

### Порядок действи для полноценного окружения для R&D

1. Проверьте доступность GPU:
```bash
nvidia-smi
```

2. Клонирование репозитория
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

4. Далее необходимо установить дополнительные модули как зависимости для запуска моделей
```bash
pip install -e training/CT-CLIP/CT_CLIP
pip install -e training/CT-CLIP/transformer_maskgit
```

5. На данный момент создан токен для скачивания исходных моделей и данных с huggingface, однако он истекает через 30 дней. 

Для продолжения работы необходимо создать новый токен и авторизоваться в исходном репозитории модели https://huggingface.co/datasets/ibrahimhamamci/CT-RATE, который используется для скачивания данных и заменить токен в файле `training/.env` на новый.
```bash
HF_TOKEN=your_huggingface_token
```
Для скачивания исходных моделей и данных необходимо указать имя в репозитории модели: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

6. Далее необходимо установить дополнительные модули как зависимости для запуска моделей
```bash
pip install -e training/CT-CLIP/CT_CLIP
pip install -e training/CT-CLIP/transformer_maskgit
```

7. Если вы хотите запускать ноутбуки для различных тестов, то необходимо установить ipykernel в venv
```bash
pip install ipykernel
```

8. Ваше основное окружение для проекта готово, теперь можно приступать к работе.

9. Если вы хотите запускать обучение MLP модели, то необходимо установить дополнительный `requirements.txt` в директории `training/classification`
```bash
pip install -r requirements.txt
```