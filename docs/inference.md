# Запуск стека и локальный инференс

Этот раздел охватывает каким образом происходит запуск решения для внутреннего тестироваия на 400 КТ

### 1. Быстрый старт из заранее подготовленных образов (Предпочтительный вариант)

Мы собрали для вас готовые образы и залили их в Docker Hub, для того, чтобы не зависеть от платформы и локальной сборки образов и быстро начать работу.
Чтобы запустить инференс для папки с архивами, необходимо:

1. Внутри файла `inference.remote.sh` необходимо заменить переменную `YOUR_INPUT_FOLDER_WITH_ZIPS` на абсолютный путь к папке с архивами.

⚠️ Обязательно указывайте АБСОЛЮТНЫЙ путь к исходной папке с архивами.

Таким образом ваш скрипт для развертывания решения будет выглядеть следующим образом:
```bash
#!/bin/bash

# Путь к входным данным в папке (рекомендуется указать АБСОЛЮТНЫЙ путь)
# Папка с zip-файлами должна иметь структуру вида:
# YOUR_INPUT_FOLDER_WITH_ZIPS/
#   study_id_1.zip
#   study_id_2.zip
#   ...
#   final_archive.zip
# НАПРИМЕР: YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"
YOUR_INPUT_FOLDER_WITH_ZIPS="ВАШ ПУТЬ К ПАПКЕ С АРХИВАМИ"
YOUR_OUTPUT_FOLDER="./output"

docker run \
    -it \
    --gpus "all" \
    -e INPUT_FOLDER="./input" \
    -v "$YOUR_INPUT_FOLDER_WITH_ZIPS":/training/input \
    -v "$YOUR_OUTPUT_FOLDER":/training/output \
    borntowarn/porcupine-inference
```

-  Папки `INPUT_FOLDER="./input"` - внутренняя папка, в которую будут монтироваться входные данные из папки `YOUR_INPUT_FOLDER_WITH_ZIPS`
-  Папка `YOUR_OUTPUT_FOLDER` - папка, в которую будут сохраняться результаты инференса: файлы `output.xlsx` и `warnings_and_errors.txt`. Она автоматически примонтируется в локальной файловой системе и будет доступна.

2. Запустить скрипт:
```bash
bash ./inference.remote.sh
```

Это запустит скачивание контейнера из Docker Hub и автоматически запустит скрипт инференса внутри него по примонтированной папке, которая указана в `YOUR_INPUT_FOLDER_WITH_ZIPS`.
Он пробежит по всем элементам папки `YOUR_INPUT_FOLDER_WITH_ZIPS`, выполнит для них инференс и сохранит папку `output` 2 файла:
- `output.xlsx` - результаты инференса в формате Excel
- `warnings_and_errors.txt` - ошибки чтения файлов и предупреждения в формате txt.


### 2. Локальный запуск инференса

Для того, чтобы запустить инференс локально, необходимо

1. Подготовить окружение в соответствии с [гайдом](./development.md)
2. Полготовить модели и веса в соответствии с [гайдом](./models.md) (пункт 3. Для разработки)
2. Подготовить папку с входными данными (архивами)

Ожидается входная директория с zip-архивами исследований:
```
YOUR_INPUT_FOLDER_WITH_ZIPS/
  study_id_1.zip
  study_id_2.zip
  ...
  final_archive.zip
```

3. Далее перейти в директорию `training` и установить зависимости:
```bash
cd training
```
4. Установить путь к входным данным в системную переменную `INPUT_FOLDER`:
```bash
export INPUT_FOLDER="/home/borntowarn/projects/chest-diseases/input"
```
Для **Windows** используйте следующую команду в командной строке :
```bash
set INPUT_FOLDER=C:/Users/ваш_пользователь/projects/chest-diseases/input
```

⚠️ На Windows избегайте пробелов в пути или экранируйте их кавычками.
5. Запустить скрипт инференса:
```bash
python inference.py
```

Это запустит скрипт инференса и сохранит результаты в папку `output`.

### 3. Локальная сборка образа и запуск инференса

1. Подготовить окружение в соответствии с [гайдом](./development.md)
2. Полготовить модели и веса в соответствии с [гайдом](./models.md) (пункт 3. Для разработки)
3. Подготовить папку с входными данными (архивами)

4. Внутри файла `inference.local.sh` необходимо заменить переменную `YOUR_INPUT_FOLDER_WITH_ZIPS` на абсолютный путь к папке с архивами.

⚠️ Контейнер ожидает примонтированную директорию по `-v "$YOUR_INPUT_FOLDER_WITH_ZIPS":/training/input` — путь слева должен существовать.

Таким образом ваш скрипт для развертывания решения будет выглядеть следующим образом:
```bash
#!/bin/bash

# Путь к входным данным в папке (рекомендуется указать АБСОЛЮТНЫЙ путь)
# Папка с zip-файлами должна иметь структуру вида:
# YOUR_INPUT_FOLDER_WITH_ZIPS/
#   study_id_1.zip
#   study_id_2.zip
#   ...
#   final_archive.zip
# НАПРИМЕР: YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"
YOUR_INPUT_FOLDER_WITH_ZIPS="ВАШ ПУТЬ К ПАПКЕ С АРХИВАМИ"
YOUR_OUTPUT_FOLDER="./output"

docker build -t inference -f ./Dockerfile.inference.yaml .

docker run \
    -it \
    --gpus "all" \
    -e INPUT_FOLDER="./input" \
    -v "$YOUR_INPUT_FOLDER_WITH_ZIPS":/training/input \
    -v "$YOUR_OUTPUT_FOLDER":/training/output \
    inference
```

-  Папки `INPUT_FOLDER="./input"` - внутренняя папка, в которую будут монтироваться входные данные из папки `YOUR_INPUT_FOLDER_WITH_ZIPS`
-  Папка `YOUR_OUTPUT_FOLDER` - папка, в которую будут сохраняться результаты инференса: файлы `output.xlsx` и `warnings_and_errors.txt`. Она автоматически примонтируется в локальной файловой системе и будет доступна.

5. Запустить скрипт инференса:
```bash
sh ./inference.local.sh
```

Это запустит локальную сборку образа, копирование необходимых моделей, весов и скриптов в образ, создание окружения в контейнере и запуск инференса внутри него по примонтированной папке, которая указана в `YOUR_INPUT_FOLDER_WITH_ZIPS`.