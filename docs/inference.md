# Запуск стека и локальный инференс

Этот раздел охватывает каким образом происходит сборка и запуск инференса для внутреннего тестироваия на 400 КТ.

Чтобы запустить решение без сборки смотрите [гайд по быстрому старту](./quick_start.md#1-быстрый-старт-внутреннего-тестирования-из-заранее-подготовленного-образа)

# 1. Локальный запуск инференса

Для того, чтобы запустить инференс локально, необходимо

1. Подготовить окружение в соответствии с [гайдом](./development.md)
2. Полготовить модели и веса в соответствии с [гайдом](./models.md#3-скачивание-исходных-моделей)
2. Подготовить папку с входными данными (архивами)

Ожидается входная директория с zip-архивами исследований:
```
YOUR_INPUT_FOLDER_WITH_ZIPS/
  study_id_1.zip
  study_id_2.zip
  ...
  final_archive.zip
```

3. Далее перейти в директорию `training`:
```bash
cd training
```
4. Установить путь к входным данным в системную переменную `INPUT_FOLDER` (для Windows используйте Git Bash):
```bash
export INPUT_FOLDER="/home/borntowarn/projects/chest-diseases/input"
```

5. Запустить скрипт инференса:
```bash
python inference.py
```

Это запустит скрипт инференса и сохранит результаты в папку `output`.

# 2. Локальная сборка образа и запуск инференса

1. Подготовить виртуальное окружение в соответствии с [гайдом](./development.md)
2. Полготовить модели и веса в соответствии с [гайдом](./models.md#3-скачивание-исходных-моделей)
3. Подготовить папку с входными данными (архивами)

4. Указать системную переменную `YOUR_INPUT_FOLDER_WITH_ZIPS` на абсолютный путь к папке с архивами:

```bash
export YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"
```
или для **Windows** в Git Bash:
```bash
export YOUR_INPUT_FOLDER_WITH_ZIPS="C:\Users\borntowarn\Downloads\Yandex.Disk.Files"
ИЛИ
export YOUR_INPUT_FOLDER_WITH_ZIPS="C:/Users/borntowarn/Downloads/Yandex.Disk.Files"
```

Ваша папка должна иметь структуру вида:
```
YOUR_INPUT_FOLDER_WITH_ZIPS/
├── study_id_1.zip
├── study_id_2.zip
├── ...
├── final_archive.zip
```

5. Запустить скрипт инференса:

**Linux**
```bash
bash ./inference.local.sh
```
**Windows**
```bash
./inference.local.sh
```
Это запустит локальную сборку образа, копирование необходимых моделей, весов и скриптов в образ, создание окружения в контейнере и запуск инференса внутри него по примонтированной папке, которая указана в `YOUR_INPUT_FOLDER_WITH_ZIPS`.