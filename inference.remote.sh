#!/bin/bash

# Путь к входным данным в папке (рекомендуется указать АБСОЛЮТНЫЙ путь)
# Папка с zip-файлами должна иметь структуру вида:
# YOUR_INPUT_FOLDER_WITH_ZIPS/
#   study_id_1.zip
#   study_id_2.zip
#   ...
#   final_archive.zip

# НАПРИМЕР: export YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"

# НАПРИМЕР: export YOUR_INPUT_FOLDER_WITH_ZIPS="C:/ВАШ/ПУТЬ/К/ПАПКЕ/С/АРХИВАМИ"

# НАПРИМЕР: export YOUR_INPUT_FOLDER_WITH_ZIPS="C:\ВАШ\ПУТЬ\К\ПАПКЕ\С\АРХИВАМИ"

###########################################################################
#                                                                         #
#   Будьте внимательны, необходимо передать путь к ПАПКЕ, а не к файлу!   #
#                                                                         #
###########################################################################

YOUR_OUTPUT_FOLDER="/$PWD/output"

if [ -z "$YOUR_INPUT_FOLDER_WITH_ZIPS" ]; then
    echo "Ошибка: переменная окружения YOUR_INPUT_FOLDER_WITH_ZIPS не установлена!"
    echo "Пожалуйста, укажите абсолютный путь к папке с архивами:"
    echo "  export YOUR_INPUT_FOLDER_WITH_ZIPS=\"/ваш/путь/к/папке/с/архивами\""
    exit 1
fi

echo "ВАШ ПУТЬ К ПАПКЕ С АРХИВАМИ: $YOUR_INPUT_FOLDER_WITH_ZIPS"
echo "ВАШ ПУТЬ К ПАПКЕ С РЕЗУЛЬТАТАМИ: $YOUR_OUTPUT_FOLDER"

docker run \
    -it \
    --gpus "all" \
    -e INPUT_FOLDER="./input" \
    -v "$YOUR_INPUT_FOLDER_WITH_ZIPS":/training/input \
    -v "$YOUR_OUTPUT_FOLDER":/training/output \
    borntowarn/porcupine-inference