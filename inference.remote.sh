#!/bin/bash

# Путь к входным данным в папке (рекомендуется указать АБСОЛЮТНЫЙ путь)
# Папка с zip-файлами должна иметь структуру вида:
# YOUR_INPUT_FOLDER_WITH_ZIPS/
#   study_id_1.zip
#   study_id_2.zip
#   ...
#   final_archive.zip
# НАПРИМЕР: YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"
# НАПРИМЕР: YOUR_INPUT_FOLDER_WITH_ZIPS="C:/ВАШ/ПУТЬ/К/ПАПКЕ/С/АРХИВАМИ"

YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"
YOUR_OUTPUT_FOLDER="./output"

docker run \
    -it \
    --gpus "all" \
    -e INPUT_FOLDER="./input" \
    -v "$YOUR_INPUT_FOLDER_WITH_ZIPS":/training/input \
    -v "$YOUR_OUTPUT_FOLDER":/training/output \
    borntowarn/porcupine-inference