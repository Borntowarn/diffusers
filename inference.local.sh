#!/bin/bash

# НА ВСЯКИЙ СЛУЧАЙ ПУТЬ УКАЗЫВАТЬ АБСОЛЮТНЫЙ
YOUR_INPUT_FOLDER_WITH_ZIPS="/home/borntowarn/projects/chest-diseases/input"
YOUR_OUTPUT_FOLDER="./output"

docker build -t inference -f ./Dockerfile.inference.yaml .

docker run \
    -it \
    --gpus "all" \
    -e INPUT_FOLDER="./input" \
    -v "$YOUR_INPUT_FOLDER_WITH_ZIPS":/training/input \
    -v "$YOUR_OUTPUT_FOLDER":/training/output \
    inference