#!/bin/bash

trtexec --onnx=vit_lipro.onnx \
        --saveEngine=vit_lipro.plan \
        --minShapes=input:1x1x240x480x480 \
        --optShapes=input:2x1x240x480x480 \
        --maxShapes=input:4x1x240x480x480 \
        --verbose \
        --device=0