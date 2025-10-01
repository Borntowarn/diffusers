#!/bin/bash

trtexec --onnx=multilabel_model.onnx \
        --saveEngine=multilabel_model.plan \
        --minShapes=input:1x512 \
        --optShapes=input:4x512 \
        --maxShapes=input:8x512 \
        --verbose \
        --device=0

trtexec --onnx=binary_model.onnx \
        --saveEngine=binary_model.plan \
        --minShapes=input:1x512 \
        --optShapes=input:4x512 \
        --maxShapes=input:8x512 \
        --verbose \
        --device=0