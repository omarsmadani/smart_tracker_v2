#!/usr/bin/env bash
# Download pretrained TFLite models for detector and feature extractor.
set -euo pipefail
cd "$(dirname "$0")/../models"

# SSD-MobileNetV2 COCO — Edge TPU + CPU variants from Coral model zoo
curl -L -O https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
curl -L -O https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite
curl -L -O https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

# MobileNetV3-Small (ImageNet) — Edge TPU + CPU variants
curl -L -O https://github.com/google-coral/test_data/raw/master/mobilenet_v3_small_1.0_224_quant_edgetpu.tflite
curl -L -O https://github.com/google-coral/test_data/raw/master/mobilenet_v3_small_1.0_224_quant.tflite

# Rename to match configs/tracker_params.yaml
mv -f mobilenet_v3_small_1.0_224_quant_edgetpu.tflite mobilenet_v3_small_quant_edgetpu.tflite
mv -f mobilenet_v3_small_1.0_224_quant.tflite mobilenet_v3_small_quant.tflite

echo "Models downloaded to $(pwd)"
