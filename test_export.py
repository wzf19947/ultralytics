#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify ONNX export fix for YOLO models."""

import os
os.chdir('/home/workspace/Feng/ultralytics')

from ultralytics import YOLO

# Test YOLO11s export
print("=" * 60)
print("Testing YOLO11s ONNX Export")
print("=" * 60)

model_path = "./Dronemodel/yolo11s_20260506/weights/best.pt"

if os.path.exists(model_path):
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully: {model_path}")
        
        # Attempt ONNX export
        results = model.export(format='onnx', imgsz=640, simplify=True)
        print(f"✓ ONNX export successful!")
        print(f"✓ Exported model: {results}")
        
    except Exception as e:
        print(f"✗ Error during export: {type(e).__name__}")
        print(f"✗ Error message: {str(e)}")
else:
    print(f"✗ Model file not found: {model_path}")

print("=" * 60)
