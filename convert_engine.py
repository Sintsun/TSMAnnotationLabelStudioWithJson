#!/usr/bin/env python3

import tensorrt as trt
import numpy as np
import os

def build_engine_from_onnx(onnx_path, engine_path, max_workspace_size=1<<30):
    """
    Build TensorRT engine from ONNX model
    """
    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Set maximum workspace size
    config.max_workspace_size = max_workspace_size
    
    # Create network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"Loading ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ONNX parsing failed')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print('Building TensorRT engine...')
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print('Engine building failed')
        return False
    
    # Save engine
    print(f"Saving engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print('Engine building completed!')
    return True

def main():
    # Check if ONNX files exist
    onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
    
    if not onnx_files:
        print("No ONNX files found")
        print("Please convert your YOLOv7 model to ONNX format first")
        print("Or provide ONNX file path")
        return
    
    # Use the first found ONNX file
    onnx_path = onnx_files[0]
    engine_path = "yolov7_new.engine"
    
    print(f"Found ONNX file: {onnx_path}")
    
    if build_engine_from_onnx(onnx_path, engine_path):
        print(f"New engine file created: {engine_path}")
        print("Please copy this file to engine_plugin/ directory")
    else:
        print("Engine building failed")

if __name__ == "__main__":
    main()
