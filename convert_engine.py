#!/usr/bin/env python3

import tensorrt as trt
import numpy as np
import os

def build_engine_from_onnx(onnx_path, engine_path, max_workspace_size=1<<30):
    """
    從 ONNX 模型建構 TensorRT 引擎
    """
    # 創建 TensorRT 記錄器
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 創建建構器
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # 設置最大工作空間大小
    config.max_workspace_size = max_workspace_size
    
    # 創建網路
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 解析 ONNX 模型
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"載入 ONNX 模型: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ONNX 解析失敗')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print('建構 TensorRT 引擎...')
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print('引擎建構失敗')
        return False
    
    # 保存引擎
    print(f"保存引擎到: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print('引擎建構完成！')
    return True

def main():
    # 檢查是否有 ONNX 檔案
    onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
    
    if not onnx_files:
        print("未找到 ONNX 檔案")
        print("請先將您的 YOLOv7 模型轉換為 ONNX 格式")
        print("或者提供 ONNX 檔案路徑")
        return
    
    # 使用第一個找到的 ONNX 檔案
    onnx_path = onnx_files[0]
    engine_path = "yolov7_new.engine"
    
    print(f"找到 ONNX 檔案: {onnx_path}")
    
    if build_engine_from_onnx(onnx_path, engine_path):
        print(f"新引擎檔案已創建: {engine_path}")
        print("請將此檔案複製到 engine_plugin/ 目錄")
    else:
        print("引擎建構失敗")

if __name__ == "__main__":
    main()
