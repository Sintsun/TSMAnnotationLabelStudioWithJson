#!/usr/bin/env python3

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import ctypes

class SimpleTensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = engine.create_execution_context()
        
        # Allocate memory
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        # TensorRT 10 uses new API
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)
            
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(tensor_dtype)
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(cuda_mem))
            
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_w = tensor_shape[-1]
                self.input_h = tensor_shape[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                self.input_name = tensor_name
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                self.output_name = tensor_name
        
        self.engine = engine
    
    def infer(self, input_image):
        self.ctx.push()
        
        # Preprocess
        input_image, origin_h, origin_w = self.preprocess_image(input_image)
        
        # Copy input to GPU
        np.copyto(self.host_inputs[0], input_image.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        
        # Execute inference (TensorRT 10 new API)
        self.context.set_tensor_address(self.input_name, self.cuda_inputs[0])
        self.context.set_tensor_address(self.output_name, self.cuda_outputs[0])
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output back to CPU
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        
        self.ctx.pop()
        
        # Post-process
        output = self.host_outputs[0]
        return self.post_process(output, origin_h, origin_w)
    
    def preprocess_image(self, image):
        h, w, c = image.shape
        r_w = self.input_w / w
        r_h = self.input_h / h
        
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (0.5, 0.5, 0.5))
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        
        return image, h, w
    
    def post_process(self, output, origin_h, origin_w):
        # Simplified post-processing
        print(f"Output shape: {output.shape}")
        print(f"Output range: {output.min():.4f} to {output.max():.4f}")
        return output
    
    def destroy(self):
        self.ctx.pop()

def main():
    print("=== Testing Simple TensorRT Engine ===")
    
    engine_path = "engines/childabuse_28032025_re.engine"
    plugin_library = "engines/libmyplugins.so"
    
    if not os.path.exists(engine_path):
        print(f"Engine file not found: {engine_path}")
        return
    
    if not os.path.exists(plugin_library):
        print(f"Plugin library not found: {plugin_library}")
        return
    
    # Load plugin library
    print("Loading plugin library...")
    ctypes.CDLL(plugin_library)
    print("✓ Plugin library loaded successfully")
    
    try:
        # Create inference engine
        print("Loading TensorRT engine...")
        trt_engine = SimpleTensorRTInference(engine_path)
        print("✓ Engine loaded successfully")
        
        # Create test image
        print("Creating test image...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Execute inference
        print("Executing inference...")
        start_time = time.time()
        output = trt_engine.infer(test_image)
        end_time = time.time()
        
        print(f"✓ Inference successful!")
        print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Output shape: {output.shape}")
        
        trt_engine.destroy()
        print("✅ Test completed! Engine is working properly")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import os
    main()
