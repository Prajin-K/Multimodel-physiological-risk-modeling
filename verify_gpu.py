import sys
import os

def check_environment():
    print("=" * 60)
    print("ENVIRONMENT & GPU VERIFICATION")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print("-" * 60)

    # 1. PyTorch & CUDA Check
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available    : {cuda_available}")
        if cuda_available:
            print(f"CUDA Version      : {torch.version.cuda}")
            print(f"Device Count      : {torch.cuda.device_count()}")
            device_name = torch.cuda.get_device_name(0)
            print(f"Primary GPU       : {device_name}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total GPU Memory  : {total_mem:.2f} GB")

            # Tensor allocation test on GPU
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print("PyTorch GPU Test  : SUCCESS (Allocated and multiplied 1000x1000 tensor on GPU)")
        else:
            print("PyTorch GPU Test  : FAILED (CUDA not available in PyTorch build)")
    except ImportError as e:
        print(f"PyTorch not installed: {e}")

    print("-" * 60)

    # 2. Keras Check
    try:
        import keras
        print(f"Keras Version     : {keras.__version__}")
        print(f"Keras Backend     : {keras.backend.backend()}")
        
        # Simple Keras tensor operation
        k_tensor = keras.ops.ones((10, 10))
        print(f"Keras Initialization: SUCCESS (Running with {keras.backend.backend()} backend)")
    except Exception as e:
        print(f"Keras check error : {e}")

    print("-" * 60)

    # 3. TensorFlow Check
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        tf_gpus = tf.config.list_physical_devices('GPU')
        print(f"TF Visible GPUs   : {tf_gpus}")
        if not tf_gpus:
            print("Note: TensorFlow on native Windows runs on CPU (native Windows CUDA")
            print("was discontinued after TF 2.10). Keras models will use GPU via PyTorch.")
    except ImportError as e:
        print(f"TensorFlow not installed: {e}")

    print("=" * 60)

if __name__ == "__main__":
    check_environment()
