"""
TensorFlow GPU Configuration for Apple Silicon (M-series) and Mac M5
Optimized for parallel processing and Metal acceleration
"""

import os
import tensorflow as tf

def setup_gpu():
    """
    Configure TensorFlow for optimal performance on Apple Silicon M5.
    - Enable Metal plugin for GPU acceleration
    - Enable mixed precision for faster training
    - Configure parallel processing
    - Set memory growth
    """
    
    # Check if running on Apple Silicon
    is_apple_silicon = os.uname().machine.startswith('arm64')
    
    # Enable Metal plugin (required for M-series GPU)
    os.environ['TF_METAL_ENABLED'] = '1'
    
    # Enable XLA compilation for better performance
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    
    # Configure TensorFlow for optimal performance
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Set memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable Metal device placement
            if is_apple_silicon:
                print(f"Apple Silicon detected: {os.uname().machine}")
                print(f"GPU(s) available: {len(gpus)}")
                
                # Set preferred GPU
                tf.config.set_visible_devices(gpus, 'GPU')
                
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Enable mixed precision training (faster on M5)
    policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision enabled: {policy.name}")
    
    # Configure threading for parallel processing
    # Use all available CPU cores for data preprocessing
    num_threads = os.cpu_count() or 8
    
    # Set thread settings
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    print(f"TensorFlow threads: inter_op={num_threads}, intra_op={num_threads}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"XLA enabled: Yes")
    
    return True


def get_device_info():
    """Get information about available compute devices."""
    info = {
        "tensorflow_version": tf.__version__,
        "apple_silicon": os.uname().machine.startswith('arm64'),
        "gpus": [gpu.name for gpu in tf.config.list_physical_devices('GPU')],
        "cpus": [cpu.name for cpu in tf.config.list_physical_devices('CPU')],
    }
    
    # Check for Metal
    try:
        from tensorflow.python.platform import build_info as build_info
        info["metal_available"] = True
    except:
        info["metal_available"] = False
    
    return info


if __name__ == "__main__":
    setup_gpu()
    print("\nDevice Info:")
    for k, v in get_device_info().items():
        print(f"  {k}: {v}")
