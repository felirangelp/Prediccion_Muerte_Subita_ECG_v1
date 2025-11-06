"""
Optimizaciones para GPU M1 (Metal Performance Shaders)
Incluye batching eficiente, procesamiento paralelo y monitoreo
"""

import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple
import time
from contextlib import contextmanager

def configure_tensorflow_m1():
    """
    Configurar TensorFlow para usar GPU Metal en M1
    """
    try:
        # Configurar para usar Metal
        if hasattr(tf.config, 'list_physical_devices'):
            devices = tf.config.list_physical_devices()
            gpu_devices = [d for d in devices if 'GPU' in d.name or 'Metal' in d.name]
            
            if gpu_devices:
                # Habilitar crecimiento de memoria
                for gpu in gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
                
                return True, gpu_devices[0].name
    except:
        pass
    
    return False, None

def get_optimal_batch_size(model_type: str, available_memory_gb: float = 16.0) -> int:
    """
    Obtener tamaño de batch óptimo para M1
    
    Args:
        model_type: Tipo de modelo ('sparse', 'hierarchical', 'hybrid')
        available_memory_gb: Memoria disponible en GB
    """
    # Tamaños de batch recomendados para M1
    base_batch_sizes = {
        'sparse': 32,
        'hierarchical': 16,
        'hybrid': 8
    }
    
    base_size = base_batch_sizes.get(model_type, 16)
    
    # Ajustar según memoria disponible
    if available_memory_gb >= 32:
        multiplier = 2.0
    elif available_memory_gb >= 16:
        multiplier = 1.0
    else:
        multiplier = 0.5
    
    return int(base_size * multiplier)

@contextmanager
def gpu_monitor():
    """
    Context manager para monitorear uso de GPU
    """
    start_time = time.time()
    start_memory = None
    
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 ** 3  # GB
    except:
        pass
    
    yield
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if start_memory:
        try:
            end_memory = process.memory_info().rss / 1024 ** 3
            memory_used = end_memory - start_memory
            print(f"⏱️  Tiempo: {elapsed:.2f}s | Memoria: {memory_used:.2f} GB")
        except:
            print(f"⏱️  Tiempo: {elapsed:.2f}s")

def process_batch_parallel(signals: List[np.ndarray], 
                           process_fn, 
                           batch_size: int = 32,
                           use_gpu: bool = True) -> List:
    """
    Procesar señales en batches paralelos optimizados para M1
    
    Args:
        signals: Lista de señales
        process_fn: Función de procesamiento
        batch_size: Tamaño de batch
        use_gpu: Si usar GPU
    """
    results = []
    
    # Dividir en batches
    n_batches = (len(signals) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(signals))
        batch = signals[start_idx:end_idx]
        
        # Procesar batch
        if use_gpu:
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                batch_results = [process_fn(signal) for signal in batch]
        else:
            batch_results = [process_fn(signal) for signal in batch]
        
        results.extend(batch_results)
    
    return results

def optimize_tensorflow_operations():
    """
    Optimizar operaciones de TensorFlow para M1
    """
    # Configurar optimizaciones
    tf.config.optimizer.set_jit(True)  # XLA JIT compilation
    
    # Configurar threading
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

def clear_gpu_cache():
    """
    Limpiar caché de GPU
    """
    try:
        import gc
        gc.collect()
        
        # Limpiar caché de TensorFlow
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
    except:
        pass

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Optimizaciones GPU M1")
    print("=" * 50)
    
    gpu_available, gpu_name = configure_tensorflow_m1()
    print(f"GPU disponible: {gpu_available}")
    if gpu_name:
        print(f"Dispositivo: {gpu_name}")
    
    # Tamaños de batch óptimos
    print(f"\n📊 Tamaños de batch recomendados:")
    print(f"   Sparse: {get_optimal_batch_size('sparse')}")
    print(f"   Hierarchical: {get_optimal_batch_size('hierarchical')}")
    print(f"   Hybrid: {get_optimal_batch_size('hybrid')}")

