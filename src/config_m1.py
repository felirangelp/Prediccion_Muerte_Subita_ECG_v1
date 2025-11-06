"""
Configuración y optimización para MacBook Pro M1 (Apple Silicon)
Optimiza el uso de GPU Metal y recursos del sistema
"""

import os
import platform
import warnings
from typing import Optional

def check_m1_system() -> bool:
    """
    Verifica si el sistema es un Mac con Apple Silicon (M1/M2/M3)
    
    Returns:
        True si es Mac con Apple Silicon, False en caso contrario
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def configure_tensorflow_m1() -> dict:
    """
    Configura TensorFlow para usar GPU Metal en Mac M1
    
    Returns:
        Diccionario con información de configuración
    """
    config = {
        'm1_detected': check_m1_system(),
        'tensorflow_backend': None,
        'gpu_available': False,
        'gpu_device': None
    }
    
    if not config['m1_detected']:
        warnings.warn("No se detectó Mac M1. Las optimizaciones M1 no se aplicarán.")
        return config
    
    try:
        import tensorflow as tf
        
        # Configurar para usar Metal Performance Shaders
        if hasattr(tf.config, 'list_physical_devices'):
            devices = tf.config.list_physical_devices()
            gpu_devices = [d for d in devices if 'GPU' in d.name or 'Metal' in d.name]
            
            if gpu_devices:
                config['gpu_available'] = True
                config['gpu_device'] = gpu_devices[0].name
                
                # Configurar crecimiento de memoria para GPU
                try:
                    for gpu in gpu_devices:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
                
                # Configurar backend
                os.environ['TF_METAL_DEVICE_LIST'] = '0'
                config['tensorflow_backend'] = 'Metal (GPU)'
            else:
                config['tensorflow_backend'] = 'CPU'
        else:
            config['tensorflow_backend'] = 'CPU'
            
    except ImportError:
        warnings.warn("TensorFlow no está instalado. Instala tensorflow-macos y tensorflow-metal para M1.")
        config['tensorflow_backend'] = 'Not Available'
    
    return config

def optimize_numpy_scipy():
    """
    Configura optimizaciones para NumPy y SciPy en M1
    """
    # Configurar variables de entorno para optimización
    if check_m1_system():
        # Usar Accelerate framework de Apple
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # Evitar conflictos con MKL
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        
        try:
            import numpy as np
            # Verificar si NumPy está usando aceleración
            if hasattr(np, '__config__'):
                print("✅ NumPy configurado para M1")
        except:
            pass

def get_optimal_batch_size(model_type: str = 'default') -> int:
    """
    Obtiene el tamaño de batch óptimo para diferentes modelos en M1
    
    Args:
        model_type: Tipo de modelo ('sparse', 'hierarchical', 'hybrid', 'default')
    
    Returns:
        Tamaño de batch recomendado
    """
    if not check_m1_system():
        # Valores conservadores para sistemas no-M1
        return {
            'sparse': 16,
            'hierarchical': 8,
            'hybrid': 4,
            'default': 16
        }.get(model_type, 16)
    
    # Valores optimizados para M1
    batch_sizes = {
        'sparse': 32,      # Representaciones dispersas son más ligeras
        'hierarchical': 16,  # Fusión jerárquica requiere más memoria
        'hybrid': 8,       # Modelo híbrido es más complejo
        'default': 32
    }
    
    return batch_sizes.get(model_type, 32)

def setup_memory_management():
    """
    Configura gestión de memoria optimizada para M1
    """
    if check_m1_system():
        # Configurar límites de memoria si es necesario
        # M1 Pro/Max tienen más memoria unificada
        import psutil
        
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if total_memory_gb >= 32:
            # Mac con 32GB+ puede manejar más datos en memoria
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        else:
            # Mac con 16GB o menos, ser más conservador
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_METAL_DEVICE_LIST'] = '0'

def print_system_info():
    """
    Imprime información del sistema y configuración
    """
    print("=" * 60)
    print("🔧 Configuración del Sistema - MacBook Pro M1")
    print("=" * 60)
    
    m1_detected = check_m1_system()
    print(f"✅ Mac M1 detectado: {m1_detected}")
    
    if m1_detected:
        print(f"📱 Sistema: {platform.system()} {platform.machine()}")
        print(f"🐍 Python: {platform.python_version()}")
        
        # Información de TensorFlow
        tf_config = configure_tensorflow_m1()
        print(f"\n🤖 TensorFlow:")
        print(f"   Backend: {tf_config['tensorflow_backend']}")
        print(f"   GPU disponible: {tf_config['gpu_available']}")
        if tf_config['gpu_device']:
            print(f"   Dispositivo GPU: {tf_config['gpu_device']}")
        
        # Información de memoria
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"\n💾 Memoria:")
            print(f"   Total: {memory.total / (1024**3):.2f} GB")
            print(f"   Disponible: {memory.available / (1024**3):.2f} GB")
        except:
            pass
        
        # Batch sizes recomendados
        print(f"\n📊 Tamaños de batch recomendados:")
        print(f"   Representaciones Dispersas: {get_optimal_batch_size('sparse')}")
        print(f"   Fusión Jerárquica: {get_optimal_batch_size('hierarchical')}")
        print(f"   Modelo Híbrido: {get_optimal_batch_size('hybrid')}")
    
    print("=" * 60)

if __name__ == "__main__":
    # Ejecutar configuración y mostrar información
    optimize_numpy_scipy()
    setup_memory_management()
    print_system_info()
