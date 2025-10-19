"""
Utilidades básicas para trabajar con datasets de ECG de PhysioNet
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List

def load_ecg_record(record_path: str, channels: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Cargar un registro ECG desde PhysioNet
    
    Args:
        record_path: Ruta al registro (ej: 'datasets/sddb/30')
        channels: Lista de canales a cargar (None = todos)
    
    Returns:
        Tuple con (señal, metadatos)
    """
    try:
        # Cargar registro completo
        record = wfdb.rdrecord(record_path, channels=channels)
        
        # Extraer señal y metadatos
        signal = record.p_signal
        metadata = {
            'fs': record.fs,  # Frecuencia de muestreo
            'sig_len': record.sig_len,  # Longitud de la señal
            'sig_name': record.sig_name,  # Nombres de las señales
            'n_sig': record.n_sig,  # Número de señales
            'duration_hours': record.sig_len / record.fs / 3600,
            'record_name': record.record_name
        }
        
        return signal, metadata
        
    except Exception as e:
        raise ValueError(f"Error cargando registro {record_path}: {str(e)}")

def plot_ecg_signal(signal: np.ndarray, fs: float, duration: float = 10.0, 
                   channels: Optional[List[int]] = None, title: str = "ECG Signal") -> None:
    """
    Visualizar señal ECG
    
    Args:
        signal: Array de señal ECG (muestras x canales)
        fs: Frecuencia de muestreo
        duration: Duración a mostrar en segundos
        channels: Canales específicos a mostrar
        title: Título del gráfico
    """
    if channels is None:
        channels = list(range(signal.shape[1]))
    
    # Calcular muestras a mostrar
    samples_to_show = int(duration * fs)
    samples_to_show = min(samples_to_show, signal.shape[0])
    
    # Crear tiempo en segundos
    time_axis = np.arange(samples_to_show) / fs
    
    # Crear subplots
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
    
    if n_channels == 1:
        axes = [axes]
    
    for i, channel in enumerate(channels):
        axes[i].plot(time_axis, signal[:samples_to_show, channel])
        axes[i].set_title(f"Canal {channel}: {title}")
        axes[i].set_xlabel("Tiempo (s)")
        axes[i].set_ylabel("Amplitud")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def get_record_info(record_path: str) -> Dict:
    """
    Obtener información detallada de un registro ECG
    
    Args:
        record_path: Ruta al registro
    
    Returns:
        Diccionario con información del registro
    """
    try:
        # Cargar solo metadatos (más rápido)
        record = wfdb.rdheader(record_path)
        
        info = {
            'record_name': record.record_name,
            'fs': record.fs,
            'sig_len': record.sig_len,
            'n_sig': record.n_sig,
            'sig_name': record.sig_name,
            'duration_hours': record.sig_len / record.fs / 3600,
            'duration_minutes': record.sig_len / record.fs / 60,
            'file_path': record_path
        }
        
        # Intentar cargar anotaciones si existen
        try:
            ann = wfdb.rdann(record_path, 'atr')
            info['annotations'] = {
                'sample': ann.sample,
                'symbol': ann.symbol,
                'description': ann.description
            }
        except:
            info['annotations'] = None
        
        return info
        
    except Exception as e:
        raise ValueError(f"Error obteniendo información de {record_path}: {str(e)}")

def list_available_records(dataset_path: str) -> List[str]:
    """
    Listar todos los registros disponibles en un dataset
    
    Args:
        dataset_path: Ruta al dataset (ej: 'datasets/sddb')
    
    Returns:
        Lista de nombres de registros disponibles
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        return []
    
    # Buscar archivos .hea
    hea_files = list(dataset_dir.glob("*.hea"))
    records = [f.stem for f in hea_files]
    
    return sorted(records)

def get_dataset_summary(dataset_path: str) -> Dict:
    """
    Obtener resumen de un dataset completo
    
    Args:
        dataset_path: Ruta al dataset
    
    Returns:
        Diccionario con resumen del dataset
    """
    records = list_available_records(dataset_path)
    
    if not records:
        return {'total_records': 0, 'records': []}
    
    records_info = []
    total_duration = 0
    
    for record_name in records:
        try:
            record_path = str(Path(dataset_path) / record_name)
            info = get_record_info(record_path)
            records_info.append(info)
            total_duration += info['duration_hours']
        except Exception as e:
            print(f"⚠️  Error procesando {record_name}: {str(e)}")
            continue
    
    return {
        'total_records': len(records_info),
        'total_duration_hours': total_duration,
        'avg_duration_hours': total_duration / len(records_info) if records_info else 0,
        'records': records_info
    }

def format_duration(hours: float) -> str:
    """
    Formatear duración en horas a formato legible
    
    Args:
        hours: Duración en horas
    
    Returns:
        String formateado
    """
    if hours < 1:
        minutes = hours * 60
        return f"{minutes:.1f} min"
    elif hours < 24:
        return f"{hours:.1f} h"
    else:
        days = hours / 24
        return f"{days:.1f} días"

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso básico
    print("🔍 Ejemplo de uso de utilidades ECG")
    print("=" * 40)
    
    # Listar datasets disponibles
    datasets = ["datasets/sddb", "datasets/nsrdb", "datasets/cudb"]
    
    for dataset_path in datasets:
        if Path(dataset_path).exists():
            print(f"\n📊 Dataset: {dataset_path}")
            summary = get_dataset_summary(dataset_path)
            print(f"   Registros: {summary['total_records']}")
            if summary['total_records'] > 0:
                print(f"   Duración total: {format_duration(summary['total_duration_hours'])}")
            else:
                print(f"   Duración total: No disponible (sin registros)")
        else:
            print(f"\n❌ Dataset no encontrado: {dataset_path}")
    
    print(f"\n💡 Para usar estas funciones:")
    print(f"   from src.utils import load_ecg_record, plot_ecg_signal")
    print(f"   signal, metadata = load_ecg_record('datasets/sddb/30')")
    print(f"   plot_ecg_signal(signal, metadata['fs'])")
