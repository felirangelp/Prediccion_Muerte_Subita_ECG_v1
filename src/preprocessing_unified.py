"""
Preprocesamiento unificado para ambos métodos de predicción de muerte súbita cardíaca
Incluye: filtrado paso banda, normalización, segmentación temporal y diezmado
"""

import numpy as np
import scipy.signal as signal
from scipy import stats
from typing import Tuple, Optional, List
import warnings
from pathlib import Path
import pickle
from cachetools import LRUCache
from functools import lru_cache

# Cache para resultados de preprocesamiento
_preprocessing_cache = LRUCache(maxsize=100)

def bandpass_filter(signal_data: np.ndarray, fs: float, 
                   lowcut: float = 0.5, highcut: float = 40.0,
                   order: int = 4) -> np.ndarray:
    """
    Filtro paso banda para eliminar deriva de línea base y ruido de alta frecuencia
    
    Args:
        signal_data: Señal ECG (muestras x canales o 1D)
        fs: Frecuencia de muestreo
        lowcut: Frecuencia de corte inferior (Hz)
        highcut: Frecuencia de corte superior (Hz)
        order: Orden del filtro Butterworth
    
    Returns:
        Señal filtrada
    """
    nyquist = fs / 2.0
    
    # Normalizar frecuencias
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Verificar que las frecuencias estén en rango válido
    if low >= 1.0 or high >= 1.0:
        warnings.warn(f"Frecuencias de corte fuera de rango. Usando valores por defecto.")
        return signal_data
    
    if low <= 0:
        low = 0.01
    
    # Diseñar filtro Butterworth paso banda
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Aplicar filtro (filtfilt para fase cero)
    if signal_data.ndim == 1:
        filtered = signal.filtfilt(b, a, signal_data)
    else:
        filtered = np.zeros_like(signal_data)
        for channel in range(signal_data.shape[1]):
            filtered[:, channel] = signal.filtfilt(b, a, signal_data[:, channel])
    
    return filtered

def normalize_signal(signal_data: np.ndarray, method: str = 'zscore',
                    per_channel: bool = True) -> np.ndarray:
    """
    Normalizar señal ECG usando diferentes métodos
    
    Args:
        signal_data: Señal ECG (muestras x canales o 1D)
        method: Método de normalización ('zscore', 'minmax', 'robust')
        per_channel: Si normalizar cada canal por separado
    
    Returns:
        Señal normalizada
    """
    normalized = signal_data.copy()
    
    if signal_data.ndim == 1:
        # Señal 1D
        if method == 'zscore':
            mean = np.mean(normalized)
            std = np.std(normalized)
            if std > 0:
                normalized = (normalized - mean) / std
        elif method == 'minmax':
            min_val = np.min(normalized)
            max_val = np.max(normalized)
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = np.median(normalized)
            mad = np.median(np.abs(normalized - median))
            if mad > 0:
                normalized = (normalized - median) / (1.4826 * mad)  # Factor para MAD
    else:
        # Señal multi-canal
        if per_channel:
            for channel in range(signal_data.shape[1]):
                channel_signal = normalized[:, channel]
                
                if method == 'zscore':
                    mean = np.mean(channel_signal)
                    std = np.std(channel_signal)
                    if std > 0:
                        normalized[:, channel] = (channel_signal - mean) / std
                elif method == 'minmax':
                    min_val = np.min(channel_signal)
                    max_val = np.max(channel_signal)
                    if max_val > min_val:
                        normalized[:, channel] = (channel_signal - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = np.median(channel_signal)
                    mad = np.median(np.abs(channel_signal - median))
                    if mad > 0:
                        normalized[:, channel] = (channel_signal - median) / (1.4826 * mad)
        else:
            # Normalizar sobre toda la señal
            if method == 'zscore':
                mean = np.mean(normalized)
                std = np.std(normalized)
                if std > 0:
                    normalized = (normalized - mean) / std
            elif method == 'minmax':
                min_val = np.min(normalized)
                max_val = np.max(normalized)
                if max_val > min_val:
                    normalized = (normalized - min_val) / (max_val - min_val)
    
    return normalized

def downsample_signal(signal_data: np.ndarray, original_fs: float,
                     target_fs: float, method: str = 'decimate') -> Tuple[np.ndarray, float]:
    """
    Diezmado/re-muestreo de señal para estandarizar frecuencia de muestreo
    
    Args:
        signal_data: Señal ECG (muestras x canales o 1D)
        original_fs: Frecuencia de muestreo original
        target_fs: Frecuencia de muestreo objetivo
        method: Método de diezmado ('decimate', 'resample', 'interpolate')
    
    Returns:
        Tuple con (señal diezmada, frecuencia de muestreo efectiva)
    """
    if original_fs == target_fs:
        return signal_data, original_fs
    
    if original_fs < target_fs:
        # Interpolación (upsampling)
        ratio = target_fs / original_fs
        num_samples = int(len(signal_data) * ratio)
        
        if signal_data.ndim == 1:
            time_original = np.arange(len(signal_data)) / original_fs
            time_new = np.linspace(0, time_original[-1], num_samples)
            downsampled = np.interp(time_new, time_original, signal_data)
        else:
            downsampled = np.zeros((num_samples, signal_data.shape[1]))
            time_original = np.arange(len(signal_data)) / original_fs
            time_new = np.linspace(0, time_original[-1], num_samples)
            for channel in range(signal_data.shape[1]):
                downsampled[:, channel] = np.interp(time_new, time_original, signal_data[:, channel])
        
        return downsampled, target_fs
    
    # Downsampling
    ratio = original_fs / target_fs
    
    if method == 'decimate':
        # Usar decimate de scipy (anti-aliasing incluido)
        # Solo usar decimate si el ratio es cercano a un entero (dentro de 0.1)
        decimation_factor = int(round(ratio))
        
        # Si el ratio no es cercano a un entero, usar resample en su lugar
        if abs(ratio - decimation_factor) > 0.1:
            method = 'resample'
        
        if method == 'decimate':
            try:
                if signal_data.ndim == 1:
                    downsampled = signal.decimate(signal_data, decimation_factor, ftype='fir')
                else:
                    downsampled = np.zeros((len(signal_data) // decimation_factor, signal_data.shape[1]))
                    for channel in range(signal_data.shape[1]):
                        downsampled[:, channel] = signal.decimate(signal_data[:, channel], 
                                                                 decimation_factor, ftype='fir')
            except (ValueError, TypeError):
                # Si decimate falla, cambiar a resample
                method = 'resample'
    elif method == 'resample':
        # Resample de scipy
        num_samples = int(len(signal_data) / ratio)
        
        if signal_data.ndim == 1:
            downsampled = signal.resample(signal_data, num_samples)
        else:
            downsampled = np.zeros((num_samples, signal_data.shape[1]))
            for channel in range(signal_data.shape[1]):
                downsampled[:, channel] = signal.resample(signal_data[:, channel], num_samples)
    else:  # interpolate
        num_samples = int(len(signal_data) / ratio)
        time_original = np.arange(len(signal_data)) / original_fs
        time_new = np.linspace(0, time_original[-1], num_samples)
        
        if signal_data.ndim == 1:
            downsampled = np.interp(time_new, time_original, signal_data)
        else:
            downsampled = np.zeros((num_samples, signal_data.shape[1]))
            for channel in range(signal_data.shape[1]):
                downsampled[:, channel] = np.interp(time_new, time_original, signal_data[:, channel])
    
    return downsampled, target_fs

def segment_temporal(signal_data: np.ndarray, fs: float,
                    window_size: float = 60.0, overlap: float = 0.0,
                    min_samples: int = 100) -> List[Tuple[np.ndarray, float]]:
    """
    Segmentar señal en ventanas temporales (útil para análisis de ventanas pre-SCD)
    
    Args:
        signal_data: Señal ECG (muestras x canales o 1D)
        fs: Frecuencia de muestreo
        window_size: Tamaño de ventana en segundos
        overlap: Solapamiento entre ventanas (0-1)
        min_samples: Mínimo de muestras requeridas para un segmento válido
    
    Returns:
        Lista de tuplas (segmento, tiempo_inicio_segundo)
    """
    window_samples = int(window_size * fs)
    step_size = int(window_samples * (1 - overlap))
    
    segments = []
    
    for start_idx in range(0, signal_data.shape[0] - window_samples + 1, step_size):
        end_idx = start_idx + window_samples
        
        if signal_data.ndim == 1:
            segment = signal_data[start_idx:end_idx]
        else:
            segment = signal_data[start_idx:end_idx, :]
        
        if len(segment) >= min_samples:
            time_start = start_idx / fs
            segments.append((segment, time_start))
    
    return segments

def preprocess_unified(ecg_signal: np.ndarray, fs: float,
                      target_fs: Optional[float] = 128.0,
                      apply_filter: bool = True,
                      apply_normalize: bool = True,
                      normalize_method: str = 'zscore',
                      segment_window: Optional[float] = None,
                      cache_key: Optional[str] = None) -> dict:
    """
    Preprocesamiento unificado completo para ambos métodos
    
    Args:
        ecg_signal: Señal ECG (muestras x canales o 1D)
        fs: Frecuencia de muestreo original
        target_fs: Frecuencia de muestreo objetivo (None = no diezmado)
        apply_filter: Si aplicar filtro paso banda
        apply_normalize: Si normalizar señal
        normalize_method: Método de normalización
        segment_window: Tamaño de ventana para segmentación (None = no segmentar)
        cache_key: Clave para cache (opcional)
    
    Returns:
        Diccionario con señal procesada y metadatos
    """
    # Verificar cache
    if cache_key and cache_key in _preprocessing_cache:
        return _preprocessing_cache[cache_key]
    
    processed_signal = ecg_signal.copy()
    metadata = {
        'original_fs': fs,
        'original_shape': ecg_signal.shape,
        'processing_steps': []
    }
    
    # 1. Filtrado paso banda
    if apply_filter:
        processed_signal = bandpass_filter(processed_signal, fs)
        # Limpiar NaN/Inf después del filtro
        processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
        metadata['processing_steps'].append('bandpass_filter')
    
    # 2. Normalización
    if apply_normalize:
        processed_signal = normalize_signal(processed_signal, method=normalize_method)
        # Limpiar NaN/Inf después de la normalización
        processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
        metadata['processing_steps'].append(f'normalize_{normalize_method}')
    
    # 3. Diezmado (si es necesario)
    if target_fs is not None and target_fs != fs:
        processed_signal, effective_fs = downsample_signal(processed_signal, fs, target_fs)
        # Limpiar NaN/Inf después del diezmado
        processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
        metadata['effective_fs'] = effective_fs
        metadata['processing_steps'].append(f'downsample_{fs}_to_{target_fs}')
    else:
        metadata['effective_fs'] = fs
    
    # Limpieza final de NaN/Inf
    processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Verificar que la señal final es válida
    if len(processed_signal) == 0 or np.all(processed_signal == 0):
        # Si la señal es completamente cero, intentar sin normalización
        if apply_normalize:
            # Reiniciar con señal original
            processed_signal = ecg_signal.copy()
            if apply_filter:
                processed_signal = bandpass_filter(processed_signal, fs)
                processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
            if target_fs is not None and target_fs != fs:
                processed_signal, effective_fs = downsample_signal(processed_signal, fs, target_fs)
                processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
                metadata['effective_fs'] = effective_fs
            else:
                metadata['effective_fs'] = fs
            processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 4. Segmentación (opcional)
    segments = None
    if segment_window is not None:
        segments = segment_temporal(processed_signal, metadata['effective_fs'], 
                                  window_size=segment_window)
        metadata['processing_steps'].append(f'segment_{segment_window}s')
        metadata['num_segments'] = len(segments)
    
    result = {
        'signal': processed_signal,
        'metadata': metadata,
        'segments': segments
    }
    
    # Guardar en cache
    if cache_key:
        _preprocessing_cache[cache_key] = result
    
    return result

def preprocess_for_sparse_method(ecg_signal: np.ndarray, fs: float) -> dict:
    """
    Preprocesamiento específico para método de representaciones dispersas
    
    Args:
        ecg_signal: Señal ECG
        fs: Frecuencia de muestreo
    
    Returns:
        Resultado del preprocesamiento
    """
    return preprocess_unified(
        ecg_signal=ecg_signal,
        fs=fs,
        target_fs=128.0,  # Estandarizar a 128 Hz
        apply_filter=True,
        apply_normalize=True,
        normalize_method='zscore',
        segment_window=60.0,  # Ventanas de 1 minuto
        cache_key=None
    )

def preprocess_for_hierarchical_method(ecg_signal: np.ndarray, fs: float) -> dict:
    """
    Preprocesamiento específico para método de fusión jerárquica
    
    Args:
        ecg_signal: Señal ECG
        fs: Frecuencia de muestreo
    
    Returns:
        Resultado del preprocesamiento
    """
    return preprocess_unified(
        ecg_signal=ecg_signal,
        fs=fs,
        target_fs=None,  # Mantener frecuencia original
        apply_filter=True,
        apply_normalize=True,
        normalize_method='zscore',
        segment_window=60.0,  # Ventanas de 1 minuto
        cache_key=None
    )

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Preprocesamiento Unificado - Predicción SCD")
    print("=" * 50)
    
    # Crear señal de ejemplo
    fs = 250  # Hz
    duration = 120  # segundos (2 minutos)
    t = np.linspace(0, duration, int(fs * duration))
    
    # Señal ECG simulada con múltiples componentes
    ecg_simulated = (
        np.sin(2 * np.pi * 1.2 * t) +  # Componente principal
        0.3 * np.sin(2 * np.pi * 2.5 * t) +  # Segundo armónico
        0.1 * np.random.randn(len(t))  # Ruido
    )
    ecg_simulated = ecg_simulated.reshape(-1, 1)  # Formato (muestras x canales)
    
    print(f"📊 Señal original:")
    print(f"   Forma: {ecg_simulated.shape}")
    print(f"   Frecuencia de muestreo: {fs} Hz")
    print(f"   Duración: {duration} segundos")
    
    # Preprocesamiento para método sparse
    print(f"\n📊 Preprocesamiento para Método 1 (Representaciones Dispersas):")
    result_sparse = preprocess_for_sparse_method(ecg_simulated, fs)
    print(f"   Señal procesada: {result_sparse['signal'].shape}")
    print(f"   Frecuencia efectiva: {result_sparse['metadata']['effective_fs']} Hz")
    print(f"   Pasos: {', '.join(result_sparse['metadata']['processing_steps'])}")
    if result_sparse['segments']:
        print(f"   Segmentos: {len(result_sparse['segments'])}")
    
    # Preprocesamiento para método hierarchical
    print(f"\n📊 Preprocesamiento para Método 2 (Fusión Jerárquica):")
    result_hierarchical = preprocess_for_hierarchical_method(ecg_simulated, fs)
    print(f"   Señal procesada: {result_hierarchical['signal'].shape}")
    print(f"   Frecuencia efectiva: {result_hierarchical['metadata']['effective_fs']} Hz")
    print(f"   Pasos: {', '.join(result_hierarchical['metadata']['processing_steps'])}")
    if result_hierarchical['segments']:
        print(f"   Segmentos: {len(result_hierarchical['segments'])}")
    
    print(f"\n💡 Para usar en otros módulos:")
    print(f"   from src.preprocessing_unified import preprocess_for_sparse_method")
    print(f"   result = preprocess_for_sparse_method(signal, fs)")

