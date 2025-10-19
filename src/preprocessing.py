"""
Funciones de preprocesamiento para señales ECG
"""

import numpy as np
import scipy.signal as signal
from scipy import stats
from typing import Tuple, Optional
import warnings

def preprocess_ecg_signal(ecg_signal: np.ndarray, fs: float, 
                         remove_baseline: bool = True,
                         filter_noise: bool = True,
                         normalize: bool = True) -> np.ndarray:
    """
    Preprocesamiento básico de señal ECG
    
    Args:
        ecg_signal: Señal ECG (muestras x canales)
        fs: Frecuencia de muestreo
        remove_baseline: Si remover deriva de línea base
        filter_noise: Si filtrar ruido de alta frecuencia
        normalize: Si normalizar la señal
    
    Returns:
        Señal ECG preprocesada
    """
    processed_signal = ecg_signal.copy()
    
    # Procesar cada canal por separado
    for channel in range(processed_signal.shape[1]):
        channel_signal = processed_signal[:, channel]
        
        # 1. Remover deriva de línea base (filtro pasa-altos)
        if remove_baseline:
            # Filtro Butterworth pasa-altos de 0.5 Hz
            nyquist = fs / 2
            low_cutoff = 0.5 / nyquist
            
            if low_cutoff < 1.0:  # Evitar problemas de diseño de filtro
                b, a = signal.butter(4, low_cutoff, btype='high')
                channel_signal = signal.filtfilt(b, a, channel_signal)
        
        # 2. Filtrar ruido de alta frecuencia (filtro pasa-bajos)
        if filter_noise:
            # Filtro Butterworth pasa-bajos de 40 Hz
            high_cutoff = 40.0 / nyquist
            
            if high_cutoff < 1.0:
                b, a = signal.butter(4, high_cutoff, btype='low')
                channel_signal = signal.filtfilt(b, a, channel_signal)
        
        # 3. Normalizar señal
        if normalize:
            # Normalización z-score
            channel_signal = (channel_signal - np.mean(channel_signal)) / np.std(channel_signal)
        
        processed_signal[:, channel] = channel_signal
    
    return processed_signal

def detect_r_peaks(ecg_signal: np.ndarray, fs: float, 
                   channel: int = 0, threshold: float = 0.6) -> np.ndarray:
    """
    Detectar picos R en señal ECG usando algoritmo de Pan-Tompkins
    
    Args:
        ecg_signal: Señal ECG (muestras x canales)
        fs: Frecuencia de muestreo
        channel: Canal a usar para detección
        threshold: Umbral para detección de picos
    
    Returns:
        Array con índices de los picos R detectados
    """
    signal_channel = ecg_signal[:, channel]
    
    # Derivada para enfatizar picos R
    derivative = np.diff(signal_channel)
    
    # Cuadrado para hacer todos los valores positivos
    squared = derivative ** 2
    
    # Ventana deslizante para suavizar
    window_size = int(0.15 * fs)  # 150ms
    if window_size % 2 == 0:
        window_size += 1
    
    smoothed = signal.savgol_filter(squared, window_size, 3)
    
    # Detectar picos usando scipy
    peaks, _ = signal.find_peaks(smoothed, 
                                 height=np.max(smoothed) * threshold,
                                 distance=int(0.2 * fs))  # Mínimo 200ms entre picos
    
    return peaks + 1  # +1 por la derivada

def calculate_hrv_features(r_peaks: np.ndarray, fs: float) -> dict:
    """
    Calcular características de variabilidad de frecuencia cardíaca (HRV)
    
    Args:
        r_peaks: Índices de picos R detectados
        fs: Frecuencia de muestreo
    
    Returns:
        Diccionario con características HRV
    """
    if len(r_peaks) < 2:
        return {}
    
    # Calcular intervalos RR (en ms)
    rr_intervals = np.diff(r_peaks) / fs * 1000  # Convertir a ms
    
    # Remover outliers (intervalos RR muy cortos o largos)
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    
    if len(rr_intervals) < 2:
        return {}
    
    # Características en dominio del tiempo
    features = {
        'mean_rr': np.mean(rr_intervals),
        'std_rr': np.std(rr_intervals),
        'rmssd': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
        'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
        'heart_rate': 60000 / np.mean(rr_intervals),  # BPM
        'rr_count': len(rr_intervals)
    }
    
    # Características en dominio de frecuencia (si hay suficientes datos)
    if len(rr_intervals) > 100:
        try:
            # Interpolación para análisis espectral
            time_intervals = np.cumsum(rr_intervals) / 1000  # Convertir a segundos
            interpolated_rr = np.interp(np.arange(0, time_intervals[-1], 1/fs), 
                                      time_intervals, rr_intervals)
            
            # Análisis espectral
            freqs, psd = signal.welch(interpolated_rr, fs=fs, nperseg=min(256, len(interpolated_rr)//4))
            
            # Bandas de frecuencia
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.4)
            
            features.update({
                'vlf_power': np.trapz(psd[vlf_band], freqs[vlf_band]),
                'lf_power': np.trapz(psd[lf_band], freqs[lf_band]),
                'hf_power': np.trapz(psd[hf_band], freqs[hf_band]),
                'lf_hf_ratio': features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
            })
        except:
            pass
    
    return features

def extract_features(ecg_signal: np.ndarray, fs: float, 
                    window_size: float = 30.0) -> dict:
    """
    Extraer características de señal ECG
    
    Args:
        ecg_signal: Señal ECG (muestras x canales)
        fs: Frecuencia de muestreo
        window_size: Tamaño de ventana en segundos para análisis
    
    Returns:
        Diccionario con características extraídas
    """
    features = {}
    
    # Procesar cada canal
    for channel in range(ecg_signal.shape[1]):
        channel_signal = ecg_signal[:, channel]
        
        # Características estadísticas básicas
        features[f'channel_{channel}_mean'] = np.mean(channel_signal)
        features[f'channel_{channel}_std'] = np.std(channel_signal)
        features[f'channel_{channel}_skewness'] = stats.skew(channel_signal)
        features[f'channel_{channel}_kurtosis'] = stats.kurtosis(channel_signal)
        features[f'channel_{channel}_rms'] = np.sqrt(np.mean(channel_signal ** 2))
        
        # Características de frecuencia
        freqs, psd = signal.welch(channel_signal, fs=fs, nperseg=min(256, len(channel_signal)//4))
        features[f'channel_{channel}_dominant_freq'] = freqs[np.argmax(psd)]
        features[f'channel_{channel}_spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        
        # Detectar picos R y calcular HRV (solo para el primer canal)
        if channel == 0:
            try:
                r_peaks = detect_r_peaks(ecg_signal, fs, channel=channel)
                if len(r_peaks) > 1:
                    hrv_features = calculate_hrv_features(r_peaks, fs)
                    features.update({f'hrv_{k}': v for k, v in hrv_features.items()})
            except:
                pass
    
    return features

def segment_signal(ecg_signal: np.ndarray, fs: float, 
                  window_size: float = 30.0, overlap: float = 0.5) -> list:
    """
    Segmentar señal ECG en ventanas deslizantes
    
    Args:
        ecg_signal: Señal ECG (muestras x canales)
        fs: Frecuencia de muestreo
        window_size: Tamaño de ventana en segundos
        overlap: Solapamiento entre ventanas (0-1)
    
    Returns:
        Lista de segmentos de señal
    """
    window_samples = int(window_size * fs)
    step_size = int(window_samples * (1 - overlap))
    
    segments = []
    
    for start in range(0, ecg_signal.shape[0] - window_samples + 1, step_size):
        end = start + window_samples
        segment = ecg_signal[start:end, :]
        segments.append(segment)
    
    return segments

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Ejemplo de preprocesamiento de ECG")
    print("=" * 40)
    
    # Crear señal de ejemplo
    fs = 250  # Hz
    duration = 10  # segundos
    t = np.linspace(0, duration, int(fs * duration))
    
    # Señal ECG simulada con ruido
    ecg_simulated = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    ecg_simulated = ecg_simulated.reshape(-1, 1)  # Formato (muestras x canales)
    
    print(f"📊 Señal original:")
    print(f"   Forma: {ecg_simulated.shape}")
    print(f"   Media: {np.mean(ecg_simulated):.3f}")
    print(f"   Desv. Est.: {np.std(ecg_simulated):.3f}")
    
    # Preprocesar señal
    processed = preprocess_ecg_signal(ecg_simulated, fs)
    
    print(f"\n📊 Señal preprocesada:")
    print(f"   Forma: {processed.shape}")
    print(f"   Media: {np.mean(processed):.3f}")
    print(f"   Desv. Est.: {np.std(processed):.3f}")
    
    # Extraer características
    features = extract_features(processed, fs)
    
    print(f"\n📊 Características extraídas:")
    for key, value in list(features.items())[:5]:  # Mostrar solo las primeras 5
        print(f"   {key}: {value:.3f}")
    
    print(f"\n💡 Para usar estas funciones:")
    print(f"   from src.preprocessing import preprocess_ecg_signal, extract_features")
    print(f"   processed = preprocess_ecg_signal(signal, fs)")
    print(f"   features = extract_features(processed, fs)")
