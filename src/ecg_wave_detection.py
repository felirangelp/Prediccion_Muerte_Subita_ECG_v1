"""
Detección de ondas P, Q, S, T en señales ECG
Basado en picos R detectados usando ventanas adaptativas alrededor de cada R
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, List, Optional, Tuple
import warnings


def detect_q_wave(ecg_signal: np.ndarray, r_peak: int, fs: float, 
                 rr_interval: float) -> Optional[int]:
    """
    Detectar onda Q antes del pico R
    
    La onda Q es el primer mínimo local antes del pico R dentro del complejo QRS.
    
    Args:
        ecg_signal: Señal ECG 1D
        r_peak: Índice del pico R
        fs: Frecuencia de muestreo
        rr_interval: Intervalo RR promedio en segundos (para definir ventana)
    
    Returns:
        Índice de onda Q detectada, o None si no se encuentra
    """
    # Ventana de búsqueda: [R - 0.1*RR, R]
    window_start = max(0, r_peak - int(0.1 * rr_interval * fs))
    window_end = r_peak
    
    if window_start >= window_end or window_start < 0:
        return None
    
    # Extraer ventana de señal
    search_window = ecg_signal[window_start:window_end]
    
    if len(search_window) == 0:
        return None
    
    # Buscar primer mínimo local antes de R
    # Usar find_peaks con valores negativos para encontrar mínimos
    inverted_window = -search_window
    minima, _ = signal.find_peaks(
        inverted_window,
        distance=max(1, int(0.02 * fs))  # Mínimo 20ms entre mínimos
    )
    
    if len(minima) > 0:
        # Tomar el mínimo más cercano a R (último en la ventana)
        q_idx = window_start + minima[-1]
        return q_idx
    
    # Si no se encuentra mínimo local, buscar mínimo global en la ventana
    min_idx = np.argmin(search_window)
    q_idx = window_start + min_idx
    
    # Validar que Q esté dentro del complejo QRS (no demasiado lejos de R)
    if abs(q_idx - r_peak) <= int(0.15 * rr_interval * fs):
        return q_idx
    
    return None


def detect_s_wave(ecg_signal: np.ndarray, r_peak: int, fs: float,
                 rr_interval: float) -> Optional[int]:
    """
    Detectar onda S después del pico R
    
    La onda S es el primer mínimo local después del pico R dentro del complejo QRS.
    
    Args:
        ecg_signal: Señal ECG 1D
        r_peak: Índice del pico R
        fs: Frecuencia de muestreo
        rr_interval: Intervalo RR promedio en segundos (para definir ventana)
    
    Returns:
        Índice de onda S detectada, o None si no se encuentra
    """
    # Ventana de búsqueda: [R, R + 0.1*RR]
    window_start = r_peak
    window_end = min(len(ecg_signal), r_peak + int(0.1 * rr_interval * fs))
    
    if window_start >= window_end or window_end > len(ecg_signal):
        return None
    
    # Extraer ventana de señal
    search_window = ecg_signal[window_start:window_end]
    
    if len(search_window) == 0:
        return None
    
    # Buscar primer mínimo local después de R
    inverted_window = -search_window
    minima, _ = signal.find_peaks(
        inverted_window,
        distance=max(1, int(0.02 * fs))  # Mínimo 20ms entre mínimos
    )
    
    if len(minima) > 0:
        # Tomar el primer mínimo después de R
        s_idx = window_start + minima[0]
        return s_idx
    
    # Si no se encuentra mínimo local, buscar mínimo global en la ventana
    min_idx = np.argmin(search_window)
    s_idx = window_start + min_idx
    
    # Validar que S esté dentro del complejo QRS
    if abs(s_idx - r_peak) <= int(0.15 * rr_interval * fs):
        return s_idx
    
    return None


def detect_t_wave(ecg_signal: np.ndarray, r_peak: int, fs: float,
                 rr_interval: float, next_r_peak: Optional[int] = None) -> Optional[int]:
    """
    Detectar onda T después del complejo QRS
    
    La onda T puede ser positiva o negativa. Se busca el extremo (máximo o mínimo)
    en la ventana después del QRS.
    
    Args:
        ecg_signal: Señal ECG 1D
        r_peak: Índice del pico R
        fs: Frecuencia de muestreo
        rr_interval: Intervalo RR promedio en segundos
        next_r_peak: Índice del siguiente pico R (para evitar solapamiento)
    
    Returns:
        Índice de onda T detectada, o None si no se encuentra
    """
    # Ventana de búsqueda: [R + 0.2*RR, R + 0.6*RR]
    window_start = r_peak + int(0.2 * rr_interval * fs)
    
    # Limitar ventana si hay siguiente R cerca
    if next_r_peak is not None:
        window_end = min(len(ecg_signal), next_r_peak - int(0.1 * rr_interval * fs))
    else:
        window_end = min(len(ecg_signal), r_peak + int(0.6 * rr_interval * fs))
    
    if window_start >= window_end or window_start >= len(ecg_signal):
        return None
    
    # Extraer ventana de señal
    search_window = ecg_signal[window_start:window_end]
    
    if len(search_window) == 0:
        return None
    
    # Buscar máximo y mínimo en la ventana
    max_idx = np.argmax(search_window)
    min_idx = np.argmin(search_window)
    
    # Determinar si T es positiva o negativa (usar el extremo más pronunciado)
    max_val = search_window[max_idx]
    min_val = search_window[min_idx]
    
    if abs(max_val) > abs(min_val):
        # T positiva
        t_idx = window_start + max_idx
    else:
        # T negativa
        t_idx = window_start + min_idx
    
    # Validar que T no solape con siguiente QRS
    if next_r_peak is not None:
        if t_idx >= next_r_peak - int(0.1 * rr_interval * fs):
            return None
    
    return t_idx


def detect_p_wave(ecg_signal: np.ndarray, r_peak: int, fs: float,
                 rr_interval: float, prev_r_peak: Optional[int] = None) -> Optional[int]:
    """
    Detectar onda P antes del complejo QRS
    
    La onda P puede ser positiva o negativa. Se busca el extremo (máximo o mínimo)
    en la ventana antes del QRS.
    
    Args:
        ecg_signal: Señal ECG 1D
        r_peak: Índice del pico R
        fs: Frecuencia de muestreo
        rr_interval: Intervalo RR promedio en segundos
        prev_r_peak: Índice del pico R anterior (para evitar solapamiento)
    
    Returns:
        Índice de onda P detectada, o None si no se encuentra
    """
    # Ventana de búsqueda: [R - 0.4*RR, R - 0.1*RR]
    window_end = r_peak - int(0.1 * rr_interval * fs)
    
    # Limitar ventana si hay R anterior cerca
    if prev_r_peak is not None:
        window_start = max(0, prev_r_peak + int(0.1 * rr_interval * fs))
    else:
        window_start = max(0, r_peak - int(0.4 * rr_interval * fs))
    
    if window_start >= window_end or window_end <= 0:
        return None
    
    # Extraer ventana de señal
    search_window = ecg_signal[window_start:window_end]
    
    if len(search_window) == 0:
        return None
    
    # Buscar máximo y mínimo en la ventana
    max_idx = np.argmax(search_window)
    min_idx = np.argmin(search_window)
    
    # Determinar si P es positiva o negativa (usar el extremo más pronunciado)
    max_val = search_window[max_idx]
    min_val = search_window[min_idx]
    
    if abs(max_val) > abs(min_val):
        # P positiva
        p_idx = window_start + max_idx
    else:
        # P negativa
        p_idx = window_start + min_idx
    
    # Validar que P no solape con QRS anterior
    if prev_r_peak is not None:
        if p_idx <= prev_r_peak + int(0.1 * rr_interval * fs):
            return None
    
    return p_idx


def detect_all_waves(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: float,
                    rr_intervals: Optional[np.ndarray] = None) -> Dict:
    """
    Detectar todas las ondas P, Q, S, T basado en picos R
    
    Para cada pico R, detecta las ondas asociadas usando ventanas adaptativas.
    
    Args:
        ecg_signal: Señal ECG 1D
        r_peaks: Array de índices de picos R detectados
        fs: Frecuencia de muestreo
        rr_intervals: Intervalos RR en segundos (opcional, se calculan si no se proporcionan)
    
    Returns:
        Dict con:
            - p_waves: Lista de índices de ondas P (puede contener None)
            - q_waves: Lista de índices de ondas Q (puede contener None)
            - s_waves: Lista de índices de ondas S (puede contener None)
            - t_waves: Lista de índices de ondas T (puede contener None)
            - wave_features: Diccionario con características de cada onda
    """
    if len(r_peaks) < 2:
        warnings.warn("Se requieren al menos 2 picos R para detectar ondas")
        return {
            'p_waves': [],
            'q_waves': [],
            's_waves': [],
            't_waves': [],
            'wave_features': {}
        }
    
    # Calcular intervalos RR si no se proporcionan
    if rr_intervals is None:
        rr_intervals_samples = np.diff(r_peaks)
        rr_intervals = rr_intervals_samples / fs  # Convertir a segundos
    else:
        rr_intervals = rr_intervals.copy()
    
    # Calcular intervalo RR promedio para usar en ventanas
    mean_rr = np.mean(rr_intervals)
    
    # Inicializar listas
    p_waves = []
    q_waves = []
    s_waves = []
    t_waves = []
    
    # Detectar ondas para cada pico R
    for i, r_peak in enumerate(r_peaks):
        # Obtener intervalo RR para este latido
        if i < len(rr_intervals):
            rr_interval = rr_intervals[i]
        else:
            rr_interval = mean_rr
        
        # Obtener picos R adyacentes
        prev_r = r_peaks[i - 1] if i > 0 else None
        next_r = r_peaks[i + 1] if i < len(r_peaks) - 1 else None
        
        # Detectar ondas
        q_wave = detect_q_wave(ecg_signal, r_peak, fs, rr_interval)
        s_wave = detect_s_wave(ecg_signal, r_peak, fs, rr_interval)
        t_wave = detect_t_wave(ecg_signal, r_peak, fs, rr_interval, next_r)
        p_wave = detect_p_wave(ecg_signal, r_peak, fs, rr_interval, prev_r)
        
        p_waves.append(p_wave)
        q_waves.append(q_wave)
        s_waves.append(s_wave)
        t_waves.append(t_wave)
    
    # Calcular características de ondas
    wave_features = {}
    
    # Amplitudes y anchos
    for wave_name, wave_indices in [('P', p_waves), ('Q', q_waves), 
                                     ('S', s_waves), ('T', t_waves)]:
        valid_waves = [w for w in wave_indices if w is not None]
        
        if len(valid_waves) > 0:
            amplitudes = [ecg_signal[w] for w in valid_waves]
            wave_features[f'{wave_name}_amplitude_mean'] = np.mean(amplitudes)
            wave_features[f'{wave_name}_amplitude_std'] = np.std(amplitudes)
            wave_features[f'{wave_name}_count'] = len(valid_waves)
        else:
            wave_features[f'{wave_name}_amplitude_mean'] = 0.0
            wave_features[f'{wave_name}_amplitude_std'] = 0.0
            wave_features[f'{wave_name}_count'] = 0
    
    # Intervalos PR, QT, QRS
    valid_q = [q for q in q_waves if q is not None]
    valid_s = [s for s in s_waves if s is not None]
    valid_t = [t for t in t_waves if t is not None]
    valid_p = [p for p in p_waves if p is not None]
    
    if len(valid_p) > 0 and len(r_peaks) > 0:
        pr_intervals = []
        for i, r_peak in enumerate(r_peaks):
            if i < len(p_waves) and p_waves[i] is not None:
                pr_interval = (r_peak - p_waves[i]) / fs * 1000  # ms
                if 120 <= pr_interval <= 200:  # Rango normal
                    pr_intervals.append(pr_interval)
        if pr_intervals:
            wave_features['PR_interval_mean'] = np.mean(pr_intervals)
    
    if len(valid_q) > 0 and len(valid_s) > 0:
        qrs_widths = []
        for i in range(min(len(q_waves), len(s_waves))):
            if q_waves[i] is not None and s_waves[i] is not None:
                qrs_width = (s_waves[i] - q_waves[i]) / fs * 1000  # ms
                if 50 <= qrs_width <= 120:  # Rango normal
                    qrs_widths.append(qrs_width)
        if qrs_widths:
            wave_features['QRS_width_mean'] = np.mean(qrs_widths)
    
    if len(valid_q) > 0 and len(valid_t) > 0:
        qt_intervals = []
        for i in range(min(len(q_waves), len(t_waves))):
            if q_waves[i] is not None and t_waves[i] is not None:
                qt_interval = (t_waves[i] - q_waves[i]) / fs * 1000  # ms
                if 350 <= qt_interval <= 450:  # Rango normal aproximado
                    qt_intervals.append(qt_interval)
        if qt_intervals:
            wave_features['QT_interval_mean'] = np.mean(qt_intervals)
    
    return {
        'p_waves': p_waves,
        'q_waves': q_waves,
        's_waves': s_waves,
        't_waves': t_waves,
        'wave_features': wave_features
    }


# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Detección de Ondas ECG")
    print("=" * 50)
    
    # Crear señal de ejemplo con latidos
    fs = 250  # Hz
    duration = 5  # segundos
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simular ECG con ondas P, QRS, T
    ecg_simulated = np.zeros(len(t))
    heart_rate = 70  # bpm
    rr_interval = 60.0 / heart_rate
    
    for i in range(int(duration / rr_interval)):
        beat_time = i * rr_interval
        beat_idx = int(beat_time * fs)
        
        if beat_idx < len(t):
            # Onda P (antes de QRS)
            p_time = beat_time - 0.15
            p_idx = int(p_time * fs)
            if 0 <= p_idx < len(t):
                ecg_simulated[p_idx] = 0.1 * np.sin(2 * np.pi * 5 * (t[p_idx] - p_time))
            
            # Complejo QRS
            qrs_duration = 0.1
            qrs_samples = int(qrs_duration * fs)
            for j in range(qrs_samples):
                idx = beat_idx + j
                if idx < len(t):
                    ecg_simulated[idx] = np.sin(2 * np.pi * 10 * (t[idx] - beat_time)) * \
                                        np.exp(-5 * (t[idx] - beat_time))
            
            # Onda T (después de QRS)
            t_time = beat_time + 0.2
            t_idx = int(t_time * fs)
            if 0 <= t_idx < len(t):
                ecg_simulated[t_idx] = 0.3 * np.sin(2 * np.pi * 2 * (t[t_idx] - t_time)) * \
                                     np.exp(-2 * (t[t_idx] - t_time))
    
    # Agregar ruido
    ecg_simulated += 0.05 * np.random.randn(len(t))
    
    # Simular picos R (en la práctica vendrían de Pan-Tompkins)
    r_peaks = []
    for i in range(int(duration / rr_interval)):
        beat_time = i * rr_interval
        r_peaks.append(int(beat_time * fs))
    r_peaks = np.array(r_peaks)
    
    print(f"📊 Señal de ejemplo:")
    print(f"   Longitud: {len(ecg_simulated)} muestras")
    print(f"   Frecuencia de muestreo: {fs} Hz")
    print(f"   Picos R simulados: {len(r_peaks)}")
    
    # Detectar todas las ondas
    waves_result = detect_all_waves(ecg_simulated, r_peaks, fs)
    
    print(f"\n📊 Resultados de detección:")
    print(f"   Ondas P detectadas: {sum(1 for p in waves_result['p_waves'] if p is not None)}")
    print(f"   Ondas Q detectadas: {sum(1 for q in waves_result['q_waves'] if q is not None)}")
    print(f"   Ondas S detectadas: {sum(1 for s in waves_result['s_waves'] if s is not None)}")
    print(f"   Ondas T detectadas: {sum(1 for t in waves_result['t_waves'] if t is not None)}")
    
    print(f"\n📊 Características:")
    for key, value in waves_result['wave_features'].items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.3f}")
    
    print(f"\n💡 Para usar en otros módulos:")
    print(f"   from src.ecg_wave_detection import detect_all_waves")
    print(f"   waves = detect_all_waves(ecg_signal, r_peaks, fs)")

