"""
Análisis de tacograma y variabilidad de frecuencia cardíaca (HRV)
Incluye cálculo de intervalos RR, tacograma y frecuencia cardíaca global
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings


def filter_rr_intervals(rr_intervals: np.ndarray, 
                        min_rr: float = 300.0, 
                        max_rr: float = 2000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtrar intervalos RR anómalos
    
    Elimina intervalos RR que están fuera del rango fisiológico normal.
    Rango típico: 300-2000 ms (30-200 bpm).
    
    Args:
        rr_intervals: Array de intervalos RR en milisegundos
        min_rr: Intervalo RR mínimo válido en ms (default: 300ms = 200 bpm)
        max_rr: Intervalo RR máximo válido en ms (default: 2000ms = 30 bpm)
    
    Returns:
        Tuple con (rr_filtered, valid_indices)
        - rr_filtered: Array de intervalos RR filtrados
        - valid_indices: Índices de intervalos válidos en el array original
    """
    if len(rr_intervals) == 0:
        return np.array([]), np.array([], dtype=int)
    
    # Crear máscara de intervalos válidos
    valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
    valid_indices = np.where(valid_mask)[0]
    
    # Filtrar intervalos
    rr_filtered = rr_intervals[valid_mask]
    
    if len(rr_filtered) < len(rr_intervals):
        removed = len(rr_intervals) - len(rr_filtered)
        warnings.warn(f"Se removieron {removed} intervalos RR anómalos "
                     f"(fuera del rango {min_rr}-{max_rr} ms)")
    
    return rr_filtered, valid_indices


def calculate_global_heart_rate(rr_intervals: np.ndarray) -> float:
    """
    Calcular frecuencia cardíaca global en bpm
    
    Calcula la frecuencia cardíaca promedio basada en los intervalos RR.
    
    Args:
        rr_intervals: Array de intervalos RR en milisegundos
    
    Returns:
        Frecuencia cardíaca global en bpm (beats per minute)
    """
    if len(rr_intervals) == 0:
        return 0.0
    
    # Filtrar intervalos anómalos antes de calcular
    rr_filtered, _ = filter_rr_intervals(rr_intervals)
    
    if len(rr_filtered) == 0:
        return 0.0
    
    # Calcular intervalo RR promedio en ms
    mean_rr = np.mean(rr_filtered)
    
    # Convertir a frecuencia cardíaca: HR = 60000 / RR_ms
    heart_rate_bpm = 60000.0 / mean_rr
    
    return heart_rate_bpm


def calculate_tachogram(r_peaks: np.ndarray, fs: float,
                       filter_anomalous: bool = True) -> Dict:
    """
    Calcular tacograma completo (intervalos RR vs tiempo)
    
    El tacograma es la representación gráfica de la variabilidad de los intervalos RR
    a lo largo del tiempo. Es fundamental para el análisis de HRV.
    
    Args:
        r_peaks: Array de índices de picos R detectados
        fs: Frecuencia de muestreo
        filter_anomalous: Si filtrar intervalos RR anómalos (default: True)
    
    Returns:
        Dict con:
            - rr_intervals: Array de intervalos RR en milisegundos
            - time_points: Array de tiempos asociados en segundos
            - tachogram_data: DataFrame de pandas con columnas ['time', 'rr_interval']
            - heart_rate_bpm: Frecuencia cardíaca global en bpm
            - metadata: Información adicional (n_peaks, n_intervals, etc.)
    """
    if len(r_peaks) < 2:
        warnings.warn("Se requieren al menos 2 picos R para calcular tacograma")
        return {
            'rr_intervals': np.array([]),
            'time_points': np.array([]),
            'tachogram_data': pd.DataFrame(columns=['time', 'rr_interval']),
            'heart_rate_bpm': 0.0,
            'metadata': {
                'n_peaks': len(r_peaks),
                'n_intervals': 0,
                'filtered': False
            }
        }
    
    # Calcular intervalos RR en muestras
    rr_intervals_samples = np.diff(r_peaks)
    
    # Convertir a milisegundos
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000.0
    
    # Filtrar intervalos anómalos si se solicita
    if filter_anomalous:
        rr_intervals_filtered, valid_indices = filter_rr_intervals(rr_intervals_ms)
        
        # Ajustar time_points para mantener correspondencia
        # Los time_points corresponden a los picos R (no a los intervalos)
        # Usamos el tiempo del primer pico R de cada intervalo
        time_points = r_peaks[valid_indices] / fs
    else:
        rr_intervals_filtered = rr_intervals_ms
        time_points = r_peaks[:-1] / fs  # Todos excepto el último
    
    # Calcular frecuencia cardíaca global
    heart_rate_bpm = calculate_global_heart_rate(rr_intervals_filtered)
    
    # Crear DataFrame con datos del tacograma
    tachogram_data = pd.DataFrame({
        'time': time_points,
        'rr_interval': rr_intervals_filtered
    })
    
    # Metadata
    metadata = {
        'n_peaks': len(r_peaks),
        'n_intervals': len(rr_intervals_filtered),
        'filtered': filter_anomalous,
        'mean_rr_ms': np.mean(rr_intervals_filtered) if len(rr_intervals_filtered) > 0 else 0.0,
        'std_rr_ms': np.std(rr_intervals_filtered) if len(rr_intervals_filtered) > 0 else 0.0,
        'min_rr_ms': np.min(rr_intervals_filtered) if len(rr_intervals_filtered) > 0 else 0.0,
        'max_rr_ms': np.max(rr_intervals_filtered) if len(rr_intervals_filtered) > 0 else 0.0,
        'heart_rate_bpm': heart_rate_bpm
    }
    
    return {
        'rr_intervals': rr_intervals_filtered,
        'time_points': time_points,
        'tachogram_data': tachogram_data,
        'heart_rate_bpm': heart_rate_bpm,
        'metadata': metadata
    }


def calculate_hrv_time_domain(rr_intervals: np.ndarray) -> Dict:
    """
    Calcular métricas de HRV en dominio del tiempo
    
    Métricas estándar de variabilidad de frecuencia cardíaca:
    - SDNN: Desviación estándar de intervalos RR
    - RMSSD: Raíz cuadrada de la media de diferencias al cuadrado
    - pNN50: Porcentaje de intervalos RR que difieren >50ms del anterior
    
    Args:
        rr_intervals: Array de intervalos RR en milisegundos
    
    Returns:
        Dict con métricas de HRV en dominio del tiempo
    """
    if len(rr_intervals) < 2:
        return {
            'mean_rr': 0.0,
            'std_rr': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0
        }
    
    # Filtrar intervalos anómalos
    rr_filtered, _ = filter_rr_intervals(rr_intervals)
    
    if len(rr_filtered) < 2:
        return {
            'mean_rr': 0.0,
            'std_rr': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0
        }
    
    # Media y desviación estándar
    mean_rr = np.mean(rr_filtered)
    std_rr = np.std(rr_filtered)
    
    # RMSSD: Raíz cuadrada de la media de diferencias al cuadrado
    if len(rr_filtered) > 1:
        differences = np.diff(rr_filtered)
        rmssd = np.sqrt(np.mean(differences ** 2))
    else:
        rmssd = 0.0
    
    # pNN50: Porcentaje de intervalos que difieren >50ms
    if len(rr_filtered) > 1:
        differences = np.abs(np.diff(rr_filtered))
        pnn50 = (np.sum(differences > 50.0) / len(differences)) * 100.0
    else:
        pnn50 = 0.0
    
    return {
        'mean_rr': mean_rr,
        'std_rr': std_rr,
        'rmssd': rmssd,
        'pnn50': pnn50
    }


# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Análisis de Tacograma y HRV")
    print("=" * 50)
    
    # Crear ejemplo de picos R simulados
    fs = 250  # Hz
    duration = 60  # segundos
    heart_rate = 70  # bpm (variable)
    
    # Simular picos R con variabilidad
    rr_interval_base = 60.0 / heart_rate  # segundos
    r_peaks = []
    current_time = 0.0
    
    while current_time < duration:
        # Agregar variabilidad aleatoria al intervalo RR
        rr_variation = np.random.normal(0, 0.05)  # 5% de variación
        rr_interval = rr_interval_base * (1 + rr_variation)
        
        current_time += rr_interval
        if current_time < duration:
            r_peaks.append(int(current_time * fs))
    
    r_peaks = np.array(r_peaks)
    
    print(f"📊 Datos de ejemplo:")
    print(f"   Frecuencia de muestreo: {fs} Hz")
    print(f"   Duración: {duration} segundos")
    print(f"   Picos R simulados: {len(r_peaks)}")
    
    # Calcular tacograma
    tachogram_result = calculate_tachogram(r_peaks, fs, filter_anomalous=True)
    
    print(f"\n📊 Resultados del Tacograma:")
    print(f"   Intervalos RR calculados: {len(tachogram_result['rr_intervals'])}")
    print(f"   Frecuencia cardíaca global: {tachogram_result['heart_rate_bpm']:.2f} bpm")
    print(f"   Intervalo RR promedio: {tachogram_result['metadata']['mean_rr_ms']:.2f} ms")
    print(f"   Desviación estándar RR: {tachogram_result['metadata']['std_rr_ms']:.2f} ms")
    
    # Calcular métricas de HRV
    hrv_metrics = calculate_hrv_time_domain(tachogram_result['rr_intervals'])
    
    print(f"\n📊 Métricas de HRV (dominio del tiempo):")
    print(f"   SDNN: {hrv_metrics['std_rr']:.2f} ms")
    print(f"   RMSSD: {hrv_metrics['rmssd']:.2f} ms")
    print(f"   pNN50: {hrv_metrics['pnn50']:.2f} %")
    
    print(f"\n💡 Para usar en otros módulos:")
    print(f"   from src.tachogram_analysis import calculate_tachogram, calculate_global_heart_rate")
    print(f"   tachogram = calculate_tachogram(r_peaks, fs)")
    print(f"   hr_bpm = calculate_global_heart_rate(rr_intervals)")

