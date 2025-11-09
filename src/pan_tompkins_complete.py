"""
Implementación completa del algoritmo Pan-Tompkins para detección de picos R en ECG
Basado en: Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm"

Incluye:
- Diferenciación usando filtro FIR
- Integración usando filtro FIR con ventana móvil
- Umbralización estadística adaptativa
- Detección de picos R usando find_peaks
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Optional, Tuple
import warnings


def differentiate_signal(ecg_signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diferenciación de señal ECG usando filtro FIR
    
    El filtro de diferenciación enfatiza los picos R y reduce ruido de baja frecuencia.
    Coeficientes basados en el algoritmo Pan-Tompkins original.
    
    Args:
        ecg_signal: Señal ECG 1D
        fs: Frecuencia de muestreo
    
    Returns:
        Tuple con (señal_diferenciada, coeficientes_b)
    """
    # Coeficientes del filtro de diferenciación (FIR)
    # b = [-1, -2, 0, 2, 1] / 8
    b = np.array([-1, -2, 0, 2, 1]) / 8.0
    
    # Para filtro FIR, a = 1
    a = np.array([1.0])
    
    # Aplicar filtro FIR usando scipy.signal.lfilter (filter no existe, usar lfilter)
    differentiated = signal.lfilter(b, a, ecg_signal)
    
    return differentiated, b


def integrate_signal(ecg_signal: np.ndarray, fs: float, 
                    window_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integración de señal usando filtro FIR con ventana móvil rectangular
    
    La integración suaviza la señal y reduce falsos positivos en la detección.
    
    Args:
        ecg_signal: Señal ECG 1D (típicamente la señal diferenciada al cuadrado)
        fs: Frecuencia de muestreo
        window_size: Tamaño de ventana en segundos (default: 0.15s = 150ms)
    
    Returns:
        Tuple con (señal_integrada, coeficientes_b)
    """
    if window_size is None:
        window_size = 0.15  # 150ms por defecto
    
    # Calcular tamaño de ventana en muestras
    N = int(fs * window_size)
    
    # Asegurar que N sea impar para mejor comportamiento
    if N % 2 == 0:
        N += 1
    
    # Coeficientes del filtro de integración (ventana rectangular)
    # b = [1, 1, 1, ..., 1] / N
    b = np.ones(N) / float(N)
    
    # Para filtro FIR, a = 1
    a = np.array([1.0])
    
    # Aplicar filtro FIR usando scipy.signal.lfilter (filter no existe, usar lfilter)
    integrated = signal.lfilter(b, a, ecg_signal)
    
    return integrated, b


def statistical_threshold(signal_integrated: np.ndarray, 
                         method: str = 'adaptive',
                         k: float = 0.5,
                         window_size: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Umbralización estadística sobre señal integrada
    
    Calcula un umbral adaptativo basado en estadísticas de la señal.
    El umbral se actualiza dinámicamente con nuevos latidos detectados.
    
    Args:
        signal_integrated: Señal integrada (después de diferenciación, cuadrado e integración)
        method: Método de umbralización ('adaptive', 'fixed', 'percentile')
        k: Factor multiplicador para desviación estándar (default: 0.5)
        window_size: Tamaño de ventana para cálculo adaptativo (muestras)
    
    Returns:
        Tuple con (umbral, señal_umbralizada)
    """
    if len(signal_integrated) == 0:
        return 0.0, signal_integrated
    
    if method == 'adaptive':
        # Umbral adaptativo mejorado: usar percentil en lugar de media+std
        # Esto es más robusto ante outliers
        max_val = np.max(signal_integrated)
        median_val = np.median(signal_integrated)
        
        # Usar percentil 60-70 como base, más robusto que media
        percentile_base = np.percentile(signal_integrated, 65)
        
        # Calcular umbral considerando la distribución
        # Si hay picos claros, el umbral debe estar entre el percentil y el máximo
        threshold = percentile_base + k * (max_val - percentile_base) * 0.3
        
        # Asegurar que el umbral sea razonable (al menos 20% del máximo)
        min_threshold = max_val * 0.2
        threshold = max(threshold, min_threshold)
        
        # También asegurar que no sea demasiado alto (máximo 60% del máximo)
        max_threshold = max_val * 0.6
        threshold = min(threshold, max_threshold)
        
    elif method == 'fixed':
        # Umbral fijo basado en percentil
        threshold = np.percentile(signal_integrated, 75)
    elif method == 'percentile':
        # Usar percentil específico
        threshold = np.percentile(signal_integrated, 70)
    else:
        raise ValueError(f"Método de umbralización desconocido: {method}")
    
    # Señal umbralizada (mantener valores originales, solo para visualización)
    signal_thresholded = signal_integrated.copy()
    
    return threshold, signal_thresholded


def pan_tompkins_complete(ecg_signal: np.ndarray, 
                         fs: float,
                         visualize: bool = False,
                         threshold_k: float = 0.5,
                         min_distance: Optional[float] = None) -> Dict:
    """
    Implementación completa del algoritmo Pan-Tompkins para detección de picos R
    
    Pasos del algoritmo:
    1. Diferenciación (filtro FIR)
    2. Cuadrado
    3. Integración (filtro FIR con ventana móvil)
    4. Umbralización estadística
    5. Detección de picos usando find_peaks
    
    Args:
        ecg_signal: Señal ECG 1D
        fs: Frecuencia de muestreo
        visualize: Si retornar señales intermedias para visualización
        threshold_k: Factor k para umbralización estadística (default: 0.5)
        min_distance: Distancia mínima entre picos en segundos (default: 0.2s = 200ms)
    
    Returns:
        Dict con:
            - r_peaks: Índices de picos R detectados
            - signals: Diccionario con señales intermedias (si visualize=True)
            - thresholds: Umbrales utilizados
            - metadata: Información adicional del procesamiento
    """
    # Validación de entrada
    if len(ecg_signal) == 0:
        warnings.warn("Señal ECG vacía")
        return {
            'r_peaks': np.array([], dtype=int),
            'signals': {} if not visualize else {
                'original': ecg_signal,
                'differentiated': np.array([]),
                'squared': np.array([]),
                'integrated': np.array([]),
                'thresholded': np.array([])
            },
            'thresholds': {},
            'metadata': {'error': 'empty_signal'}
        }
    
    if fs <= 0:
        raise ValueError(f"Frecuencia de muestreo debe ser positiva, recibido: {fs}")
    
    if len(ecg_signal) < int(0.5 * fs):  # Mínimo 0.5 segundos
        warnings.warn(f"Señal muy corta ({len(ecg_signal)} muestras), puede afectar detección")
    
    # Configurar distancia mínima entre picos
    if min_distance is None:
        min_distance = 0.2  # 200ms por defecto
    min_distance_samples = int(min_distance * fs)
    
    # Paso 1: Diferenciación usando filtro FIR
    signal_diff, b_diff = differentiate_signal(ecg_signal, fs)
    
    # Paso 2: Cuadrado (hacer todos los valores positivos y amplificar picos)
    signal_squared = signal_diff ** 2
    
    # Paso 3: Integración usando filtro FIR con ventana móvil
    signal_integrated, b_int = integrate_signal(signal_squared, fs)
    
    # Paso 4: Umbralización estadística
    threshold, signal_thresholded = statistical_threshold(
        signal_integrated, 
        method='adaptive',
        k=threshold_k
    )
    
    # Paso 5: Detección de picos R usando find_peaks
    # Usar umbral adaptativo calculado con parámetros adicionales para mejor precisión
    max_integrated = np.max(signal_integrated)
    median_integrated = np.median(signal_integrated)
    
    # Calcular prominencia mínima de forma adaptativa
    # Basado en la diferencia entre máximo y mediana
    signal_range = max_integrated - median_integrated
    min_prominence = max(signal_range * 0.15, max_integrated * 0.05)  # Al menos 15% del rango o 5% del máximo
    
    # Calcular ancho mínimo del pico (en muestras)
    min_width = max(1, int(0.02 * fs))  # Mínimo 20ms de ancho
    
    peaks, properties = signal.find_peaks(
        signal_integrated,
        height=threshold,
        distance=min_distance_samples,
        prominence=min_prominence,  # Asegurar que los picos tienen suficiente prominencia
        width=min_width  # Asegurar que los picos tienen un ancho mínimo
    )
    
    # Post-procesamiento: Refinar detección buscando el máximo ABSOLUTO en la señal original
    # Esto corrige desplazamientos causados por la integración y asegura que los picos
    # detectados correspondan a los picos R reales en la señal original
    refined_peaks = []
    # Ventana de búsqueda amplia para capturar todo el complejo QRS completo
    # El complejo QRS típicamente tiene 80-120ms de ancho, así que usamos 150ms para estar seguros
    search_window = max(int(0.15 * fs), 25)  # 150ms o mínimo 25 muestras
    
    # Calcular estadísticas globales de la señal para validación
    signal_max = np.max(ecg_signal)
    signal_min = np.min(ecg_signal)
    signal_range = signal_max - signal_min
    signal_median = np.median(ecg_signal)
    # Umbral mínimo más flexible: debe estar en el 40% superior del rango
    # Esto asegura que seleccionamos picos prominentes, no pequeñas deflexiones
    min_peak_threshold = signal_min + 0.6 * signal_range
    
    for peak in peaks:
        # Buscar el máximo ABSOLUTO en la señal original alrededor del pico detectado
        # Usar ventana centrada en el pico detectado
        start_idx = max(0, peak - search_window)
        end_idx = min(len(ecg_signal), peak + search_window)
        
        if end_idx > start_idx:
            search_region = ecg_signal[start_idx:end_idx]
            
            # SIEMPRE buscar el máximo absoluto en la ventana (no máximos locales)
            # Esto asegura que encontremos el verdadero pico R del complejo QRS
            local_max_idx = np.argmax(search_region)
            refined_peak = start_idx + local_max_idx
            peak_value = ecg_signal[refined_peak]
            
            # Validación mejorada: verificar prominencia relativa dentro de la ventana
            # El pico debe ser significativamente mayor que el mínimo en la ventana
            region_min = np.min(search_region)
            region_range = peak_value - region_min
            # El pico debe tener al menos 50% del rango de la ventana
            prominence_ratio = region_range / (signal_range + 1e-10)  # Evitar división por cero
            
            # Validación: el pico debe tener suficiente amplitud y prominencia
            if peak_value >= min_peak_threshold or prominence_ratio > 0.3:
                # Verificar que es realmente un máximo local (mayor que sus vecinos)
                check_window = max(5, int(0.02 * fs))  # 20ms o mínimo 5 muestras
                left_check = max(0, refined_peak - check_window)
                right_check = min(len(ecg_signal), refined_peak + check_window + 1)
                
                neighbors = ecg_signal[left_check:right_check]
                if len(neighbors) > 0:
                    # El pico debe ser el máximo en su vecindad
                    if peak_value >= np.max(neighbors):
                        refined_peaks.append(refined_peak)
                else:
                    refined_peaks.append(refined_peak)
            # Si no cumple, intentar buscar en una ventana más amplia
            elif len(refined_peaks) == 0 or (refined_peak - refined_peaks[-1] if len(refined_peaks) > 0 else 0) > min_distance_samples:
                # Buscar en ventana más amplia (200ms)
                wider_window = max(int(0.2 * fs), 30)
                wider_start = max(0, peak - wider_window)
                wider_end = min(len(ecg_signal), peak + wider_window)
                if wider_end > wider_start:
                    wider_region = ecg_signal[wider_start:wider_end]
                    wider_max_idx = np.argmax(wider_region)
                    wider_peak = wider_start + wider_max_idx
                    wider_value = ecg_signal[wider_peak]
                    wider_region_min = np.min(wider_region)
                    wider_prominence = (wider_value - wider_region_min) / (signal_range + 1e-10)
                    if wider_value >= min_peak_threshold or wider_prominence > 0.3:
                        refined_peaks.append(wider_peak)
    
    # Convertir a array y ordenar
    refined_peaks = np.array(refined_peaks, dtype=int)
    if len(refined_peaks) > 0:
        refined_peaks = np.sort(refined_peaks)
        
        # Filtrar picos que están demasiado cerca (pueden ser duplicados)
        if len(refined_peaks) > 1:
            # Calcular distancias entre picos consecutivos
            peak_distances = np.diff(refined_peaks)
            # Mantener solo picos con distancia mínima
            valid_mask = np.ones(len(refined_peaks), dtype=bool)
            for i in range(len(peak_distances)):
                if peak_distances[i] < min_distance_samples:
                    # Mantener el pico con mayor amplitud
                    if ecg_signal[refined_peaks[i]] < ecg_signal[refined_peaks[i+1]]:
                        valid_mask[i] = False
                    else:
                        valid_mask[i+1] = False
            refined_peaks = refined_peaks[valid_mask]
    
    peaks = refined_peaks
    
    # Preparar resultado
    result = {
        'r_peaks': peaks,
        'thresholds': {
            'adaptive_threshold': threshold,
            'threshold_k': threshold_k
        },
        'metadata': {
            'n_peaks_detected': len(peaks),
            'fs': fs,
            'signal_length': len(ecg_signal),
            'min_distance_samples': min_distance_samples,
            'differentiation_coeffs': b_diff.tolist(),
            'integration_window_size': len(b_int)
        }
    }
    
    # Agregar señales intermedias si se solicita visualización
    if visualize:
        result['signals'] = {
            'original': ecg_signal,
            'differentiated': signal_diff,
            'squared': signal_squared,
            'integrated': signal_integrated,
            'thresholded': signal_thresholded
        }
    
    return result


# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Algoritmo Pan-Tompkins Completo")
    print("=" * 50)
    
    # Crear señal de ejemplo
    fs = 250  # Hz
    duration = 10  # segundos
    t = np.linspace(0, duration, int(fs * duration))
    
    # Señal ECG simulada con latidos
    ecg_simulated = np.zeros(len(t))
    heart_rate = 70  # bpm
    rr_interval = 60.0 / heart_rate  # segundos
    
    for i in range(int(duration / rr_interval)):
        beat_time = i * rr_interval
        beat_idx = int(beat_time * fs)
        
        if beat_idx < len(t):
            # Simular complejo QRS
            qrs_duration = 0.1  # 100ms
            qrs_samples = int(qrs_duration * fs)
            
            for j in range(qrs_samples):
                idx = beat_idx + j
                if idx < len(t):
                    # Forma de onda QRS simplificada
                    ecg_simulated[idx] = np.sin(2 * np.pi * 10 * (t[idx] - beat_time)) * \
                                        np.exp(-5 * (t[idx] - beat_time))
    
    # Agregar ruido
    ecg_simulated += 0.1 * np.random.randn(len(t))
    
    print(f"📊 Señal de ejemplo:")
    print(f"   Longitud: {len(ecg_simulated)} muestras")
    print(f"   Frecuencia de muestreo: {fs} Hz")
    print(f"   Duración: {duration} segundos")
    
    # Aplicar Pan-Tompkins completo
    result = pan_tompkins_complete(ecg_simulated, fs, visualize=True)
    
    print(f"\n📊 Resultados:")
    print(f"   Picos R detectados: {len(result['r_peaks'])}")
    print(f"   Umbral utilizado: {result['thresholds']['adaptive_threshold']:.4f}")
    print(f"   Primeros 5 picos R (índices): {result['r_peaks'][:5]}")
    
    print(f"\n💡 Para usar en otros módulos:")
    print(f"   from src.pan_tompkins_complete import pan_tompkins_complete")
    print(f"   result = pan_tompkins_complete(signal, fs, visualize=True)")

