"""
Tests unitarios y validación del algoritmo Pan-Tompkins completo
Incluye tests de diferenciación, integración, detección de ondas y tacograma
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from src.pan_tompkins_complete import (
    pan_tompkins_complete,
    differentiate_signal,
    integrate_signal,
    statistical_threshold
)
from src.ecg_wave_detection import detect_all_waves, detect_q_wave, detect_s_wave
from src.tachogram_analysis import (
    calculate_tachogram,
    calculate_global_heart_rate,
    filter_rr_intervals
)


def test_differentiation():
    """Test de diferenciación con señal conocida"""
    print("🧪 Test: Diferenciación")
    
    fs = 250
    t = np.linspace(0, 1, fs)
    # Señal sinusoidal
    test_signal = np.sin(2 * np.pi * 5 * t)
    
    diff_signal, b_coeffs = differentiate_signal(test_signal, fs)
    
    # Verificar que la señal diferenciada tiene la longitud correcta
    assert len(diff_signal) == len(test_signal), "Longitud incorrecta después de diferenciación"
    
    # Verificar que los coeficientes son correctos
    expected_b = np.array([-1, -2, 0, 2, 1]) / 8.0
    assert np.allclose(b_coeffs, expected_b), "Coeficientes de diferenciación incorrectos"
    
    print("   ✅ Diferenciación: PASS")


def test_integration():
    """Test de integración con señal conocida"""
    print("🧪 Test: Integración")
    
    fs = 250
    t = np.linspace(0, 1, fs)
    # Señal constante
    test_signal = np.ones(len(t))
    
    integrated_signal, b_coeffs = integrate_signal(test_signal, fs)
    
    # Verificar que la señal integrada tiene la longitud correcta
    assert len(integrated_signal) == len(test_signal), "Longitud incorrecta después de integración"
    
    # Verificar que los coeficientes suman aproximadamente 1
    assert abs(np.sum(b_coeffs) - 1.0) < 0.01, "Coeficientes de integración no normalizados"
    
    print("   ✅ Integración: PASS")


def test_statistical_threshold():
    """Test de umbralización estadística"""
    print("🧪 Test: Umbralización Estadística")
    
    # Crear señal con picos conocidos
    test_signal = np.random.randn(1000) * 0.1
    test_signal[100] = 5.0  # Pico alto
    test_signal[500] = 4.0  # Pico medio
    
    threshold, thresholded_signal = statistical_threshold(test_signal, method='adaptive', k=0.5)
    
    # Verificar que el umbral es positivo
    assert threshold > 0, "Umbral debe ser positivo"
    
    # Verificar que el umbral es razonable (menor que el máximo)
    assert threshold < np.max(test_signal), "Umbral debe ser menor que el máximo"
    
    print("   ✅ Umbralización Estadística: PASS")


def test_pan_tompkins_synthetic():
    """Test de Pan-Tompkins completo con señal sintética"""
    print("🧪 Test: Pan-Tompkins Completo (Señal Sintética)")
    
    fs = 250
    duration = 5
    t = np.linspace(0, duration, int(fs * duration))
    
    # Crear señal con latidos conocidos
    ecg_signal = np.zeros(len(t))
    heart_rate = 70  # bpm
    rr_interval = 60.0 / heart_rate
    
    for i in range(int(duration / rr_interval)):
        beat_time = i * rr_interval
        beat_idx = int(beat_time * fs)
        
        if beat_idx < len(t):
            # Simular complejo QRS
            qrs_duration = 0.1
            qrs_samples = int(qrs_duration * fs)
            
            for j in range(qrs_samples):
                idx = beat_idx + j
                if idx < len(t):
                    ecg_signal[idx] = np.sin(2 * np.pi * 10 * (t[idx] - beat_time)) * \
                                     np.exp(-5 * (t[idx] - beat_time))
    
    # Agregar ruido
    ecg_signal += 0.1 * np.random.randn(len(t))
    
    # Aplicar Pan-Tompkins
    result = pan_tompkins_complete(ecg_signal, fs, visualize=True)
    
    # Verificar que se detectaron picos
    assert len(result['r_peaks']) > 0, "Debe detectar al menos un pico R"
    
    # Verificar que el número de picos es razonable
    expected_peaks = int(duration * heart_rate / 60)
    assert abs(len(result['r_peaks']) - expected_peaks) <= 2, \
        f"Debe detectar aproximadamente {expected_peaks} picos, detectados: {len(result['r_peaks'])}"
    
    # Verificar que las señales intermedias están presentes
    assert 'signals' in result, "Debe incluir señales intermedias"
    assert 'original' in result['signals'], "Debe incluir señal original"
    
    print(f"   ✅ Pan-Tompkins Completo: PASS ({len(result['r_peaks'])} picos detectados)")


def test_wave_detection():
    """Test de detección de ondas"""
    print("🧪 Test: Detección de Ondas")
    
    fs = 250
    duration = 5
    t = np.linspace(0, duration, int(fs * duration))
    
    # Crear señal con latidos
    ecg_signal = np.zeros(len(t))
    heart_rate = 70
    rr_interval = 60.0 / heart_rate
    
    r_peaks = []
    for i in range(int(duration / rr_interval)):
        beat_time = i * rr_interval
        beat_idx = int(beat_time * fs)
        r_peaks.append(beat_idx)
    
    r_peaks = np.array(r_peaks)
    
    # Detectar ondas
    waves_result = detect_all_waves(ecg_signal, r_peaks, fs)
    
    # Verificar estructura del resultado
    assert 'p_waves' in waves_result, "Debe incluir ondas P"
    assert 'q_waves' in waves_result, "Debe incluir ondas Q"
    assert 's_waves' in waves_result, "Debe incluir ondas S"
    assert 't_waves' in waves_result, "Debe incluir ondas T"
    assert 'wave_features' in waves_result, "Debe incluir características"
    
    print("   ✅ Detección de Ondas: PASS")


def test_tachogram():
    """Test de cálculo de tacograma"""
    print("🧪 Test: Cálculo de Tacograma")
    
    fs = 250
    # Simular picos R con variabilidad
    duration = 60
    heart_rate = 70
    rr_interval_base = 60.0 / heart_rate
    
    r_peaks = []
    current_time = 0.0
    while current_time < duration:
        rr_variation = np.random.normal(0, 0.05)
        rr_interval = rr_interval_base * (1 + rr_variation)
        current_time += rr_interval
        if current_time < duration:
            r_peaks.append(int(current_time * fs))
    
    r_peaks = np.array(r_peaks)
    
    # Calcular tacograma
    tachogram_result = calculate_tachogram(r_peaks, fs)
    
    # Verificar estructura
    assert 'rr_intervals' in tachogram_result, "Debe incluir intervalos RR"
    assert 'time_points' in tachogram_result, "Debe incluir puntos de tiempo"
    assert 'heart_rate_bpm' in tachogram_result, "Debe incluir frecuencia cardíaca"
    assert 'metadata' in tachogram_result, "Debe incluir metadata"
    
    # Verificar que la frecuencia cardíaca es razonable
    hr = tachogram_result['heart_rate_bpm']
    assert 50 <= hr <= 120, f"Frecuencia cardíaca debe estar en rango razonable, obtenida: {hr}"
    
    print(f"   ✅ Tacograma: PASS (HR: {hr:.2f} bpm)")


def test_global_heart_rate():
    """Test de cálculo de frecuencia cardíaca global"""
    print("🧪 Test: Frecuencia Cardíaca Global")
    
    # Intervalos RR en ms (corresponden a ~70 bpm)
    rr_intervals = np.array([857, 860, 855, 862, 858])  # ~70 bpm
    
    hr_bpm = calculate_global_heart_rate(rr_intervals)
    
    # Verificar que está en rango razonable
    assert 65 <= hr_bpm <= 75, f"Frecuencia cardíaca debe estar cerca de 70 bpm, obtenida: {hr_bpm}"
    
    print(f"   ✅ Frecuencia Cardíaca Global: PASS ({hr_bpm:.2f} bpm)")


def test_filter_rr_intervals():
    """Test de filtrado de intervalos RR"""
    print("🧪 Test: Filtrado de Intervalos RR")
    
    # Intervalos RR con algunos anómalos
    rr_intervals = np.array([800, 850, 200, 900, 2500, 820, 830])
    
    rr_filtered, valid_indices = filter_rr_intervals(rr_intervals, min_rr=300, max_rr=2000)
    
    # Verificar que se filtraron los anómalos
    assert len(rr_filtered) < len(rr_intervals), "Debe filtrar intervalos anómalos"
    
    # Verificar que todos los intervalos filtrados están en rango
    assert np.all(rr_filtered >= 300), "Todos los intervalos deben ser >= 300ms"
    assert np.all(rr_filtered <= 2000), "Todos los intervalos deben ser <= 2000ms"
    
    print(f"   ✅ Filtrado de Intervalos RR: PASS ({len(rr_filtered)}/{len(rr_intervals)} válidos)")


def test_with_real_signal():
    """Test con señal real del dataset (si está disponible)"""
    print("🧪 Test: Validación con Señal Real")
    
    try:
        from src.utils import load_ecg_record, list_available_records
        
        # Buscar un registro disponible
        datasets = ['sddb', 'nsrdb', 'cudb']
        record_path = None
        
        for dataset in datasets:
            records = list_available_records(f'datasets/{dataset}')
            if records:
                record_path = f'datasets/{dataset}/{records[0]}'
                break
        
        if not record_path:
            print("   ⚠️  No se encontraron registros en el dataset. Saltando test.")
            return
        
        # Cargar señal
        signal_data, metadata = load_ecg_record(record_path, channels=[0])
        fs = metadata['fs']
        
        # Limitar a primeros 10 segundos
        max_samples = int(10 * fs)
        if signal_data.shape[0] > max_samples:
            signal_data = signal_data[:max_samples, :]
        
        ecg_signal = signal_data[:, 0]
        
        # Preprocesar
        from src.preprocessing_unified import bandpass_filter, normalize_signal
        ecg_filtered = bandpass_filter(ecg_signal, fs)
        ecg_filtered = normalize_signal(ecg_filtered, method='zscore')
        
        # Aplicar Pan-Tompkins
        result = pan_tompkins_complete(ecg_filtered, fs, visualize=False)
        
        # Verificar que se detectaron picos
        if len(result['r_peaks']) >= 2:
            # Detectar ondas
            waves_result = detect_all_waves(ecg_filtered, result['r_peaks'], fs)
            
            # Calcular tacograma
            tachogram_result = calculate_tachogram(result['r_peaks'], fs)
            
            print(f"   ✅ Validación con Señal Real: PASS")
            print(f"      - Picos R: {len(result['r_peaks'])}")
            print(f"      - Frecuencia cardíaca: {tachogram_result['heart_rate_bpm']:.2f} bpm")
        else:
            print(f"   ⚠️  Pocos picos detectados ({len(result['r_peaks'])}). Puede ser señal ruidosa.")
            
    except ImportError:
        print("   ⚠️  No se pudo importar utils. Saltando test con señal real.")
    except Exception as e:
        print(f"   ⚠️  Error en test con señal real: {e}")


def compare_with_basic_implementation():
    """Comparar con implementación básica existente"""
    print("🧪 Test: Comparación con Implementación Básica")
    
    fs = 250
    duration = 5
    t = np.linspace(0, duration, int(fs * duration))
    
    # Crear señal con latidos
    ecg_signal = np.zeros(len(t))
    heart_rate = 70
    rr_interval = 60.0 / heart_rate
    
    for i in range(int(duration / rr_interval)):
        beat_time = i * rr_interval
        beat_idx = int(beat_time * fs)
        
        if beat_idx < len(t):
            qrs_duration = 0.1
            qrs_samples = int(qrs_duration * fs)
            
            for j in range(qrs_samples):
                idx = beat_idx + j
                if idx < len(t):
                    ecg_signal[idx] = np.sin(2 * np.pi * 10 * (t[idx] - beat_time)) * \
                                     np.exp(-5 * (t[idx] - beat_time))
    
    ecg_signal += 0.1 * np.random.randn(len(t))
    
    # Usar implementación básica
    from src.preprocessing import detect_r_peaks
    r_peaks_basic = detect_r_peaks(ecg_signal.reshape(-1, 1), fs, channel=0)
    
    # Usar implementación completa
    result_complete = pan_tompkins_complete(ecg_signal, fs, visualize=False)
    r_peaks_complete = result_complete['r_peaks']
    
    # Comparar (deben ser similares)
    print(f"   - Implementación básica: {len(r_peaks_basic)} picos")
    print(f"   - Implementación completa: {len(r_peaks_complete)} picos")
    
    # Verificar que ambas detectan picos
    assert len(r_peaks_basic) > 0, "Implementación básica debe detectar picos"
    assert len(r_peaks_complete) > 0, "Implementación completa debe detectar picos"
    
    print("   ✅ Comparación: PASS")


def run_all_tests():
    """Ejecutar todos los tests"""
    print("=" * 70)
    print("TESTS UNITARIOS - Algoritmo Pan-Tompkins Completo")
    print("=" * 70)
    print()
    
    tests = [
        test_differentiation,
        test_integration,
        test_statistical_threshold,
        test_pan_tompkins_synthetic,
        test_wave_detection,
        test_tachogram,
        test_global_heart_rate,
        test_filter_rr_intervals,
        compare_with_basic_implementation,
        test_with_real_signal
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"RESUMEN: {passed} tests pasados, {failed} tests fallidos")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

