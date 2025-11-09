"""
Demo interactivo del algoritmo Pan-Tompkins completo
Demuestra detección de picos R, ondas P/Q/S/T, y análisis de tacograma
"""

import sys
from pathlib import Path
import numpy as np

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from src.pan_tompkins_complete import pan_tompkins_complete
from src.ecg_wave_detection import detect_all_waves
from src.tachogram_analysis import calculate_tachogram, calculate_hrv_time_domain
from src.utils import load_ecg_record
from scripts.visualize_pan_tompkins import (
    visualize_pan_tompkins_steps,
    visualize_detected_waves,
    visualize_tachogram
)


def demo_with_synthetic_signal():
    """Demo con señal sintética"""
    print("=" * 70)
    print("DEMO: Algoritmo Pan-Tompkins Completo - Señal Sintética")
    print("=" * 70)
    
    # Crear señal ECG sintética
    fs = 250  # Hz
    duration = 10  # segundos
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simular ECG con latidos
    ecg_signal = np.zeros(len(t))
    heart_rate = 75  # bpm
    rr_interval = 60.0 / heart_rate
    
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
                    # Forma de onda QRS
                    ecg_signal[idx] = np.sin(2 * np.pi * 10 * (t[idx] - beat_time)) * \
                                     np.exp(-5 * (t[idx] - beat_time))
    
    # Agregar ruido
    ecg_signal += 0.1 * np.random.randn(len(t))
    
    print(f"\n📊 Señal generada:")
    print(f"   Longitud: {len(ecg_signal)} muestras")
    print(f"   Frecuencia de muestreo: {fs} Hz")
    print(f"   Duración: {duration} segundos")
    
    # Aplicar Pan-Tompkins completo
    print(f"\n🔍 Aplicando algoritmo Pan-Tompkins...")
    result = pan_tompkins_complete(ecg_signal, fs, visualize=True)
    
    print(f"   ✅ Picos R detectados: {len(result['r_peaks'])}")
    print(f"   ✅ Umbral utilizado: {result['thresholds']['adaptive_threshold']:.4f}")
    
    # Detectar ondas
    print(f"\n🔍 Detectando ondas P, Q, S, T...")
    waves_result = detect_all_waves(ecg_signal, result['r_peaks'], fs)
    
    p_count = sum(1 for p in waves_result['p_waves'] if p is not None)
    q_count = sum(1 for q in waves_result['q_waves'] if q is not None)
    s_count = sum(1 for s in waves_result['s_waves'] if s is not None)
    t_count = sum(1 for t in waves_result['t_waves'] if t is not None)
    
    print(f"   ✅ Ondas P: {p_count}")
    print(f"   ✅ Ondas Q: {q_count}")
    print(f"   ✅ Ondas S: {s_count}")
    print(f"   ✅ Ondas T: {t_count}")
    
    # Calcular tacograma
    print(f"\n🔍 Calculando tacograma...")
    tachogram_result = calculate_tachogram(result['r_peaks'], fs)
    
    print(f"   ✅ Intervalos RR: {len(tachogram_result['rr_intervals'])}")
    print(f"   ✅ Frecuencia cardíaca: {tachogram_result['heart_rate_bpm']:.2f} bpm")
    print(f"   ✅ RR promedio: {tachogram_result['metadata']['mean_rr_ms']:.2f} ms")
    
    # Calcular métricas HRV
    hrv_metrics = calculate_hrv_time_domain(tachogram_result['rr_intervals'])
    print(f"\n📊 Métricas HRV:")
    print(f"   SDNN: {hrv_metrics['std_rr']:.2f} ms")
    print(f"   RMSSD: {hrv_metrics['rmssd']:.2f} ms")
    print(f"   pNN50: {hrv_metrics['pnn50']:.2f} %")
    
    # Generar visualizaciones
    print(f"\n📊 Generando visualizaciones...")
    
    # Preparar señales para visualización
    signals_dict = result['signals'].copy()
    signals_dict['threshold'] = result['thresholds']['adaptive_threshold']
    
    # Visualización paso a paso
    fig_steps = visualize_pan_tompkins_steps(
        signals_dict, fs, result['r_peaks'],
        duration=5.0  # Mostrar primeros 5 segundos
    )
    output_steps = "results/pan_tompkins_steps.html"
    fig_steps.write_html(output_steps)
    print(f"   ✅ Pasos del algoritmo: {output_steps}")
    
    # Visualización de ondas
    waves_dict = waves_result.copy()
    waves_dict['r_peaks'] = result['r_peaks']
    fig_waves = visualize_detected_waves(
        ecg_signal, waves_dict, fs, duration=5.0
    )
    output_waves = "results/pan_tompkins_waves.html"
    fig_waves.write_html(output_waves)
    print(f"   ✅ Ondas detectadas: {output_waves}")
    
    # Visualización de tacograma
    fig_tacho = visualize_tachogram(tachogram_result)
    output_tacho = "results/pan_tompkins_tachogram.html"
    fig_tacho.write_html(output_tacho)
    print(f"   ✅ Tacograma: {output_tacho}")
    
    print(f"\n✅ Demo completado exitosamente!")
    print(f"   Visualizaciones guardadas en: results/")
    
    return result, waves_result, tachogram_result


def demo_with_real_signal(record_path: str = None):
    """Demo con señal real del dataset"""
    print("=" * 70)
    print("DEMO: Algoritmo Pan-Tompkins Completo - Señal Real")
    print("=" * 70)
    
    # Intentar cargar señal real
    if record_path is None:
        # Buscar un registro disponible
        from src.utils import list_available_records
        
        datasets = ['sddb', 'nsrdb', 'cudb']
        records = []
        
        for dataset in datasets:
            dataset_records = list_available_records(f'datasets/{dataset}')
            if dataset_records:
                records.extend([(dataset, r) for r in dataset_records[:1]])
                break
        
        if not records:
            print("⚠️  No se encontraron registros en el dataset.")
            print("   Usando señal sintética en su lugar...")
            return demo_with_synthetic_signal()
        
        dataset, record_name = records[0]
        record_path = f'datasets/{dataset}/{record_name}'
        print(f"📂 Usando registro: {record_path}")
    
    try:
        # Cargar señal
        signal, metadata = load_ecg_record(record_path, channels=[0])
        fs = metadata['fs']
        
        # Limitar a primeros 30 segundos para demo
        max_duration = 30  # segundos
        max_samples = int(max_duration * fs)
        if signal.shape[0] > max_samples:
            signal = signal[:max_samples, :]
        
        ecg_signal = signal[:, 0]  # Primer canal
        
        print(f"\n📊 Señal cargada:")
        print(f"   Registro: {record_path}")
        print(f"   Longitud: {len(ecg_signal)} muestras")
        print(f"   Frecuencia de muestreo: {fs} Hz")
        print(f"   Duración: {len(ecg_signal) / fs:.1f} segundos")
        
        # Preprocesar señal (filtrado básico)
        from src.preprocessing_unified import bandpass_filter, normalize_signal
        ecg_filtered = bandpass_filter(ecg_signal, fs)
        ecg_filtered = normalize_signal(ecg_filtered, method='zscore')
        
        # Aplicar Pan-Tompkins completo
        print(f"\n🔍 Aplicando algoritmo Pan-Tompkins...")
        result = pan_tompkins_complete(ecg_filtered, fs, visualize=True)
        
        print(f"   ✅ Picos R detectados: {len(result['r_peaks'])}")
        print(f"   ✅ Umbral utilizado: {result['thresholds']['adaptive_threshold']:.4f}")
        
        if len(result['r_peaks']) < 2:
            print("⚠️  Pocos picos R detectados. Continuando con demo sintética...")
            return demo_with_synthetic_signal()
        
        # Detectar ondas
        print(f"\n🔍 Detectando ondas P, Q, S, T...")
        waves_result = detect_all_waves(ecg_filtered, result['r_peaks'], fs)
        
        p_count = sum(1 for p in waves_result['p_waves'] if p is not None)
        q_count = sum(1 for q in waves_result['q_waves'] if q is not None)
        s_count = sum(1 for s in waves_result['s_waves'] if s is not None)
        t_count = sum(1 for t in waves_result['t_waves'] if t is not None)
        
        print(f"   ✅ Ondas P: {p_count}")
        print(f"   ✅ Ondas Q: {q_count}")
        print(f"   ✅ Ondas S: {s_count}")
        print(f"   ✅ Ondas T: {t_count}")
        
        # Calcular tacograma
        print(f"\n🔍 Calculando tacograma...")
        tachogram_result = calculate_tachogram(result['r_peaks'], fs)
        
        print(f"   ✅ Intervalos RR: {len(tachogram_result['rr_intervals'])}")
        print(f"   ✅ Frecuencia cardíaca: {tachogram_result['heart_rate_bpm']:.2f} bpm")
        print(f"   ✅ RR promedio: {tachogram_result['metadata']['mean_rr_ms']:.2f} ms")
        
        # Calcular métricas HRV
        hrv_metrics = calculate_hrv_time_domain(tachogram_result['rr_intervals'])
        print(f"\n📊 Métricas HRV:")
        print(f"   SDNN: {hrv_metrics['std_rr']:.2f} ms")
        print(f"   RMSSD: {hrv_metrics['rmssd']:.2f} ms")
        print(f"   pNN50: {hrv_metrics['pnn50']:.2f} %")
        
        # Generar visualizaciones
        print(f"\n📊 Generando visualizaciones...")
        
        # Preparar señales para visualización
        signals_dict = result['signals'].copy()
        signals_dict['threshold'] = result['thresholds']['adaptive_threshold']
        
        # Visualización paso a paso (primeros 10 segundos)
        fig_steps = visualize_pan_tompkins_steps(
            signals_dict, fs, result['r_peaks'],
            duration=10.0
        )
        output_steps = "results/pan_tompkins_steps_real.html"
        fig_steps.write_html(output_steps)
        print(f"   ✅ Pasos del algoritmo: {output_steps}")
        
        # Visualización de ondas
        waves_dict = waves_result.copy()
        waves_dict['r_peaks'] = result['r_peaks']
        fig_waves = visualize_detected_waves(
            ecg_filtered, waves_dict, fs, duration=10.0
        )
        output_waves = "results/pan_tompkins_waves_real.html"
        fig_waves.write_html(output_waves)
        print(f"   ✅ Ondas detectadas: {output_waves}")
        
        # Visualización de tacograma
        fig_tacho = visualize_tachogram(tachogram_result)
        output_tacho = "results/pan_tompkins_tachogram_real.html"
        fig_tacho.write_html(output_tacho)
        print(f"   ✅ Tacograma: {output_tacho}")
        
        print(f"\n✅ Demo completado exitosamente!")
        print(f"   Visualizaciones guardadas en: results/")
        
        return result, waves_result, tachogram_result
        
    except Exception as e:
        print(f"⚠️  Error cargando señal real: {e}")
        print("   Usando señal sintética en su lugar...")
        return demo_with_synthetic_signal()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo del algoritmo Pan-Tompkins completo')
    parser.add_argument('--real', action='store_true',
                       help='Usar señal real del dataset (si está disponible)')
    parser.add_argument('--record', type=str, default=None,
                       help='Ruta al registro específico a usar')
    
    args = parser.parse_args()
    
    # Crear directorio de resultados si no existe
    Path("results").mkdir(exist_ok=True)
    
    if args.real or args.record:
        demo_with_real_signal(args.record)
    else:
        demo_with_synthetic_signal()


if __name__ == "__main__":
    main()

