"""
Script para generar resultados completos de Pan-Tompkins con señales reales
Incluye procesamiento, análisis, gráficas Plotly y métricas para el dashboard
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List
import warnings

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from src.pan_tompkins_complete import pan_tompkins_complete
from src.ecg_wave_detection import detect_all_waves
from src.tachogram_analysis import calculate_tachogram, calculate_hrv_time_domain
from src.utils import load_ecg_record, list_available_records
from src.preprocessing_unified import bandpass_filter, normalize_signal
from scripts.visualize_pan_tompkins import (
    visualize_pan_tompkins_steps,
    visualize_detected_waves,
    visualize_tachogram
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def process_signal_for_dashboard(record_path: str, label: str, max_duration: float = 30.0) -> Dict:
    """
    Procesar una señal para el dashboard
    
    Args:
        record_path: Ruta al registro
        label: Etiqueta ('Normal' o 'SCD')
        max_duration: Duración máxima a procesar en segundos
    
    Returns:
        Dict con resultados del procesamiento
    """
    try:
        # Cargar señal
        signal_data, metadata = load_ecg_record(record_path, channels=[0])
        fs = metadata['fs']
        
        # Limitar duración
        max_samples = int(max_duration * fs)
        if signal_data.shape[0] > max_samples:
            signal_data = signal_data[:max_samples, :]
        
        ecg_signal = signal_data[:, 0]
        
        # Preprocesar
        ecg_filtered = bandpass_filter(ecg_signal, fs)
        ecg_filtered = normalize_signal(ecg_filtered, method='zscore')
        
        # Aplicar Pan-Tompkins
        result = pan_tompkins_complete(ecg_filtered, fs, visualize=True, threshold_k=0.5)
        
        if len(result['r_peaks']) < 2:
            return None
        
        # Detectar ondas
        waves_result = detect_all_waves(ecg_filtered, result['r_peaks'], fs)
        
        # Calcular tacograma
        tachogram_result = calculate_tachogram(result['r_peaks'], fs)
        
        # Calcular métricas HRV
        hrv_metrics = calculate_hrv_time_domain(tachogram_result['rr_intervals'])
        
        # Preparar señales para visualización (primeros 10 segundos)
        viz_duration = min(10.0, len(ecg_filtered) / fs)
        signals_dict = result['signals'].copy()
        signals_dict['threshold'] = result['thresholds']['adaptive_threshold']
        
        # Generar gráficas Plotly
        fig_steps = visualize_pan_tompkins_steps(
            signals_dict, fs, result['r_peaks'], duration=viz_duration
        )
        
        waves_dict = waves_result.copy()
        waves_dict['r_peaks'] = result['r_peaks']
        fig_waves = visualize_detected_waves(
            ecg_filtered, waves_dict, fs, duration=viz_duration
        )
        
        fig_tacho = visualize_tachogram(tachogram_result)
        
        # Convertir gráficas a JSON para embed en HTML
        steps_json = fig_steps.to_json()
        waves_json = fig_waves.to_json()
        tacho_json = fig_tacho.to_json()
        
        return {
            'record_path': record_path,
            'label': label,
            'fs': float(fs),
            'duration': float(len(ecg_filtered) / fs),
            'n_peaks': len(result['r_peaks']),
            'threshold': float(result['thresholds']['adaptive_threshold']),
            'waves_detected': {
                'p_count': sum(1 for p in waves_result['p_waves'] if p is not None),
                'q_count': sum(1 for q in waves_result['q_waves'] if q is not None),
                's_count': sum(1 for s in waves_result['s_waves'] if s is not None),
                't_count': sum(1 for t in waves_result['t_waves'] if t is not None)
            },
            'hrv_metrics': {
                'heart_rate_bpm': float(tachogram_result['heart_rate_bpm']),
                'mean_rr_ms': float(tachogram_result['metadata']['mean_rr_ms']),
                'std_rr_ms': float(tachogram_result['metadata']['std_rr_ms']),
                'rmssd': float(hrv_metrics['rmssd']),
                'pnn50': float(hrv_metrics['pnn50'])
            },
            'wave_features': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in waves_result['wave_features'].items()},
            'plots': {
                'steps': steps_json,
                'waves': waves_json,
                'tachogram': tacho_json
            }
        }
        
    except Exception as e:
        warnings.warn(f"Error procesando {record_path}: {e}")
        return None


def generate_comparison_analysis(results: List[Dict]) -> Dict:
    """
    Generar análisis comparativo entre señales normales y SCD
    
    Args:
        results: Lista de resultados procesados
    
    Returns:
        Dict con análisis comparativo
    """
    normal_results = [r for r in results if r and r['label'] == 'Normal']
    scd_results = [r for r in results if r and r['label'] == 'SCD']
    
    analysis = {
        'summary': {
            'total_processed': len(results),
            'normal_count': len(normal_results),
            'scd_count': len(scd_results)
        },
        'detection_accuracy': {},
        'hrv_comparison': {},
        'wave_detection_rates': {}
    }
    
    # Análisis de precisión de detección
    if normal_results:
        normal_hr = np.mean([r['hrv_metrics']['heart_rate_bpm'] for r in normal_results])
        normal_peaks = np.mean([r['n_peaks'] for r in normal_results])
        analysis['detection_accuracy']['normal'] = {
            'avg_heart_rate': float(normal_hr),
            'avg_peaks_detected': float(normal_peaks)
        }
    
    if scd_results:
        scd_hr = np.mean([r['hrv_metrics']['heart_rate_bpm'] for r in scd_results])
        scd_peaks = np.mean([r['n_peaks'] for r in scd_results])
        analysis['detection_accuracy']['scd'] = {
            'avg_heart_rate': float(scd_hr),
            'avg_peaks_detected': float(scd_peaks)
        }
    
    # Comparación HRV
    if normal_results and scd_results:
        normal_sdnn = np.mean([r['hrv_metrics']['std_rr_ms'] for r in normal_results])
        scd_sdnn = np.mean([r['hrv_metrics']['std_rr_ms'] for r in scd_results])
        normal_rmssd = np.mean([r['hrv_metrics']['rmssd'] for r in normal_results])
        scd_rmssd = np.mean([r['hrv_metrics']['rmssd'] for r in scd_results])
        
        analysis['hrv_comparison'] = {
            'normal': {
                'sdnn': float(normal_sdnn),
                'rmssd': float(normal_rmssd)
            },
            'scd': {
                'sdnn': float(scd_sdnn),
                'rmssd': float(scd_rmssd)
            },
            'difference': {
                'sdnn_diff': float(scd_sdnn - normal_sdnn),
                'rmssd_diff': float(scd_rmssd - normal_rmssd)
            }
        }
    
    # Tasas de detección de ondas
    if normal_results:
        normal_p_rate = np.mean([r['waves_detected']['p_count'] / max(r['n_peaks'], 1) 
                                 for r in normal_results])
        normal_q_rate = np.mean([r['waves_detected']['q_count'] / max(r['n_peaks'], 1) 
                                for r in normal_results])
        analysis['wave_detection_rates']['normal'] = {
            'p_rate': float(normal_p_rate),
            'q_rate': float(normal_q_rate)
        }
    
    if scd_results:
        scd_p_rate = np.mean([r['waves_detected']['p_count'] / max(r['n_peaks'], 1) 
                              for r in scd_results])
        scd_q_rate = np.mean([r['waves_detected']['q_count'] / max(r['n_peaks'], 1) 
                              for r in scd_results])
        analysis['wave_detection_rates']['scd'] = {
            'p_rate': float(scd_p_rate),
            'q_rate': float(scd_q_rate)
        }
    
    return analysis


def generate_synthetic_signal(label: str, fs: float = 250, duration: float = 30.0) -> np.ndarray:
    """
    Generar señal ECG sintética
    
    Args:
        label: 'Normal' o 'SCD'
        fs: Frecuencia de muestreo
        duration: Duración en segundos
    
    Returns:
        Señal ECG sintética
    """
    t = np.linspace(0, duration, int(fs * duration))
    ecg_signal = np.zeros(len(t))
    
    if label == 'Normal':
        # Ritmo sinusal normal: ~70 bpm, variabilidad baja
        heart_rate = 70
        rr_variability = 0.02  # 2% de variación
    else:  # SCD
        # Ritmo con arritmias: ~75 bpm, variabilidad alta
        heart_rate = 75
        rr_variability = 0.15  # 15% de variación (mayor variabilidad)
    
    rr_interval_base = 60.0 / heart_rate
    current_time = 0.0
    
    while current_time < duration:
        # Agregar variabilidad al intervalo RR
        if label == 'SCD':
            # Variabilidad más alta para SCD
            rr_variation = np.random.normal(0, rr_variability)
        else:
            rr_variation = np.random.normal(0, rr_variability)
        
        rr_interval = rr_interval_base * (1 + rr_variation)
        beat_time = current_time
        beat_idx = int(beat_time * fs)
        
        if beat_idx < len(t):
            # Simular complejo QRS
            qrs_duration = 0.1
            qrs_samples = int(qrs_duration * fs)
            
            for j in range(qrs_samples):
                idx = beat_idx + j
                if idx < len(t):
                    # Forma de onda QRS
                    ecg_signal[idx] = np.sin(2 * np.pi * 10 * (t[idx] - beat_time)) * \
                                     np.exp(-5 * (t[idx] - beat_time))
            
            # Simular onda T (solo para algunas señales)
            if np.random.rand() > 0.3:
                t_time = beat_time + 0.2
                t_idx = int(t_time * fs)
                if 0 <= t_idx < len(t):
                    ecg_signal[t_idx] = 0.3 * np.sin(2 * np.pi * 2 * (t[t_idx] - t_time)) * \
                                       np.exp(-2 * (t[t_idx] - t_time))
        
        current_time += rr_interval
    
    # Agregar ruido
    noise_level = 0.08 if label == 'Normal' else 0.12
    ecg_signal += noise_level * np.random.randn(len(t))
    
    return ecg_signal


def generate_synthetic_results(n_signals: int = 3) -> List[Dict]:
    """
    Generar resultados con señales sintéticas
    
    Args:
        n_signals: Número de señales por tipo
    
    Returns:
        Lista de resultados procesados
    """
    results = []
    fs = 250
    duration = 30.0
    
    # Generar señales normales
    print("   Generando señales sintéticas Normal...")
    for i in range(n_signals):
        print(f"      Señal {i+1}/{n_signals}")
        ecg_signal = generate_synthetic_signal('Normal', fs, duration)
        
        # Preprocesar
        ecg_filtered = bandpass_filter(ecg_signal, fs)
        ecg_filtered = normalize_signal(ecg_filtered, method='zscore')
        
        # Aplicar Pan-Tompkins
        result = pan_tompkins_complete(ecg_filtered, fs, visualize=True, threshold_k=0.5)
        
        if len(result['r_peaks']) < 2:
            continue
        
        # Detectar ondas
        waves_result = detect_all_waves(ecg_filtered, result['r_peaks'], fs)
        
        # Calcular tacograma
        tachogram_result = calculate_tachogram(result['r_peaks'], fs)
        
        # Calcular métricas HRV
        hrv_metrics = calculate_hrv_time_domain(tachogram_result['rr_intervals'])
        
        # Preparar señales para visualización
        viz_duration = min(10.0, len(ecg_filtered) / fs)
        signals_dict = result['signals'].copy()
        signals_dict['threshold'] = result['thresholds']['adaptive_threshold']
        
        # Generar gráficas
        fig_steps = visualize_pan_tompkins_steps(
            signals_dict, fs, result['r_peaks'], duration=viz_duration
        )
        
        waves_dict = waves_result.copy()
        waves_dict['r_peaks'] = result['r_peaks']
        fig_waves = visualize_detected_waves(
            ecg_filtered, waves_dict, fs, duration=viz_duration
        )
        
        fig_tacho = visualize_tachogram(tachogram_result)
        
        # Convertir a JSON
        results.append({
            'record_path': f'synthetic_normal_{i+1}',
            'label': 'Normal',
            'fs': float(fs),
            'duration': float(len(ecg_filtered) / fs),
            'n_peaks': len(result['r_peaks']),
            'threshold': float(result['thresholds']['adaptive_threshold']),
            'waves_detected': {
                'p_count': sum(1 for p in waves_result['p_waves'] if p is not None),
                'q_count': sum(1 for q in waves_result['q_waves'] if q is not None),
                's_count': sum(1 for s in waves_result['s_waves'] if s is not None),
                't_count': sum(1 for t in waves_result['t_waves'] if t is not None)
            },
            'hrv_metrics': {
                'heart_rate_bpm': float(tachogram_result['heart_rate_bpm']),
                'mean_rr_ms': float(tachogram_result['metadata']['mean_rr_ms']),
                'std_rr_ms': float(tachogram_result['metadata']['std_rr_ms']),
                'rmssd': float(hrv_metrics['rmssd']),
                'pnn50': float(hrv_metrics['pnn50'])
            },
            'wave_features': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in waves_result['wave_features'].items()},
            'plots': {
                'steps': fig_steps.to_json(),
                'waves': fig_waves.to_json(),
                'tachogram': fig_tacho.to_json()
            }
        })
        print(f"         ✅ {results[-1]['n_peaks']} picos R, HR: {results[-1]['hrv_metrics']['heart_rate_bpm']:.1f} bpm")
    
    # Generar señales SCD
    print("\n   Generando señales sintéticas SCD...")
    for i in range(n_signals):
        print(f"      Señal {i+1}/{n_signals}")
        ecg_signal = generate_synthetic_signal('SCD', fs, duration)
        
        # Preprocesar
        ecg_filtered = bandpass_filter(ecg_signal, fs)
        ecg_filtered = normalize_signal(ecg_filtered, method='zscore')
        
        # Aplicar Pan-Tompkins
        result = pan_tompkins_complete(ecg_filtered, fs, visualize=True, threshold_k=0.5)
        
        if len(result['r_peaks']) < 2:
            continue
        
        # Detectar ondas
        waves_result = detect_all_waves(ecg_filtered, result['r_peaks'], fs)
        
        # Calcular tacograma
        tachogram_result = calculate_tachogram(result['r_peaks'], fs)
        
        # Calcular métricas HRV
        hrv_metrics = calculate_hrv_time_domain(tachogram_result['rr_intervals'])
        
        # Preparar señales para visualización
        viz_duration = min(10.0, len(ecg_filtered) / fs)
        signals_dict = result['signals'].copy()
        signals_dict['threshold'] = result['thresholds']['adaptive_threshold']
        
        # Generar gráficas
        fig_steps = visualize_pan_tompkins_steps(
            signals_dict, fs, result['r_peaks'], duration=viz_duration
        )
        
        waves_dict = waves_result.copy()
        waves_dict['r_peaks'] = result['r_peaks']
        fig_waves = visualize_detected_waves(
            ecg_filtered, waves_dict, fs, duration=viz_duration
        )
        
        fig_tacho = visualize_tachogram(tachogram_result)
        
        # Convertir a JSON
        results.append({
            'record_path': f'synthetic_scd_{i+1}',
            'label': 'SCD',
            'fs': float(fs),
            'duration': float(len(ecg_filtered) / fs),
            'n_peaks': len(result['r_peaks']),
            'threshold': float(result['thresholds']['adaptive_threshold']),
            'waves_detected': {
                'p_count': sum(1 for p in waves_result['p_waves'] if p is not None),
                'q_count': sum(1 for q in waves_result['q_waves'] if q is not None),
                's_count': sum(1 for s in waves_result['s_waves'] if s is not None),
                't_count': sum(1 for t in waves_result['t_waves'] if t is not None)
            },
            'hrv_metrics': {
                'heart_rate_bpm': float(tachogram_result['heart_rate_bpm']),
                'mean_rr_ms': float(tachogram_result['metadata']['mean_rr_ms']),
                'std_rr_ms': float(tachogram_result['metadata']['std_rr_ms']),
                'rmssd': float(hrv_metrics['rmssd']),
                'pnn50': float(hrv_metrics['pnn50'])
            },
            'wave_features': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in waves_result['wave_features'].items()},
            'plots': {
                'steps': fig_steps.to_json(),
                'waves': fig_waves.to_json(),
                'tachogram': fig_tacho.to_json()
            }
        })
        print(f"         ✅ {results[-1]['n_peaks']} picos R, HR: {results[-1]['hrv_metrics']['heart_rate_bpm']:.1f} bpm")
    
    return results


def generate_hrv_comparison_plot(results: List[Dict]) -> str:
    """
    Generar gráfica comparativa de métricas HRV entre Normal y SCD
    
    Usa gráficos de barras con barras de error para mejor visualización
    
    Args:
        results: Lista de resultados procesados
    
    Returns:
        JSON de la gráfica Plotly
    """
    normal_results = [r for r in results if r and r['label'] == 'Normal']
    scd_results = [r for r in results if r and r['label'] == 'SCD']
    
    if not normal_results or not scd_results:
        return None
    
    # Extraer métricas y calcular estadísticas
    normal_sdnn = [r['hrv_metrics']['std_rr_ms'] for r in normal_results]
    scd_sdnn = [r['hrv_metrics']['std_rr_ms'] for r in scd_results]
    normal_rmssd = [r['hrv_metrics']['rmssd'] for r in normal_results]
    scd_rmssd = [r['hrv_metrics']['rmssd'] for r in scd_results]
    normal_hr = [r['hrv_metrics']['heart_rate_bpm'] for r in normal_results]
    scd_hr = [r['hrv_metrics']['heart_rate_bpm'] for r in scd_results]
    normal_pnn50 = [r['hrv_metrics']['pnn50'] for r in normal_results]
    scd_pnn50 = [r['hrv_metrics']['pnn50'] for r in scd_results]
    
    # Calcular medias y desviaciones estándar
    
    metrics_data = {
        'SDNN (ms)': {
            'normal': {'mean': np.mean(normal_sdnn), 'std': np.std(normal_sdnn), 'values': normal_sdnn},
            'scd': {'mean': np.mean(scd_sdnn), 'std': np.std(scd_sdnn), 'values': scd_sdnn}
        },
        'RMSSD (ms)': {
            'normal': {'mean': np.mean(normal_rmssd), 'std': np.std(normal_rmssd), 'values': normal_rmssd},
            'scd': {'mean': np.mean(scd_rmssd), 'std': np.std(scd_rmssd), 'values': scd_rmssd}
        },
        'Frecuencia Cardíaca (bpm)': {
            'normal': {'mean': np.mean(normal_hr), 'std': np.std(normal_hr), 'values': normal_hr},
            'scd': {'mean': np.mean(scd_hr), 'std': np.std(scd_hr), 'values': scd_hr}
        },
        'pNN50 (%)': {
            'normal': {'mean': np.mean(normal_pnn50), 'std': np.std(normal_pnn50), 'values': normal_pnn50},
            'scd': {'mean': np.mean(scd_pnn50), 'std': np.std(scd_pnn50), 'values': scd_pnn50}
        }
    }
    
    # Crear subplots: 2 filas x 2 columnas
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metrics_data.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Colores
    normal_color = '#667eea'
    scd_color = '#f5576c'
    
    # Crear gráficos de barras con barras de error para cada métrica
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, (metric_name, data) in enumerate(metrics_data.items()):
        row, col = positions[idx]
        
        # Valores para Normal y SCD
        normal_mean = data['normal']['mean']
        normal_std = data['normal']['std']
        scd_mean = data['scd']['mean']
        scd_std = data['scd']['std']
        
        # Gráfico de barras con barras de error
        fig.add_trace(
            go.Bar(
                x=['Normal', 'SCD'],
                y=[normal_mean, scd_mean],
                error_y=dict(
                    type='data',
                    array=[normal_std, scd_std],
                    visible=True,
                    thickness=2,
                    width=3
                ),
                marker=dict(color=[normal_color, scd_color]),
                name=metric_name,
                text=[f'{normal_mean:.1f}±{normal_std:.1f}', f'{scd_mean:.1f}±{scd_std:.1f}'],
                textposition='outside',
                showlegend=(idx == 0)  # Solo mostrar leyenda en el primero
            ),
            row=row, col=col
        )
        
        # Agregar puntos individuales para mostrar la distribución
        fig.add_trace(
            go.Scatter(
                x=['Normal'] * len(data['normal']['values']) + ['SCD'] * len(data['scd']['values']),
                y=list(data['normal']['values']) + list(data['scd']['values']),
                mode='markers',
                marker=dict(
                    color=[normal_color] * len(data['normal']['values']) + [scd_color] * len(data['scd']['values']),
                    size=6,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                name='Valores Individuales',
                showlegend=False
            ),
            row=row, col=col
        )
    
    # Actualizar layout
    fig.update_layout(
        title=dict(
            text='Comparación de Métricas HRV: Normal vs SCD',
            x=0.5,
            font=dict(size=20)
        ),
        height=800,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Actualizar ejes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text='Grupo', row=row, col=col, tickmode='linear', tick0=0, dtick=1)
            # No establecer título Y aquí, se tomará del subplot_title
    
    # Establecer títulos Y específicos
    fig.update_yaxes(title_text='SDNN (ms)', row=1, col=1)
    fig.update_yaxes(title_text='RMSSD (ms)', row=1, col=2)
    fig.update_yaxes(title_text='HR (bpm)', row=2, col=1)
    fig.update_yaxes(title_text='pNN50 (%)', row=2, col=2)
    
    return fig.to_json()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar resultados de Pan-Tompkins para dashboard')
    parser.add_argument('--max-records', type=int, default=5,
                       help='Máximo número de registros por dataset a procesar')
    parser.add_argument('--output', type=str, default='results/pan_tompkins_results.json',
                       help='Archivo de salida JSON')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERACIÓN DE RESULTADOS PAN-TOMPKINS PARA DASHBOARD")
    print("=" * 70)
    print()
    
    # Buscar registros disponibles
    results = []
    
    # Procesar registros NSRDB (Normal)
    print("📂 Procesando registros NSRDB (Normal)...")
    nsrdb_records = list_available_records('datasets/nsrdb')
    if nsrdb_records:
        for i, record_name in enumerate(nsrdb_records[:args.max_records]):
            record_path = f'datasets/nsrdb/{record_name}'
            print(f"   Procesando {i+1}/{min(len(nsrdb_records), args.max_records)}: {record_name}")
            result = process_signal_for_dashboard(record_path, 'Normal')
            if result:
                results.append(result)
                print(f"      ✅ {result['n_peaks']} picos R detectados, HR: {result['hrv_metrics']['heart_rate_bpm']:.1f} bpm")
    
    # Procesar registros SDDB (SCD)
    print(f"\n📂 Procesando registros SDDB (SCD)...")
    sddb_records = list_available_records('datasets/sddb')
    if sddb_records:
        for i, record_name in enumerate(sddb_records[:args.max_records]):
            record_path = f'datasets/sddb/{record_name}'
            print(f"   Procesando {i+1}/{min(len(sddb_records), args.max_records)}: {record_name}")
            result = process_signal_for_dashboard(record_path, 'SCD')
            if result:
                results.append(result)
                print(f"      ✅ {result['n_peaks']} picos R detectados, HR: {result['hrv_metrics']['heart_rate_bpm']:.1f} bpm")
    
    # Si no hay registros reales, generar señales sintéticas
    if not results:
        print("\n⚠️  No se encontraron registros reales. Generando señales sintéticas para demostración...")
        results = generate_synthetic_results(args.max_records)
    
    if not results:
        print("\n⚠️  No se pudieron procesar registros. Verificar que los datasets estén disponibles.")
        return
    
    print(f"\n📊 Total procesado: {len(results)} señales")
    
    # Generar análisis comparativo
    print("\n📈 Generando análisis comparativo...")
    analysis = generate_comparison_analysis(results)
    
    # Generar gráfica comparativa HRV
    print("📊 Generando gráfica comparativa HRV...")
    hrv_comparison_plot = generate_hrv_comparison_plot(results)
    
    # Preparar datos finales
    output_data = {
        'results': results,
        'analysis': analysis,
        'hrv_comparison_plot': hrv_comparison_plot,
        'summary': {
            'total_signals': len(results),
            'normal_signals': len([r for r in results if r['label'] == 'Normal']),
            'scd_signals': len([r for r in results if r['label'] == 'SCD'])
        }
    }
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {output_path}")
    print(f"   - {len(results)} señales procesadas")
    print(f"   - {len([r for r in results if r['label'] == 'Normal'])} señales normales")
    print(f"   - {len([r for r in results if r['label'] == 'SCD'])} señales SCD")
    
    # Mostrar resumen de análisis
    if analysis.get('hrv_comparison'):
        hrv_comp = analysis['hrv_comparison']
        print(f"\n📊 Resumen de Análisis HRV:")
        print(f"   SDNN Normal: {hrv_comp['normal']['sdnn']:.2f} ms")
        print(f"   SDNN SCD: {hrv_comp['scd']['sdnn']:.2f} ms")
        print(f"   Diferencia: {hrv_comp['difference']['sdnn_diff']:.2f} ms")
        print(f"   RMSSD Normal: {hrv_comp['normal']['rmssd']:.2f} ms")
        print(f"   RMSSD SCD: {hrv_comp['scd']['rmssd']:.2f} ms")
        print(f"   Diferencia: {hrv_comp['difference']['rmssd_diff']:.2f} ms")


if __name__ == "__main__":
    main()

