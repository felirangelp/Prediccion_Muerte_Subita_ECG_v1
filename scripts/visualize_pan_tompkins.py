"""
Visualización de resultados del algoritmo Pan-Tompkins
Incluye visualización paso a paso, ondas detectadas y tacograma
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, Optional, List
from pathlib import Path


def visualize_pan_tompkins_steps(signals_dict: Dict, fs: float, 
                                 r_peaks: Optional[np.ndarray] = None,
                                 output_file: Optional[str] = None,
                                 duration: Optional[float] = None) -> go.Figure:
    """
    Visualizar todos los pasos del algoritmo Pan-Tompkins
    
    Crea una figura con 6 subplots mostrando:
    1. Señal ECG original
    2. Señal diferenciada
    3. Señal al cuadrado
    4. Señal integrada
    5. Señal umbralizada
    6. Picos R detectados sobre señal original
    
    Args:
        signals_dict: Diccionario con señales de cada paso
            - 'original': Señal ECG original
            - 'differentiated': Señal diferenciada
            - 'squared': Señal al cuadrado
            - 'integrated': Señal integrada
            - 'thresholded': Señal umbralizada
        fs: Frecuencia de muestreo
        r_peaks: Picos R detectados (opcional)
        output_file: Archivo de salida HTML (opcional)
        duration: Duración a visualizar en segundos (opcional, muestra todo si None)
    
    Returns:
        Figura de Plotly
    """
    # Crear subplots: 2 filas x 3 columnas para ocupar más ancho
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '1. Señal ECG Original',
            '2. Señal Diferenciada',
            '3. Señal al Cuadrado',
            '4. Señal Integrada',
            '5. Señal Umbralizada',
            '6. Picos R Detectados'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.05,  # Reducir espaciado horizontal
        column_widths=[1, 1, 1],  # Columnas iguales
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Determinar rango de muestras a mostrar
    original_signal = signals_dict.get('original', np.array([]))
    if len(original_signal) == 0:
        raise ValueError("Señal original vacía en signals_dict")
    
    if duration is not None:
        max_samples = int(duration * fs)
        max_samples = min(max_samples, len(original_signal))
    else:
        max_samples = len(original_signal)
    
    time_axis = np.arange(max_samples) / fs
    
    # 1. Señal ECG Original
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=original_signal[:max_samples],
            mode='lines',
            name='ECG Original',
            line=dict(color='#667eea', width=1.5)
        ),
        row=1, col=1
    )
    
    # 2. Señal Diferenciada
    if 'differentiated' in signals_dict:
        diff_signal = signals_dict['differentiated'][:max_samples]
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=diff_signal,
                mode='lines',
                name='Diferenciada',
                line=dict(color='#11998e', width=1.5)
            ),
            row=1, col=2
        )
    
    # 3. Señal al Cuadrado
    if 'squared' in signals_dict:
        squared_signal = signals_dict['squared'][:max_samples]
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=squared_signal,
                mode='lines',
                name='Al Cuadrado',
                line=dict(color='#f093fb', width=1.5)
            ),
            row=1, col=3
        )
    
    # 4. Señal Integrada
    if 'integrated' in signals_dict:
        integrated_signal = signals_dict['integrated'][:max_samples]
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=integrated_signal,
                mode='lines',
                name='Integrada',
                line=dict(color='#f5576c', width=1.5)
            ),
            row=2, col=1
        )
        
        # Agregar línea de umbral si está disponible
        if 'threshold' in signals_dict:
            threshold_val = signals_dict['threshold']
            fig.add_hline(
                y=threshold_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Umbral: {threshold_val:.3f}",
                row=2, col=1
            )
    
    # 5. Señal Umbralizada
    if 'thresholded' in signals_dict:
        thresholded_signal = signals_dict['thresholded'][:max_samples]
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=thresholded_signal,
                mode='lines',
                name='Umbralizada',
                line=dict(color='#4facfe', width=1.5)
            ),
            row=2, col=2
        )
    
    # 6. Picos R Detectados sobre Señal Original
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=original_signal[:max_samples],
            mode='lines',
            name='ECG Original',
            line=dict(color='#667eea', width=1.5),
            showlegend=False
        ),
        row=2, col=3
    )
    
    if r_peaks is not None and len(r_peaks) > 0:
        # Filtrar picos R dentro del rango mostrado
        r_peaks_filtered = r_peaks[r_peaks < max_samples]
        if len(r_peaks_filtered) > 0:
            r_peaks_times = r_peaks_filtered / fs
            r_peaks_values = original_signal[r_peaks_filtered]
            
            fig.add_trace(
                go.Scatter(
                    x=r_peaks_times,
                    y=r_peaks_values,
                    mode='markers',
                    name='Picos R',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='circle',
                        line=dict(width=2, color='darkred')
                    )
                ),
                row=2, col=3
            )
    
    # Actualizar layout para ocupar todo el ancho disponible
    fig.update_layout(
        title=dict(
            text='Algoritmo Pan-Tompkins - Pasos del Procesamiento',
            x=0.5,
            font=dict(size=20)
        ),
        height=900,  # Altura ajustada para 2 filas
        width=None,  # Se ajustará al contenedor
        showlegend=True,
        template='plotly_white',
        autosize=True,
        margin=dict(l=30, r=30, t=80, b=30),  # Márgenes mínimos
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Actualizar ejes para 2 filas x 3 columnas
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(title_text='Tiempo (s)', row=i, col=j)
            fig.update_yaxes(title_text='Amplitud', row=i, col=j)
    
    # Guardar si se especifica archivo
    if output_file:
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Gráfica guardada en: {output_file}")
    
    return fig


def visualize_detected_waves(ecg_signal: np.ndarray, waves_dict: Dict, fs: float,
                            duration: Optional[float] = None,
                            output_file: Optional[str] = None) -> go.Figure:
    """
    Visualizar ondas detectadas (P, Q, R, S, T) sobre señal ECG
    
    Args:
        ecg_signal: Señal ECG original 1D
        waves_dict: Diccionario con ondas detectadas
            - 'p_waves': Lista de índices de ondas P
            - 'q_waves': Lista de índices de ondas Q
            - 'r_peaks': Array de índices de picos R (o 'r_peaks' en waves_dict)
            - 's_waves': Lista de índices de ondas S
            - 't_waves': Lista de índices de ondas T
        fs: Frecuencia de muestreo
        duration: Duración a visualizar en segundos (opcional)
        output_file: Archivo de salida HTML (opcional)
    
    Returns:
        Figura de Plotly
    """
    # Determinar rango a mostrar
    if duration is not None:
        max_samples = int(duration * fs)
        max_samples = min(max_samples, len(ecg_signal))
    else:
        max_samples = len(ecg_signal)
    
    time_axis = np.arange(max_samples) / fs
    signal_to_plot = ecg_signal[:max_samples]
    
    # Crear figura
    fig = go.Figure()
    
    # Señal ECG base
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal_to_plot,
            mode='lines',
            name='Señal ECG',
            line=dict(color='#667eea', width=2)
        )
    )
    
    # Colores para cada onda
    wave_colors = {
        'P': '#4facfe',      # Azul
        'Q': '#11998e',      # Verde
        'R': '#f5576c',      # Rojo
        'S': '#f093fb',      # Púrpura
        'T': '#764ba2'       # Púrpura oscuro
    }
    
    # Detectar y marcar ondas
    r_peaks = waves_dict.get('r_peaks', np.array([]))
    if len(r_peaks) == 0:
        # Intentar obtener de otra clave
        r_peaks = waves_dict.get('r_waves', np.array([]))
    
    # Filtrar picos R dentro del rango
    if len(r_peaks) > 0:
        r_peaks_filtered = r_peaks[r_peaks < max_samples]
        if len(r_peaks_filtered) > 0:
            r_times = r_peaks_filtered / fs
            r_values = ecg_signal[r_peaks_filtered]
            
            fig.add_trace(
                go.Scatter(
                    x=r_times,
                    y=r_values,
                    mode='markers',
                    name='Onda R',
                    marker=dict(
                        color=wave_colors['R'],
                        size=12,
                        symbol='circle',
                        line=dict(width=2, color='darkred')
                    )
                )
            )
    
    # Ondas P
    p_waves = waves_dict.get('p_waves', [])
    if p_waves:
        p_valid = [p for p in p_waves if p is not None and p < max_samples]
        if p_valid:
            p_times = np.array(p_valid) / fs
            p_values = ecg_signal[np.array(p_valid)]
            
            fig.add_trace(
                go.Scatter(
                    x=p_times,
                    y=p_values,
                    mode='markers',
                    name='Onda P',
                    marker=dict(
                        color=wave_colors['P'],
                        size=8,
                        symbol='triangle-up'
                    )
                )
            )
    
    # Ondas Q
    q_waves = waves_dict.get('q_waves', [])
    if q_waves:
        q_valid = [q for q in q_waves if q is not None and q < max_samples]
        if q_valid:
            q_times = np.array(q_valid) / fs
            q_values = ecg_signal[np.array(q_valid)]
            
            fig.add_trace(
                go.Scatter(
                    x=q_times,
                    y=q_values,
                    mode='markers',
                    name='Onda Q',
                    marker=dict(
                        color=wave_colors['Q'],
                        size=8,
                        symbol='triangle-down'
                    )
                )
            )
    
    # Ondas S
    s_waves = waves_dict.get('s_waves', [])
    if s_waves:
        s_valid = [s for s in s_waves if s is not None and s < max_samples]
        if s_valid:
            s_times = np.array(s_valid) / fs
            s_values = ecg_signal[np.array(s_valid)]
            
            fig.add_trace(
                go.Scatter(
                    x=s_times,
                    y=s_values,
                    mode='markers',
                    name='Onda S',
                    marker=dict(
                        color=wave_colors['S'],
                        size=8,
                        symbol='triangle-down'
                    )
                )
            )
    
    # Ondas T
    t_waves = waves_dict.get('t_waves', [])
    if t_waves:
        t_valid = [t for t in t_waves if t is not None and t < max_samples]
        if t_valid:
            t_times = np.array(t_valid) / fs
            t_values = ecg_signal[np.array(t_valid)]
            
            fig.add_trace(
                go.Scatter(
                    x=t_times,
                    y=t_values,
                    mode='markers',
                    name='Onda T',
                    marker=dict(
                        color=wave_colors['T'],
                        size=8,
                        symbol='diamond'
                    )
                )
            )
    
    # Actualizar layout
    fig.update_layout(
        title=dict(
            text='Ondas ECG Detectadas (P, Q, R, S, T)',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title='Tiempo (segundos)',
        yaxis_title='Amplitud (mV)',
        height=600,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Guardar si se especifica archivo
    if output_file:
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Gráfica guardada en: {output_file}")
    
    return fig


def visualize_tachogram(tachogram_data: Dict, output_file: Optional[str] = None) -> go.Figure:
    """
    Visualizar tacograma (intervalos RR vs tiempo)
    
    Args:
        tachogram_data: Datos del tacograma (resultado de calculate_tachogram)
            - 'rr_intervals': Array de intervalos RR en ms
            - 'time_points': Array de tiempos en s
            - 'tachogram_data': DataFrame (opcional)
        output_file: Archivo de salida HTML (opcional)
    
    Returns:
        Figura de Plotly con subplots (tacograma e histograma)
    """
    # Crear subplots: 1 fila x 2 columnas
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Tacograma (RR vs Tiempo)', 'Histograma de Intervalos RR'),
        column_widths=[0.7, 0.3]
    )
    
    # Obtener datos
    if 'tachogram_data' in tachogram_data and hasattr(tachogram_data['tachogram_data'], 'values'):
        # Usar DataFrame
        df = tachogram_data['tachogram_data']
        time_points = df['time'].values
        rr_intervals = df['rr_interval'].values
    else:
        # Usar arrays directamente
        time_points = tachogram_data.get('time_points', np.array([]))
        rr_intervals = tachogram_data.get('rr_intervals', np.array([]))
    
    if len(rr_intervals) == 0:
        raise ValueError("Datos de tacograma vacíos")
    
    # 1. Tacograma (scatter plot)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=rr_intervals,
            mode='lines+markers',
            name='Intervalos RR',
            line=dict(color='#667eea', width=2),
            marker=dict(color='#667eea', size=4)
        ),
        row=1, col=1
    )
    
    # Agregar línea de media
    mean_rr = np.mean(rr_intervals)
    fig.add_hline(
        y=mean_rr,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Media: {mean_rr:.1f} ms",
        row=1, col=1
    )
    
    # 2. Histograma de intervalos RR
    fig.add_trace(
        go.Histogram(
            x=rr_intervals,
            nbinsx=30,
            name='Distribución RR',
            marker_color='#11998e',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        title=dict(
            text='Análisis de Tacograma y Variabilidad de Frecuencia Cardíaca',
            x=0.5,
            font=dict(size=18)
        ),
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text='Tiempo (s)', row=1, col=1)
    fig.update_yaxes(title_text='Intervalo RR (ms)', row=1, col=1)
    fig.update_xaxes(title_text='Intervalo RR (ms)', row=1, col=2)
    fig.update_yaxes(title_text='Frecuencia', row=1, col=2)
    
    # Agregar información de HRV si está disponible
    if 'metadata' in tachogram_data:
        metadata = tachogram_data['metadata']
        hr_bpm = metadata.get('heart_rate_bpm', 0)
        mean_rr = metadata.get('mean_rr_ms', 0)
        std_rr = metadata.get('std_rr_ms', 0)
        
        # Agregar anotación con métricas
        annotations_text = (
            f"<b>Métricas HRV:</b><br>"
            f"Frecuencia Cardíaca: {hr_bpm:.1f} bpm<br>"
            f"RR Promedio: {mean_rr:.1f} ms<br>"
            f"SDNN: {std_rr:.1f} ms"
        )
        
        fig.add_annotation(
            text=annotations_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
    
    # Guardar si se especifica archivo
    if output_file:
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"✅ Gráfica guardada en: {output_file}")
    
    return fig


# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Visualización Pan-Tompkins")
    print("=" * 50)
    print("Este módulo proporciona funciones de visualización.")
    print("Ver scripts/demo_pan_tompkins.py para ejemplos de uso completo.")

