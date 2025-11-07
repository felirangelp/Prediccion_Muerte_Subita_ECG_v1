"""
Script para extraer segmentos de ECG en intervalos temporales específicos antes del evento SCD
Basado en metodologías de Sensors 2021 y Symmetry 2025

Soporta dos esquemas:
- Sensors: intervalos de 5, 10, 15, 20, 25, 30 minutos antes de SCD
- Symmetry: intervalos de 0-10, 10-20, 20-30, 30-40, 40-50, 50-60 minutos antes de SCD
"""

import numpy as np
import wfdb
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing_unified import preprocess_unified, downsample_signal


def extract_scd_event_time(record_path: str) -> Optional[float]:
    """
    Extraer el tiempo exacto del evento SCD desde archivo de anotaciones
    
    Args:
        record_path: Ruta al registro (sin extensión)
    
    Returns:
        Tiempo del evento SCD en segundos desde el inicio del registro, o None si no se encuentra
    """
    try:
        # Leer anotaciones
        ann = wfdb.rdann(record_path, 'atr')
        
        # Buscar anotación de evento SCD/VF
        # En SDDB, los eventos críticos están marcados con códigos específicos
        # Código 22 = VF (Ventricular Fibrillation)
        # Código 23 = VT (Ventricular Tachycardia)
        # También buscar 'VF' o 'SCD' en los símbolos
        
        scd_time = None
        
        # Buscar en símbolos
        if hasattr(ann, 'symbol') and ann.symbol is not None:
            for i, symbol in enumerate(ann.symbol):
                if symbol in ['VF', 'SCD', 'V']:  # V puede ser VF
                    scd_time = ann.sample[i] / ann.fs  # Convertir muestras a segundos
                    break
        
        # Si no se encuentra en símbolos, buscar en aux_note o comentarios
        if scd_time is None and hasattr(ann, 'aux_note'):
            for i, note in enumerate(ann.aux_note):
                if note and ('VF' in note.upper() or 'SCD' in note.upper() or 'VENTRICULAR FIBRILLATION' in note.upper()):
                    scd_time = ann.sample[i] / ann.fs
                    break
        
        # Si aún no se encuentra, usar el último evento anotado como aproximación
        # (en SDDB, el evento SCD generalmente está cerca del final)
        if scd_time is None and len(ann.sample) > 0:
            # Leer información del registro para obtener duración total
            record = wfdb.rdrecord(record_path)
            total_duration = len(record.p_signal) / record.fs
            
            # En SDDB, el evento generalmente ocurre cerca del final
            # Usar el último 5% del registro como aproximación
            scd_time = total_duration * 0.95
        
        return scd_time
        
    except Exception as e:
        print(f"⚠️  Error extrayendo tiempo SCD de {record_path}: {e}")
        return None


def extract_pre_scd_intervals(ecg_signal: np.ndarray, 
                              fs: float,
                              scd_time_seconds: float,
                              intervals: List[int] = [5, 10, 15, 20, 25, 30],
                              interval_duration: float = 60.0) -> Dict[int, np.ndarray]:
    """
    Extraer segmentos de ECG en intervalos específicos antes del SCD
    
    Args:
        ecg_signal: Señal ECG completa (1D array)
        fs: Frecuencia de muestreo
        scd_time_seconds: Tiempo del evento SCD en segundos desde el inicio
        intervals: Lista de minutos antes de SCD a extraer [5, 10, 15, ...]
        interval_duration: Duración de cada intervalo en segundos (default: 60s = 1 minuto)
    
    Returns:
        Dict con {minutos_antes_scd: segmento_ecg}
    """
    segments = {}
    
    for minutes_before in intervals:
        # Calcular tiempo de inicio del intervalo
        interval_start_time = scd_time_seconds - (minutes_before * 60 + interval_duration)
        interval_end_time = scd_time_seconds - (minutes_before * 60)
        
        # Verificar que el intervalo esté dentro de la señal
        if interval_start_time < 0:
            continue  # Intervalo fuera de rango
        
        # Convertir a índices de muestras
        start_idx = int(interval_start_time * fs)
        end_idx = int(interval_end_time * fs)
        
        if end_idx > len(ecg_signal):
            continue  # Intervalo fuera de rango
        
        # Extraer segmento
        segment = ecg_signal[start_idx:end_idx]
        
        if len(segment) > 0:
            segments[minutes_before] = segment
    
    return segments


def extract_symmetry_intervals(ecg_signal: np.ndarray,
                               fs: float,
                               scd_time_seconds: float,
                               intervals: List[Tuple[int, int]] = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]) -> Dict[Tuple[int, int], List[np.ndarray]]:
    """
    Extraer segmentos según esquema Symmetry (intervalos de 10 minutos)
    
    Args:
        ecg_signal: Señal ECG completa
        fs: Frecuencia de muestreo
        scd_time_seconds: Tiempo del evento SCD en segundos
        intervals: Lista de tuplas (min_inicio, min_fin) antes de SCD
    
    Returns:
        Dict con {(min_inicio, min_fin): [segmentos_de_1_minuto]}
    """
    segments_by_interval = {}
    
    for min_start, min_end in intervals:
        # Cada intervalo de 10 minutos se divide en 10 segmentos de 1 minuto
        interval_segments = []
        
        for minute_offset in range(min_start, min_end):
            # Calcular tiempo de inicio y fin del segmento de 1 minuto
            segment_start_time = scd_time_seconds - ((minute_offset + 1) * 60)
            segment_end_time = scd_time_seconds - (minute_offset * 60)
            
            # Verificar que esté dentro de la señal
            if segment_start_time < 0:
                continue
            
            start_idx = int(segment_start_time * fs)
            end_idx = int(segment_end_time * fs)
            
            if end_idx > len(ecg_signal):
                continue
            
            segment = ecg_signal[start_idx:end_idx]
            if len(segment) > 0:
                interval_segments.append(segment)
        
        if interval_segments:
            segments_by_interval[(min_start, min_end)] = interval_segments
    
    return segments_by_interval


def process_dataset_temporal_intervals(dataset_path: str,
                                      dataset_type: str,
                                      scheme: str = 'sensors',
                                      target_fs: float = 128.0,
                                      max_records: Optional[int] = None) -> Dict:
    """
    Procesar dataset completo extrayendo intervalos temporales
    
    Args:
        dataset_path: Ruta al directorio del dataset
        dataset_type: 'sddb' o 'nsrdb'
        scheme: 'sensors' o 'symmetry'
        target_fs: Frecuencia de muestreo objetivo
        max_records: Número máximo de registros a procesar (None = todos)
    
    Returns:
        Dict con estructura de datos temporal
    """
    dataset_path = Path(dataset_path)
    
    # Listar registros disponibles
    records = []
    for file in dataset_path.glob('*.hea'):
        record_name = file.stem
        records.append(record_name)
    
    records = sorted(set(records))
    
    if max_records:
        records = records[:max_records]
    
    print(f"📂 Procesando {len(records)} registros de {dataset_type}...")
    
    all_segments = []
    all_labels = []
    all_metadata = []
    
    for record_name in tqdm(records, desc=f"Procesando {dataset_type}"):
        record_path = str(dataset_path / record_name)
        
        try:
            # Cargar señal
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
            fs = record.fs
            
            # Preprocesar señal con manejo de errores
            try:
                preprocessed = preprocess_unified(
                    signal, fs, 
                    target_fs=target_fs,
                    apply_filter=True,
                    apply_normalize=True
                )
                signal_processed = preprocessed['signal']
                fs_processed = preprocessed['metadata']['effective_fs']
            except Exception as e:
                print(f"⚠️  Error preprocesando {record_name}: {e}")
                # Intentar sin filtro si falla
                try:
                    preprocessed = preprocess_unified(
                        signal, fs, 
                        target_fs=target_fs,
                        apply_filter=False,  # Desactivar filtro si causa problemas
                        apply_normalize=True
                    )
                    signal_processed = preprocessed['signal']
                    fs_processed = preprocessed['metadata']['effective_fs']
                except Exception as e2:
                    print(f"⚠️  Error incluso sin filtro para {record_name}: {e2}, omitiendo...")
                    continue
            
            # Verificar que la señal procesada es válida
            if len(signal_processed) == 0 or np.all(np.isnan(signal_processed)) or np.all(np.isinf(signal_processed)):
                print(f"⚠️  Señal procesada inválida para {record_name}, omitiendo...")
                continue
            
            if dataset_type == 'sddb':
                # Extraer tiempo del evento SCD
                scd_time = extract_scd_event_time(record_path)
                
                if scd_time is None:
                    print(f"⚠️  No se pudo extraer tiempo SCD de {record_name}, omitiendo...")
                    continue
                
                if scheme == 'sensors':
                    # Esquema Sensors: intervalos de 5, 10, 15, 20, 25, 30 minutos
                    intervals = [5, 10, 15, 20, 25, 30]
                    segments = extract_pre_scd_intervals(
                        signal_processed, fs_processed, scd_time, intervals
                    )
                    
                    for minutes_before, segment in segments.items():
                        all_segments.append(segment)
                        all_labels.append(minutes_before)  # Etiqueta = minutos antes de SCD
                        all_metadata.append({
                            'record': record_name,
                            'interval': minutes_before,
                            'scd_time': scd_time,
                            'scheme': 'sensors'
                        })
                
                elif scheme == 'symmetry':
                    # Esquema Symmetry: intervalos de 10 minutos (0-10, 10-20, ...)
                    intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
                    segments_dict = extract_symmetry_intervals(
                        signal_processed, fs_processed, scd_time, intervals
                    )
                    
                    for interval_tuple, segment_list in segments_dict.items():
                        for segment in segment_list:
                            all_segments.append(segment)
                            # Etiqueta: promedio de minutos antes de SCD para el intervalo
                            avg_minutes = (interval_tuple[0] + interval_tuple[1]) / 2
                            all_labels.append(avg_minutes)
                            all_metadata.append({
                                'record': record_name,
                                'interval': interval_tuple,
                                'scd_time': scd_time,
                                'scheme': 'symmetry'
                            })
            
            elif dataset_type == 'nsrdb':
                # Para NSR, extraer segmentos aleatorios de 1 minuto
                # (según paper Sensors, se extrae aleatoriamente)
                segment_duration_samples = int(60.0 * fs_processed)
                
                if len(signal_processed) < segment_duration_samples:
                    continue
                
                # Extraer múltiples segmentos aleatorios
                n_segments = min(10, len(signal_processed) // segment_duration_samples)
                
                for _ in range(n_segments):
                    start_idx = np.random.randint(0, len(signal_processed) - segment_duration_samples)
                    segment = signal_processed[start_idx:start_idx + segment_duration_samples]
                    
                    all_segments.append(segment)
                    all_labels.append('Normal')  # Etiqueta para clase normal
                    all_metadata.append({
                        'record': record_name,
                        'interval': 'Normal',
                        'scheme': scheme
                    })
        
        except Exception as e:
            print(f"⚠️  Error procesando {record_name}: {e}")
            continue
    
    print(f"✅ Extraídos {len(all_segments)} segmentos de {dataset_type}")
    
    return {
        'segments': all_segments,
        'labels': all_labels,
        'metadata': all_metadata,
        'scheme': scheme,
        'dataset_type': dataset_type
    }


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraer intervalos temporales pre-SCD')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--output', type=str, default='results/temporal_intervals_data.pkl',
                       help='Archivo de salida')
    parser.add_argument('--scheme', type=str, choices=['sensors', 'symmetry'], default='sensors',
                       help='Esquema de intervalos: sensors (5,10,15,20,25,30 min) o symmetry (0-10,10-20,...)')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("⏱️  EXTRACCIÓN DE INTERVALOS TEMPORALES PRE-SCD")
    print("=" * 70)
    print(f"Esquema: {args.scheme}")
    print()
    
    # Rutas a datasets
    sddb_path = Path(args.data_dir) / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = Path(args.data_dir) / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    # Procesar SDDB
    print("\n📊 Procesando SDDB (pacientes con SCD)...")
    sddb_data = process_dataset_temporal_intervals(
        str(sddb_path), 'sddb', args.scheme, max_records=args.max_records
    )
    
    # Procesar NSRDB
    print("\n📊 Procesando NSRDB (pacientes sanos)...")
    nsrdb_data = process_dataset_temporal_intervals(
        str(nsrdb_path), 'nsrdb', args.scheme, max_records=args.max_records
    )
    
    # Combinar datos
    all_data = {
        'sddb': sddb_data,
        'nsrdb': nsrdb_data,
        'scheme': args.scheme,
        'total_segments': len(sddb_data['segments']) + len(nsrdb_data['segments'])
    }
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    print("\n" + "=" * 70)
    print("✅ EXTRACCIÓN COMPLETADA")
    print("=" * 70)
    print(f"📁 Archivo guardado: {args.output}")
    print(f"📊 Total de segmentos extraídos: {all_data['total_segments']}")
    print(f"   - SDDB: {len(sddb_data['segments'])} segmentos")
    print(f"   - NSRDB: {len(nsrdb_data['segments'])} segmentos")


if __name__ == "__main__":
    main()

