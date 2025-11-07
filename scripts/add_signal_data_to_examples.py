"""
Script rápido para añadir datos de señal a los ejemplos existentes
para generar gráficas Plotly interactivas
"""

import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing_unified import preprocess_unified
from scripts.train_models import load_dataset, prepare_training_data
from sklearn.model_selection import train_test_split

def add_signal_data_to_examples(input_file: str, output_file: str, data_dir: str):
    """
    Añadir datos de señal a los ejemplos existentes en el archivo JSON
    
    Args:
        input_file: Archivo JSON con ejemplos existentes
        output_file: Archivo de salida (puede ser el mismo)
        data_dir: Directorio con datasets
    """
    print("🔄 Añadiendo datos de señal para gráficas Plotly interactivas...\n")
    
    # Cargar archivo existente
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    examples = data.get('visualization_examples', [])
    if not examples:
        print("❌ No hay ejemplos en el archivo")
        return
    
    print(f"📊 Encontrados {len(examples)} ejemplos")
    
    # Verificar si ya tienen datos
    if 'signal_data' in examples[0]:
        print("✅ Los ejemplos ya tienen datos de señal")
        return
    
    # Cargar datos completos para obtener las señales
    print("\n📂 Cargando datos completos...")
    data_dir = Path(data_dir)
    sddb_path = data_dir / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = data_dir / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    sddb_signals, sddb_labels, _ = load_dataset(str(sddb_path), 'sddb', max_records=None)
    nsrdb_signals, nsrdb_labels, _ = load_dataset(str(nsrdb_path), 'nsrdb', max_records=None)
    
    all_signals = sddb_signals + nsrdb_signals
    all_labels = np.concatenate([sddb_labels, nsrdb_labels])
    
    # Preparar datos igual que en el script original
    fs = 128.0
    window_size = 30.0
    X_segments, y_segments = prepare_training_data(all_signals, all_labels, fs=fs, window_size=window_size)
    
    # Dividir igual que en el script original
    X_train, X_test, y_train, y_test = train_test_split(
        X_segments, y_segments, test_size=0.2, random_state=42, stratify=y_segments
    )
    
    print(f"✅ Datos cargados: {len(X_test)} segmentos en conjunto de prueba\n")
    
    # Añadir datos de señal a cada ejemplo
    print("📊 Procesando ejemplos...")
    for i, example in enumerate(examples):
        # Extraer índice del signal_id (formato: "test_12345")
        signal_id = example.get('signal_id', '')
        if signal_id.startswith('test_'):
            try:
                idx = int(signal_id.split('_')[1])
                
                if idx < len(X_test):
                    signal = X_test[idx]
                    
                    # Procesar señal
                    processed = preprocess_unified(signal, fs=fs, target_fs=128.0)
                    processed_signal = processed['signal']
                    
                    # Extraer primeros 10 segundos para Plotly
                    duration_samples = int(10.0 * fs)  # 1280 muestras
                    signal_data_for_plot = processed_signal[:duration_samples].tolist()
                    time_axis = (np.arange(len(signal_data_for_plot)) / fs).tolist()
                    
                    # Añadir datos al ejemplo
                    example['signal_data'] = signal_data_for_plot
                    example['time_axis'] = time_axis
                    example['fs'] = float(fs)
                    
                    print(f"   ✅ Ejemplo {i+1}/{len(examples)}: {len(signal_data_for_plot)} puntos")
                else:
                    print(f"   ⚠️  Ejemplo {i+1}: Índice {idx} fuera de rango")
            except (ValueError, IndexError) as e:
                print(f"   ⚠️  Ejemplo {i+1}: Error extrayendo índice - {e}")
        else:
            print(f"   ⚠️  Ejemplo {i+1}: signal_id no válido - {signal_id}")
    
    # Guardar archivo actualizado
    print(f"\n💾 Guardando archivo actualizado...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Archivo guardado: {output_file}")
    print(f"   Ejemplos actualizados: {len(examples)}")
    
    # Verificar
    examples_with_data = sum(1 for ex in examples if 'signal_data' in ex)
    print(f"   Ejemplos con datos Plotly: {examples_with_data}/{len(examples)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Añadir datos de señal a ejemplos existentes')
    parser.add_argument('--input', type=str, default='results/realtime_predictions.json',
                       help='Archivo JSON de entrada')
    parser.add_argument('--output', type=str, default='results/realtime_predictions.json',
                       help='Archivo JSON de salida')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    
    args = parser.parse_args()
    
    add_signal_data_to_examples(args.input, args.output, args.data_dir)

if __name__ == "__main__":
    main()

