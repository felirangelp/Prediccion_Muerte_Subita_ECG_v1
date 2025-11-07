"""
Script para generar predicciones en tiempo real usando señales reales
y crear visualizaciones para el dashboard
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import json
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import base64
from io import BytesIO

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.utils import load_ecg_record, list_available_records
from src.preprocessing_unified import preprocess_unified
from scripts.train_models import load_dataset, prepare_training_data
from sklearn.model_selection import train_test_split

def signal_to_base64(signal: np.ndarray, fs: float, duration: float = 10.0) -> str:
    """Convertir señal a imagen base64 para el dashboard"""
    samples_to_show = int(duration * fs)
    samples_to_show = min(samples_to_show, len(signal))
    
    time_axis = np.arange(samples_to_show) / fs
    signal_to_plot = signal[:samples_to_show]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, signal_to_plot, linewidth=1.5, color='#667eea')
    ax.set_xlabel('Tiempo (segundos)', fontsize=12)
    ax.set_ylabel('Amplitud (mV)', fontsize=12)
    ax.set_title('Señal ECG Procesada', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def process_real_signals(models_dir: str, data_dir: str, output_file: str, use_test_set: bool = True):
    """
    Procesar señales reales del conjunto de prueba y generar predicciones para el dashboard
    
    Args:
        models_dir: Directorio con modelos entrenados
        data_dir: Directorio con datasets
        output_file: Archivo de salida JSON
        use_test_set: Si True, usa el conjunto de prueba completo; si False, usa ejemplos
    """
    print("🔄 Generando predicciones en tiempo real con señales reales del conjunto de prueba...")
    
    models_dir = Path(models_dir)
    data_dir = Path(data_dir)
    
    # Cargar modelos
    models = {}
    
    if (models_dir / 'sparse_classifier.pkl').exists():
        try:
            models['sparse'] = SparseRepresentationClassifier.load(
                str(models_dir / 'sparse_classifier.pkl')
            )
            print("✅ Modelo sparse cargado")
        except Exception as e:
            print(f"⚠️  Error cargando sparse: {e}")
    
    if (models_dir / 'hierarchical_classifier_fusion.h5').exists():
        try:
            models['hierarchical'] = HierarchicalFusionClassifier.load(
                str(models_dir / 'hierarchical_classifier')
            )
            print("✅ Modelo hierarchical cargado")
        except Exception as e:
            print(f"⚠️  Error cargando hierarchical: {e}")
    
    if (models_dir / 'hybrid_model_sparse.pkl').exists():
        try:
            models['hybrid'] = HybridSCDClassifier.load(
                str(models_dir / 'hybrid_model')
            )
            print("✅ Modelo hybrid cargado")
        except Exception as e:
            print(f"⚠️  Error cargando hybrid: {e}")
    
    if not models:
        print("❌ No se encontraron modelos entrenados")
        return None
    
    # Cargar datos completos
    sddb_path = data_dir / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = data_dir / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    print(f"\n📂 Cargando datos completos...")
    
    # Cargar todos los datos disponibles
    sddb_signals, sddb_labels, sddb_metadata = load_dataset(str(sddb_path), 'sddb', max_records=None)
    nsrdb_signals, nsrdb_labels, nsrdb_metadata = load_dataset(str(nsrdb_path), 'nsrdb', max_records=None)
    
    # Preparar datos
    all_signals = sddb_signals + nsrdb_signals
    all_labels = np.concatenate([sddb_labels, nsrdb_labels])
    all_metadata = sddb_metadata + nsrdb_metadata
    
    # Segmentar señales en ventanas de 30 segundos
    fs = 128.0
    window_size = 30.0
    X_segments, y_segments = prepare_training_data(all_signals, all_labels, fs=fs, window_size=window_size)
    
    # Dividir en train/test (igual que en entrenamiento)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_segments, y_segments, test_size=0.2, random_state=42, stratify=y_segments
    )
    
    print(f"📊 Conjunto de prueba: {len(X_test)} segmentos")
    print(f"   Normal: {np.sum(y_test == 0)}, SCD: {np.sum(y_test == 1)}")
    
    # Procesar conjunto de prueba completo
    results = {
        'test_set_predictions': [],
        'summary': {
            'total_processed': 0,
            'sparse_predictions': {'normal': 0, 'scd': 0, 'correct': 0, 'incorrect': 0},
            'hierarchical_predictions': {'normal': 0, 'scd': 0, 'correct': 0, 'incorrect': 0},
            'hybrid_predictions': {'normal': 0, 'scd': 0, 'correct': 0, 'incorrect': 0},
            'ensemble_predictions': {'normal': 0, 'scd': 0, 'correct': 0, 'incorrect': 0},
            'true_labels': {'normal': int(np.sum(y_test == 0)), 'scd': int(np.sum(y_test == 1))}
        },
        'metrics': {
            'sparse': {},
            'hierarchical': {},
            'hybrid': {},
            'ensemble': {}
        }
    }
    
    print(f"\n🔍 Procesando {len(X_test)} segmentos del conjunto de prueba...")
    
    # Procesar en lotes para eficiencia
    batch_size = 50
    all_sparse_preds = []
    all_hierarchical_preds = []
    all_hybrid_preds = []
    all_sparse_probas = []
    all_hierarchical_probas = []
    all_hybrid_probas = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_signals = X_test[i:batch_end]
        batch_labels = y_test[i:batch_end]
        
        # Preprocesar batch
        processed_batch = []
        for signal in batch_signals:
            processed = preprocess_unified(signal, fs=fs, target_fs=128.0)
            processed_batch.append(processed['signal'])
        
        # Predecir con cada modelo
        if 'sparse' in models:
            try:
                batch_preds = models['sparse'].predict(processed_batch)
                batch_probas = models['sparse'].predict_proba(processed_batch)
                all_sparse_preds.extend(batch_preds)
                all_sparse_probas.extend(batch_probas)
            except Exception as e:
                print(f"⚠️  Error en predicción sparse batch {i}: {e}")
                all_sparse_preds.extend([0] * len(batch_signals))
                all_sparse_probas.extend([[0.5, 0.5]] * len(batch_signals))
        
        if 'hierarchical' in models:
            try:
                batch_preds = models['hierarchical'].predict(processed_batch, fs=fs)
                batch_probas = models['hierarchical'].predict_proba(processed_batch, fs=fs)
                all_hierarchical_preds.extend(batch_preds)
                all_hierarchical_probas.extend(batch_probas)
            except Exception as e:
                print(f"⚠️  Error en predicción hierarchical batch {i}: {e}")
                all_hierarchical_preds.extend([0] * len(batch_signals))
                all_hierarchical_probas.extend([[0.5, 0.5]] * len(batch_signals))
        
        if 'hybrid' in models:
            try:
                batch_preds = models['hybrid'].predict(processed_batch, fs=fs)
                batch_probas = models['hybrid'].predict_proba(processed_batch, fs=fs)
                all_hybrid_preds.extend(batch_preds)
                all_hybrid_probas.extend(batch_probas)
            except Exception as e:
                print(f"⚠️  Error en predicción hybrid batch {i}: {e}")
                all_hybrid_preds.extend([0] * len(batch_signals))
                all_hybrid_probas.extend([[0.5, 0.5]] * len(batch_signals))
        
        if (i + batch_size) % 200 == 0:
            print(f"   Procesados {min(i + batch_size, len(X_test))}/{len(X_test)} segmentos...")
    
    # Convertir a arrays numpy
    all_sparse_preds = np.array(all_sparse_preds)
    all_hierarchical_preds = np.array(all_hierarchical_preds)
    all_hybrid_preds = np.array(all_hybrid_preds)
    all_sparse_probas = np.array(all_sparse_probas)
    all_hierarchical_probas = np.array(all_hierarchical_probas)
    all_hybrid_probas = np.array(all_hybrid_probas)
    
    # Calcular ensemble (votación mayoritaria)
    ensemble_preds = []
    ensemble_probas = []
    for i in range(len(X_test)):
        votes = []
        probas_scd = []
        
        if len(all_sparse_preds) > i:
            votes.append(int(all_sparse_preds[i]))
            if len(all_sparse_probas) > i and len(all_sparse_probas[i]) > 1:
                probas_scd.append(float(all_sparse_probas[i][1]))
        
        if len(all_hierarchical_preds) > i:
            votes.append(int(all_hierarchical_preds[i]))
            if len(all_hierarchical_probas) > i and len(all_hierarchical_probas[i]) > 1:
                probas_scd.append(float(all_hierarchical_probas[i][1]))
        
        if len(all_hybrid_preds) > i:
            votes.append(int(all_hybrid_preds[i]))
            if len(all_hybrid_probas) > i and len(all_hybrid_probas[i]) > 1:
                probas_scd.append(float(all_hybrid_probas[i][1]))
        
        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        ensemble_preds.append(ensemble_pred)
        
        avg_proba_scd = np.mean(probas_scd) if probas_scd else 0.5
        ensemble_probas.append([1 - avg_proba_scd, avg_proba_scd])
    
    ensemble_preds = np.array(ensemble_preds)
    ensemble_probas = np.array(ensemble_probas)
    
    # Calcular métricas y actualizar resumen
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if len(all_sparse_preds) == len(y_test):
        results['summary']['sparse_predictions']['normal'] = int(np.sum(all_sparse_preds == 0))
        results['summary']['sparse_predictions']['scd'] = int(np.sum(all_sparse_preds == 1))
        results['summary']['sparse_predictions']['correct'] = int(np.sum(all_sparse_preds == y_test))
        results['summary']['sparse_predictions']['incorrect'] = int(np.sum(all_sparse_preds != y_test))
        results['metrics']['sparse'] = {
            'accuracy': float(accuracy_score(y_test, all_sparse_preds)),
            'precision': float(precision_score(y_test, all_sparse_preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, all_sparse_preds, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, all_sparse_preds, average='weighted', zero_division=0))
        }
    
    if len(all_hierarchical_preds) == len(y_test):
        results['summary']['hierarchical_predictions']['normal'] = int(np.sum(all_hierarchical_preds == 0))
        results['summary']['hierarchical_predictions']['scd'] = int(np.sum(all_hierarchical_preds == 1))
        results['summary']['hierarchical_predictions']['correct'] = int(np.sum(all_hierarchical_preds == y_test))
        results['summary']['hierarchical_predictions']['incorrect'] = int(np.sum(all_hierarchical_preds != y_test))
        results['metrics']['hierarchical'] = {
            'accuracy': float(accuracy_score(y_test, all_hierarchical_preds)),
            'precision': float(precision_score(y_test, all_hierarchical_preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, all_hierarchical_preds, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, all_hierarchical_preds, average='weighted', zero_division=0))
        }
    
    if len(all_hybrid_preds) == len(y_test):
        results['summary']['hybrid_predictions']['normal'] = int(np.sum(all_hybrid_preds == 0))
        results['summary']['hybrid_predictions']['scd'] = int(np.sum(all_hybrid_preds == 1))
        results['summary']['hybrid_predictions']['correct'] = int(np.sum(all_hybrid_preds == y_test))
        results['summary']['hybrid_predictions']['incorrect'] = int(np.sum(all_hybrid_preds != y_test))
        results['metrics']['hybrid'] = {
            'accuracy': float(accuracy_score(y_test, all_hybrid_preds)),
            'precision': float(precision_score(y_test, all_hybrid_preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, all_hybrid_preds, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, all_hybrid_preds, average='weighted', zero_division=0))
        }
    
    if len(ensemble_preds) == len(y_test):
        results['summary']['ensemble_predictions']['normal'] = int(np.sum(ensemble_preds == 0))
        results['summary']['ensemble_predictions']['scd'] = int(np.sum(ensemble_preds == 1))
        results['summary']['ensemble_predictions']['correct'] = int(np.sum(ensemble_preds == y_test))
        results['summary']['ensemble_predictions']['incorrect'] = int(np.sum(ensemble_preds != y_test))
        results['metrics']['ensemble'] = {
            'accuracy': float(accuracy_score(y_test, ensemble_preds)),
            'precision': float(precision_score(y_test, ensemble_preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, ensemble_preds, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, ensemble_preds, average='weighted', zero_division=0))
        }
    
    results['summary']['total_processed'] = len(X_test)
    
    # Guardar algunas señales de ejemplo para visualización (primeras 5 de cada clase)
    example_indices = []
    normal_indices = np.where(y_test == 0)[0][:5]
    scd_indices = np.where(y_test == 1)[0][:5]
    example_indices = list(normal_indices) + list(scd_indices)
    
    results['visualization_examples'] = []
    for idx in example_indices:
        signal = X_test[idx]
        true_label = y_test[idx]
        
        processed = preprocess_unified(signal, fs=fs, target_fs=128.0)
        processed_signal = processed['signal']
        signal_image = signal_to_base64(processed_signal, fs, duration=10.0)
        
        # Guardar datos de la señal para gráficas interactivas Plotly (primeros 10 segundos)
        duration_samples = int(10.0 * fs)  # 10 segundos a 128 Hz = 1280 muestras
        signal_data_for_plot = processed_signal[:duration_samples].tolist()
        time_axis = (np.arange(len(signal_data_for_plot)) / fs).tolist()
        
        example = {
            'signal_id': f"test_{idx}",
            'true_label': int(true_label),
            'true_label_name': 'SCD' if true_label == 1 else 'Normal',
            'signal_image': signal_image,  # Mantener imagen para compatibilidad
            'signal_data': signal_data_for_plot,  # Datos para Plotly
            'time_axis': time_axis,  # Eje de tiempo para Plotly
            'fs': float(fs),  # Frecuencia de muestreo
            'predictions': {},
            'probabilities': {}
        }
        
        if len(all_sparse_preds) > idx:
            example['predictions']['sparse'] = int(all_sparse_preds[idx])
            example['predictions']['sparse_name'] = 'SCD' if all_sparse_preds[idx] == 1 else 'Normal'
            if len(all_sparse_probas) > idx:
                example['probabilities']['sparse'] = {
                    'normal': float(all_sparse_probas[idx][0]),
                    'scd': float(all_sparse_probas[idx][1]) if len(all_sparse_probas[idx]) > 1 else float(all_sparse_probas[idx][0])
                }
        
        if len(all_hierarchical_preds) > idx:
            example['predictions']['hierarchical'] = int(all_hierarchical_preds[idx])
            example['predictions']['hierarchical_name'] = 'SCD' if all_hierarchical_preds[idx] == 1 else 'Normal'
            if len(all_hierarchical_probas) > idx:
                example['probabilities']['hierarchical'] = {
                    'normal': float(all_hierarchical_probas[idx][0]),
                    'scd': float(all_hierarchical_probas[idx][1]) if len(all_hierarchical_probas[idx]) > 1 else float(all_hierarchical_probas[idx][0])
                }
        
        if len(all_hybrid_preds) > idx:
            example['predictions']['hybrid'] = int(all_hybrid_preds[idx])
            example['predictions']['hybrid_name'] = 'SCD' if all_hybrid_preds[idx] == 1 else 'Normal'
            if len(all_hybrid_probas) > idx:
                example['probabilities']['hybrid'] = {
                    'normal': float(all_hybrid_probas[idx][0]),
                    'scd': float(all_hybrid_probas[idx][1]) if len(all_hybrid_probas[idx]) > 1 else float(all_hybrid_probas[idx][0])
                }
        
        if len(ensemble_preds) > idx:
            example['predictions']['ensemble'] = int(ensemble_preds[idx])
            example['predictions']['ensemble_name'] = 'SCD' if ensemble_preds[idx] == 1 else 'Normal'
            if len(ensemble_probas) > idx:
                example['probabilities']['ensemble'] = {
                    'normal': float(ensemble_probas[idx][0]),
                    'scd': float(ensemble_probas[idx][1]) if len(ensemble_probas[idx]) > 1 else float(ensemble_probas[idx][0])
                }
        
        results['visualization_examples'].append(example)
    
    # Guardar resultados
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {output_file}")
    print(f"   Total procesado: {results['summary']['total_processed']} segmentos del conjunto de prueba")
    print(f"\n📊 Métricas de Rendimiento:")
    if results['metrics']['sparse']:
        print(f"   Sparse - Accuracy: {results['metrics']['sparse']['accuracy']*100:.2f}%")
    if results['metrics']['hierarchical']:
        print(f"   Hierarchical - Accuracy: {results['metrics']['hierarchical']['accuracy']*100:.2f}%")
    if results['metrics']['hybrid']:
        print(f"   Hybrid - Accuracy: {results['metrics']['hybrid']['accuracy']*100:.2f}%")
    if results['metrics']['ensemble']:
        print(f"   Ensemble - Accuracy: {results['metrics']['ensemble']['accuracy']*100:.2f}%")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar predicciones en tiempo real')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--output', type=str, default='results/realtime_predictions.json',
                       help='Archivo de salida JSON')
    parser.add_argument('--use-test-set', action='store_true', default=True,
                       help='Usar conjunto de prueba completo')
    
    args = parser.parse_args()
    
    results = process_real_signals(
        args.models_dir,
        args.data_dir,
        args.output,
        use_test_set=args.use_test_set
    )
    
    if results:
        print("\n📊 Resumen de Predicciones:")
        print(f"   Sparse - Normal: {results['summary']['sparse_predictions']['normal']}, SCD: {results['summary']['sparse_predictions']['scd']}, Correctas: {results['summary']['sparse_predictions']['correct']}")
        print(f"   Hierarchical - Normal: {results['summary']['hierarchical_predictions']['normal']}, SCD: {results['summary']['hierarchical_predictions']['scd']}, Correctas: {results['summary']['hierarchical_predictions']['correct']}")
        print(f"   Hybrid - Normal: {results['summary']['hybrid_predictions']['normal']}, SCD: {results['summary']['hybrid_predictions']['scd']}, Correctas: {results['summary']['hybrid_predictions']['correct']}")
        print(f"   Ensemble - Normal: {results['summary']['ensemble_predictions']['normal']}, SCD: {results['summary']['ensemble_predictions']['scd']}, Correctas: {results['summary']['ensemble_predictions']['correct']}")

if __name__ == "__main__":
    main()

