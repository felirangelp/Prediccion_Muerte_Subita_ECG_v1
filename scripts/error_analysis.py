"""
Script para análisis de errores: falsos positivos, falsos negativos y patrones
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.analysis_data_structures import ErrorAnalysisResults
from scripts.train_models import load_dataset, prepare_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def analyze_errors(model, X_test, y_test, model_name: str, fs: float = 128.0):
    """
    Analizar errores de clasificación
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        model_name: Nombre del modelo
        fs: Frecuencia de muestreo
    """
    print(f"   Analizando errores para {model_name}...")
    
    # Predecir
    try:
        if 'Hierarchical' in model_name or 'Hybrid' in model_name:
            predictions = model.predict(X_test, fs=fs)
        else:
            predictions = model.predict(X_test)
    except Exception as e:
        print(f"   ⚠️  Error prediciendo: {e}")
        return None
    
    # Identificar errores
    false_positives = []
    false_negatives = []
    
    for i, (pred, true_label) in enumerate(zip(predictions, y_test)):
        if pred == 1 and true_label == 0:
            false_positives.append(i)
        elif pred == 0 and true_label == 1:
            false_negatives.append(i)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    
    # Analizar patrones
    error_patterns = {
        'false_positive_rate': len(false_positives) / len(y_test[y_test == 0]) if len(y_test[y_test == 0]) > 0 else 0,
        'false_negative_rate': len(false_negatives) / len(y_test[y_test == 1]) if len(y_test[y_test == 1]) > 0 else 0,
        'total_errors': len(false_positives) + len(false_negatives),
        'error_rate': (len(false_positives) + len(false_negatives)) / len(y_test)
    }
    
    # Analizar características de señales con error (si es posible)
    error_samples_metadata = []
    
    # Para falsos positivos
    for idx in false_positives[:10]:  # Limitar a 10 para no sobrecargar
        try:
            signal = X_test[idx]
            metadata = {
                'index': int(idx),
                'error_type': 'false_positive',
                'predicted': int(predictions[idx]),
                'true_label': int(y_test[idx]),
                'signal_length': len(signal),
                'signal_mean': float(np.mean(signal)),
                'signal_std': float(np.std(signal)),
                'signal_max': float(np.max(signal)),
                'signal_min': float(np.min(signal))
            }
            error_samples_metadata.append(metadata)
        except:
            continue
    
    # Para falsos negativos
    for idx in false_negatives[:10]:
        try:
            signal = X_test[idx]
            metadata = {
                'index': int(idx),
                'error_type': 'false_negative',
                'predicted': int(predictions[idx]),
                'true_label': int(y_test[idx]),
                'signal_length': len(signal),
                'signal_mean': float(np.mean(signal)),
                'signal_std': float(np.std(signal)),
                'signal_max': float(np.max(signal)),
                'signal_min': float(np.min(signal))
            }
            error_samples_metadata.append(metadata)
        except:
            continue
    
    print(f"      Falsos positivos: {len(false_positives)}")
    print(f"      Falsos negativos: {len(false_negatives)}")
    print(f"      Tasa de error: {error_patterns['error_rate']:.4f}")
    
    return ErrorAnalysisResults(
        model_name=model_name.lower().replace(' ', '_'),
        false_positives=false_positives,
        false_negatives=false_negatives,
        error_patterns=error_patterns,
        confusion_matrix=cm,
        error_samples_metadata=error_samples_metadata
    )


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar errores de clasificación')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--output', type=str, default='results/error_analysis_results.pkl',
                       help='Archivo de salida con resultados')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    parser.add_argument('--model', type=str, default='all', choices=['sparse', 'hierarchical', 'hybrid', 'all'],
                       help='Modelo a analizar')
    
    args = parser.parse_args()
    
    # Crear directorio de resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # Cargar modelos
    models_dir = Path(args.models_dir)
    models = {}
    model_names = {}
    
    if args.model in ['sparse', 'all']:
        if (models_dir / 'sparse_classifier.pkl').exists():
            try:
                models['sparse'] = SparseRepresentationClassifier.load(
                    str(models_dir / 'sparse_classifier.pkl')
                )
                model_names['sparse'] = 'Sparse Representations'
                print("✅ Modelo sparse cargado")
            except Exception as e:
                print(f"⚠️  Error cargando sparse: {e}")
    
    if args.model in ['hierarchical', 'all']:
        if (models_dir / 'hierarchical_classifier_fusion.h5').exists():
            try:
                models['hierarchical'] = HierarchicalFusionClassifier.load(
                    str(models_dir / 'hierarchical_classifier')
                )
                model_names['hierarchical'] = 'Hierarchical Fusion'
                print("✅ Modelo hierarchical cargado")
            except Exception as e:
                print(f"⚠️  Error cargando hierarchical: {e}")
    
    if args.model in ['hybrid', 'all']:
        if (models_dir / 'hybrid_model_sparse.pkl').exists():
            try:
                models['hybrid'] = HybridSCDClassifier.load(
                    str(models_dir / 'hybrid_model')
                )
                model_names['hybrid'] = 'Hybrid Model'
                print("✅ Modelo hybrid cargado")
            except Exception as e:
                print(f"⚠️  Error cargando hybrid: {e}")
    
    if not models:
        print("❌ No se encontraron modelos entrenados")
        return
    
    # Cargar datos de prueba
    print("\n📂 Cargando datos de prueba...")
    sddb_path = Path(args.data_dir) / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = Path(args.data_dir) / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    sddb_signals, sddb_labels, _ = load_dataset(str(sddb_path), 'sddb', args.max_records)
    nsrdb_signals, nsrdb_labels, _ = load_dataset(str(nsrdb_path), 'nsrdb', args.max_records)
    
    all_signals = sddb_signals + nsrdb_signals
    all_labels = np.concatenate([sddb_labels, nsrdb_labels])
    
    X, y = prepare_training_data(all_signals, all_labels, fs=128.0, window_size=30.0)
    
    # Dividir en test
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Datos de prueba: {len(X_test)} muestras")
    
    # Analizar cada modelo
    all_results = {}
    
    if 'sparse' in models:
        print("\n📊 Analizando errores del modelo Sparse...")
        result = analyze_errors(
            models['sparse'], X_test, y_test, model_names['sparse'], fs=128.0
        )
        if result:
            all_results['sparse'] = result
    
    if 'hierarchical' in models:
        print("\n📊 Analizando errores del modelo Hierarchical...")
        result = analyze_errors(
            models['hierarchical'], X_test, y_test, model_names['hierarchical'], fs=128.0
        )
        if result:
            all_results['hierarchical'] = result
    
    if 'hybrid' in models:
        print("\n📊 Analizando errores del modelo Hybrid...")
        result = analyze_errors(
            models['hybrid'], X_test, y_test, model_names['hybrid'], fs=128.0
        )
        if result:
            all_results['hybrid'] = result
    
    # Guardar resultados
    with open(args.output, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n✅ Resultados guardados en: {args.output}")
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE ANÁLISIS DE ERRORES")
    print("="*60)
    for model_name, result in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"   Falsos positivos: {len(result.false_positives)}")
        print(f"   Falsos negativos: {len(result.false_negatives)}")
        print(f"   Tasa de error: {result.error_patterns['error_rate']:.4f}")

if __name__ == "__main__":
    main()

