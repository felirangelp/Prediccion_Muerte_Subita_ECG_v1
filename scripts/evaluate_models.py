"""
Script para evaluar modelos y generar métricas completas
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.utils import load_ecg_record, list_available_records

def evaluate_model(model, X_test, y_test, model_name: str, fs: float = 128.0):
    """
    Evaluar modelo completo
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        model_name: Nombre del modelo
        fs: Frecuencia de muestreo
    """
    print(f"\n📊 Evaluando {model_name}...")
    
    # Predecir
    if hasattr(model, 'predict'):
        if 'Hierarchical' in model_name or 'Hybrid' in model_name:
            predictions = model.predict(X_test, fs=fs)
            probabilities = model.predict_proba(X_test, fs=fs)
        else:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
    else:
        return None
    
    # Métricas básicas
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    # AUC-ROC
    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            auc = roc_auc_score(y_test, probabilities, multi_class='ovo')
    except:
        auc = 0.0
    
    # Curva ROC
    try:
        if probabilities.shape[1] == 2:
            fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
        else:
            fpr, tpr, thresholds = None, None, None
    except:
        fpr, tpr, thresholds = None, None, None
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr, thresholds),
        'predictions': predictions,
        'probabilities': probabilities
    }
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    
    return results

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar modelos de predicción SCD')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--output', type=str, default='results/evaluation_results.pkl',
                       help='Archivo de salida con resultados')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    
    args = parser.parse_args()
    
    # Crear directorio de resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # Cargar modelos
    models_dir = Path(args.models_dir)
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
        return
    
    # Cargar datos de prueba
    print("\n📂 Cargando datos de prueba...")
    sddb_path = Path(args.data_dir) / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = Path(args.data_dir) / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    # Cargar datos (simplificado - usar subset para evaluación rápida)
    from scripts.train_models import load_dataset, prepare_training_data
    
    sddb_signals, sddb_labels, _ = load_dataset(str(sddb_path), 'sddb', args.max_records)
    nsrdb_signals, nsrdb_labels, _ = load_dataset(str(nsrdb_path), 'nsrdb', args.max_records)
    
    all_signals = sddb_signals + nsrdb_signals
    all_labels = np.concatenate([sddb_labels, nsrdb_labels])
    
    X, y = prepare_training_data(all_signals, all_labels, fs=128.0, window_size=30.0)
    
    # Dividir en test
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Datos de prueba: {len(X_test)} muestras")
    
    # Evaluar cada modelo
    all_results = {}
    
    if 'sparse' in models:
        results = evaluate_model(
            models['sparse'], X_test, y_test, 'Sparse Representations', fs=128.0
        )
        if results:
            all_results['sparse'] = results
    
    if 'hierarchical' in models:
        results = evaluate_model(
            models['hierarchical'], X_test, y_test, 'Hierarchical Fusion', fs=128.0
        )
        if results:
            all_results['hierarchical'] = results
    
    if 'hybrid' in models:
        results = evaluate_model(
            models['hybrid'], X_test, y_test, 'Hybrid Model', fs=128.0
        )
        if results:
            all_results['hybrid'] = results
    
    # Guardar resultados
    with open(args.output, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n✅ Resultados guardados en: {args.output}")
    
    # Crear resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Modelo': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1_score']:.4f}",
            'AUC-ROC': f"{results['auc_roc']:.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

