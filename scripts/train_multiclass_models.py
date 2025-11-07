"""
Script para análisis multi-clase vs binario
Entrena modelos multi-clase (distinguir entre intervalos temporales) y compara con binario
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.analysis_data_structures import MulticlassAnalysisResults, MulticlassResult

def load_temporal_data(temporal_data_file: str = 'results/temporal_intervals_data.pkl') -> Dict:
    """Cargar datos temporales"""
    with open(temporal_data_file, 'rb') as f:
        return pickle.load(f)

def prepare_multiclass_data(temporal_data: Dict) -> tuple:
    """
    Preparar datos para clasificación multi-clase
    Clases: Normal (0), 5min (1), 10min (2), 15min (3), 20min (4), 25min (5), 30min (6)
    """
    all_segments = []
    all_labels = []
    
    # Datos NSRDB (clase Normal = 0)
    nsrdb_segments = temporal_data['nsrdb']['segments']
    for seg in nsrdb_segments:
        all_segments.append(seg)
        all_labels.append(0)  # Normal
    
    # Datos SDDB (clases según intervalo temporal)
    sddb_segments = temporal_data['sddb']['segments']
    sddb_labels = temporal_data['sddb']['labels']
    
    # Mapear intervalos a clases numéricas
    interval_to_class = {
        5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6
    }
    
    for seg, label in zip(sddb_segments, sddb_labels):
        all_segments.append(seg)
        if isinstance(label, (int, float)):
            class_label = interval_to_class.get(int(label), 1)
        else:
            class_label = 1  # Default
        all_labels.append(class_label)
    
    return all_segments, np.array(all_labels)

def train_multiclass_model(model_type: str, X_train: List[np.ndarray], y_train: np.ndarray,
                          X_test: List[np.ndarray], y_test: np.ndarray, fs: float = 128.0) -> Dict:
    """Entrenar modelo multi-clase"""
    print(f"\n🔄 Entrenando modelo {model_type} multi-clase...")
    
    n_classes = len(np.unique(y_train))
    
    if model_type == 'sparse':
        model = SparseRepresentationClassifier(
            n_atoms=30,
            n_nonzero_coefs=3,
            svm_kernel='rbf',
            multi_class=True
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
    
    elif model_type == 'hierarchical':
        model = HierarchicalFusionClassifier(
            tcn_filters=32,
            fusion_dim=64,
            n_classes=n_classes
        )
        model.fit(X_train, y_train, fs=fs, epochs=20, batch_size=8)
        predictions = model.predict(X_test, fs=fs)
        probabilities = model.predict_proba(X_test, fs=fs)
    
    elif model_type == 'hybrid':
        model = HybridSCDClassifier(
            n_atoms=50,
            n_nonzero_coefs=5,
            wavelet='db4',
            tcn_filters=32,
            fusion_dim=64
        )
        model.fit(X_train, y_train, fs=fs, epochs_hierarchical=10, batch_size=8)
        predictions = model.predict(X_test, fs=fs)
        probabilities = model.predict_proba(X_test, fs=fs)
    
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    # AUC-ROC multi-clase
    try:
        if probabilities.shape[1] > 2:
            auc = roc_auc_score(y_test, probabilities, multi_class='ovo')
        else:
            auc = roc_auc_score(y_test, probabilities[:, 1])
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities,
        'n_classes': n_classes,
        'n_samples': len(y_test)
    }

def train_binary_model(model_type: str, X_train: List[np.ndarray], y_train: np.ndarray,
                      X_test: List[np.ndarray], y_test: np.ndarray, fs: float = 128.0) -> Dict:
    """Entrenar modelo binario (SCD vs Normal)"""
    print(f"\n🔄 Entrenando modelo {model_type} binario...")
    
    # Convertir a binario: Normal (0) vs SCD (1-6 -> 1)
    y_train_binary = (y_train > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)
    
    if model_type == 'sparse':
        model = SparseRepresentationClassifier(
            n_atoms=30,
            n_nonzero_coefs=3,
            svm_kernel='rbf',
            multi_class=False
        )
        model.fit(X_train, y_train_binary)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
    
    elif model_type == 'hierarchical':
        model = HierarchicalFusionClassifier(
            tcn_filters=32,
            fusion_dim=64,
            n_classes=2
        )
        model.fit(X_train, y_train_binary, fs=fs, epochs=20, batch_size=8)
        predictions = model.predict(X_test, fs=fs)
        probabilities = model.predict_proba(X_test, fs=fs)
    
    elif model_type == 'hybrid':
        model = HybridSCDClassifier(
            n_atoms=50,
            n_nonzero_coefs=5,
            wavelet='db4',
            tcn_filters=32,
            fusion_dim=64
        )
        model.fit(X_train, y_train_binary, fs=fs, epochs_hierarchical=10, batch_size=8)
        predictions = model.predict(X_test, fs=fs)
        probabilities = model.predict_proba(X_test, fs=fs)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test_binary, predictions)
    precision = precision_score(y_test_binary, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test_binary, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test_binary, predictions, average='weighted', zero_division=0)
    
    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(y_test_binary, probabilities[:, 1])
        else:
            auc = 0.0
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_test_binary, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities,
        'n_classes': 2,
        'n_samples': len(y_test_binary)
    }

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis multi-clase vs binario')
    parser.add_argument('--temporal-data', type=str, default='results/temporal_intervals_data.pkl',
                       help='Archivo con datos temporales')
    parser.add_argument('--output', type=str, default='results/multiclass_results.pkl',
                       help='Archivo de salida')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['sparse', 'hierarchical', 'hybrid'],
                       default=['hierarchical'],
                       help='Modelos a entrenar')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporción de datos para prueba')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔀 ANÁLISIS MULTI-CLASE VS BINARIO")
    print("=" * 70)
    
    # Cargar datos temporales
    print(f"\n📂 Cargando datos desde: {args.temporal_data}")
    temporal_data = load_temporal_data(args.temporal_data)
    
    # Preparar datos multi-clase
    X, y = prepare_multiclass_data(temporal_data)
    print(f"📊 Total muestras: {len(X)}")
    print(f"   Clases: {sorted(np.unique(y))} (Normal=0, 5min=1, 10min=2, 15min=3, 20min=4, 25min=5, 30min=6)")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"   Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")
    
    # Crear objeto de resultados
    multiclass_results = MulticlassAnalysisResults()
    
    # Entrenar modelos multi-clase y binarios
    for model_type in args.models:
        print(f"\n{'='*70}")
        print(f"Modelo: {model_type.upper()}")
        print(f"{'='*70}")
        
        # Multi-clase
        multiclass_metrics = train_multiclass_model(
            model_type, X_train, y_train, X_test, y_test, fs=128.0
        )
        
        # Multi-clase
        multiclass_metrics = train_multiclass_model(
            model_type, X_train, y_train, X_test, y_test, fs=128.0
        )
        
        # Preparar datos para MulticlassResult
        unique_classes = sorted(np.unique(y_test))
        class_names = ['Normal', '5min', '10min', '15min', '20min', '25min', '30min']
        classes = [class_names[i] if i < len(class_names) else f'Class_{i}' for i in unique_classes]
        
        # Calcular métricas por clase
        from sklearn.metrics import precision_recall_fscore_support
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_test, multiclass_metrics['predictions'], labels=unique_classes, zero_division=0
        )
        
        precision_dict = {classes[i]: float(precision_per_class[i]) for i in range(len(classes))}
        recall_dict = {classes[i]: float(recall_per_class[i]) for i in range(len(classes))}
        f1_dict = {classes[i]: float(f1_per_class[i]) for i in range(len(classes))}
        n_samples_dict = {classes[i]: int(support[i]) for i in range(len(classes))}
        
        multiclass_result = MulticlassResult(
            classes=classes,
            accuracy=multiclass_metrics['accuracy'],
            precision_per_class=precision_dict,
            recall_per_class=recall_dict,
            f1_per_class=f1_dict,
            confusion_matrix=multiclass_metrics['confusion_matrix'],
            predictions=multiclass_metrics['predictions'],
            probabilities=multiclass_metrics['probabilities'],
            n_samples_per_class=n_samples_dict
        )
        multiclass_results.multiclass_results[model_type] = multiclass_result
        
        # Binario - guardar solo accuracy
        binary_metrics = train_binary_model(
            model_type, X_train, y_train, X_test, y_test, fs=128.0
        )
        multiclass_results.binary_results[model_type] = binary_metrics['accuracy']
        
        print(f"\n✅ Multi-clase: Accuracy={multiclass_metrics['accuracy']:.4f}, AUC={multiclass_metrics['auc_roc']:.4f}")
        print(f"✅ Binario: Accuracy={binary_metrics['accuracy']:.4f}, AUC={binary_metrics['auc_roc']:.4f}")
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    multiclass_results.save(str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"📁 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()

