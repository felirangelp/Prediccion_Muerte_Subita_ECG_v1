"""
Script para entrenar todos los modelos (Sparse, Hierarchical, Hybrid)
"""

import sys
from pathlib import Path
from typing import Optional, List
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_ecg_record, list_available_records
from src.preprocessing_unified import preprocess_for_sparse_method, preprocess_for_hierarchical_method
from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier

def load_dataset(dataset_path: str, dataset_type: str, max_records: Optional[int] = None):
    """
    Cargar dataset completo
    
    Args:
        dataset_path: Ruta al dataset
        dataset_type: 'sddb' o 'nsrdb'
        max_records: Número máximo de registros a cargar
    """
    records = list_available_records(dataset_path)
    
    if max_records:
        records = records[:max_records]
    
    signals = []
    labels = []
    metadata_list = []
    
    print(f"📂 Cargando {len(records)} registros de {dataset_type}...")
    
    for record_name in tqdm(records):
        try:
            record_path = str(Path(dataset_path) / record_name)
            signal, metadata = load_ecg_record(record_path, channels=[0])
            
            # Usar solo primer canal
            if signal.ndim > 1:
                signal = signal[:, 0]
            
            signals.append(signal)
            labels.append(1 if dataset_type == 'sddb' else 0)
            metadata_list.append(metadata)
            
        except Exception as e:
            print(f"⚠️  Error cargando {record_name}: {e}")
            continue
    
    return signals, np.array(labels), metadata_list

def prepare_training_data(signals: List[np.ndarray], labels: np.ndarray, 
                         fs: float, window_size: float = 60.0):
    """
    Preparar datos para entrenamiento segmentando en ventanas
    
    Args:
        signals: Lista de señales
        labels: Etiquetas
        fs: Frecuencia de muestreo
        window_size: Tamaño de ventana en segundos
    """
    X = []
    y = []
    
    window_samples = int(window_size * fs)
    
    for signal, label in zip(signals, labels):
        # Segmentar señal en ventanas de 60 segundos
        for start in range(0, len(signal) - window_samples + 1, window_samples):
            segment = signal[start:start + window_samples]
            X.append(segment)
            y.append(label)
        
        # Si queda un segmento final, también incluirlo
        if len(signal) > window_samples:
            segment = signal[-window_samples:]
            X.append(segment)
            y.append(label)
    
    return X, np.array(y)

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción SCD')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio para guardar modelos')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    parser.add_argument('--train-sparse', action='store_true',
                       help='Entrenar modelo sparse')
    parser.add_argument('--train-hierarchical', action='store_true',
                       help='Entrenar modelo hierarchical')
    parser.add_argument('--train-hybrid', action='store_true',
                       help='Entrenar modelo hybrid')
    parser.add_argument('--train-all', action='store_true',
                       help='Entrenar todos los modelos')
    
    args = parser.parse_args()
    
    # Crear directorio de modelos
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)
    
    # Cargar datasets
    sddb_path = Path(args.data_dir) / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = Path(args.data_dir) / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    if not sddb_path.exists() or not nsrdb_path.exists():
        print("❌ Datasets no encontrados. Por favor descarga los datasets primero.")
        return
    
    print("📊 Cargando datasets...")
    sddb_signals, sddb_labels, _ = load_dataset(str(sddb_path), 'sddb', args.max_records)
    nsrdb_signals, nsrdb_labels, _ = load_dataset(str(nsrdb_path), 'nsrdb', args.max_records)
    
    # Combinar datasets
    all_signals = sddb_signals + nsrdb_signals
    all_labels = np.concatenate([sddb_labels, nsrdb_labels])
    
    # Preparar datos (segmentar en ventanas de 60 segundos)
    print("🔧 Preparando datos para entrenamiento...")
    # OPTIMIZACIÓN: Reducir tamaño de ventana para menos datos
    X, y = prepare_training_data(all_signals, all_labels, fs=128.0, window_size=30.0)  # Reducido de 60s a 30s
    
    print(f"✅ Datos preparados: {len(X)} muestras, {len(np.unique(y))} clases")
    
    # Dividir en train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 División: {len(X_train)} train, {len(X_test)} test")
    
    # OPTIMIZACIÓN: Entrenar primero modelos que usan GPU (más rápidos)
    # Luego entrenar modelo Sparse (más lento, solo CPU)
    
    # Entrenar modelos que usan GPU primero
    if args.train_all or args.train_hierarchical:
        print("\n" + "="*60)
        print("🎯 Entrenando Modelo 2: Fusión Jerárquica (USA GPU)")
        print("="*60)
        
        hierarchical_classifier = HierarchicalFusionClassifier(
            tcn_filters=32,
            fusion_dim=64,
            n_classes=2
        )
        
        hierarchical_classifier.fit(
            X_train, y_train, fs=128.0,
            epochs=20,  # Reducido para demo
            batch_size=8
        )
        
        hierarchical_classifier.save(str(models_dir / 'hierarchical_classifier'))
        
        # Evaluar
        predictions = hierarchical_classifier.predict(X_test, fs=128.0)
        accuracy = np.mean(predictions == y_test)
        print(f"✅ Precisión en test: {accuracy:.4f}")
    
    if args.train_all or args.train_hybrid:
        print("\n" + "="*60)
        print("🎯 Entrenando Modelo Híbrido (USA GPU)")
        print("="*60)
        
        hybrid_classifier = HybridSCDClassifier(
            n_atoms=50,
            n_nonzero_coefs=5,
            wavelet='db4',
            tcn_filters=32,
            fusion_dim=64
        )
        
        hybrid_classifier.fit(
            X_train, y_train, fs=128.0,
            epochs_hierarchical=10,  # Reducido para demo
            batch_size=8
        )
        
        hybrid_classifier.save(str(models_dir / 'hybrid_model'))
        
        # Evaluar
        predictions = hybrid_classifier.predict(X_test, fs=128.0)
        accuracy = np.mean(predictions == y_test)
        print(f"✅ Precisión en test: {accuracy:.4f}")
    
    # Entrenar modelo Sparse al final (más lento, solo CPU)
    if args.train_all or args.train_sparse:
        print("\n" + "="*60)
        print("🎯 Entrenando Modelo 1: Representaciones Dispersas (SOLO CPU)")
        print("="*60)
        print("⚠️  Este modelo es más lento porque usa solo CPU")
        print("    Los modelos anteriores ya usaron GPU Metal")
        print("="*60)
        
        sparse_classifier = SparseRepresentationClassifier(
            n_atoms=30,  # Reducido de 50 a 30 para velocidad
            n_nonzero_coefs=3,  # Reducido de 5 a 3 para velocidad
            svm_kernel='rbf',
            multi_class=False
        )
        
        sparse_classifier.fit(X_train, y_train)
        sparse_classifier.save(str(models_dir / 'sparse_classifier.pkl'))
        
        # Evaluar
        predictions = sparse_classifier.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"✅ Precisión en test: {accuracy:.4f}")
    
    print("\n✅ Entrenamiento completado")

if __name__ == "__main__":
    main()

