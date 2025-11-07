"""
Script para validación inter-paciente
Divide datos por paciente (no por muestras) para validación más realista
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, List, Set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.analysis_data_structures import InterPatientValidationResults, InterPatientSplit

def load_temporal_data(temporal_data_file: str = 'results/temporal_intervals_data.pkl') -> Dict:
    """Cargar datos temporales"""
    with open(temporal_data_file, 'rb') as f:
        return pickle.load(f)

def split_by_patient(temporal_data: Dict, test_patients_ratio: float = 0.3) -> List[InterPatientSplit]:
    """
    Dividir datos por paciente (no por muestras)
    Retorna lista de splits para validación cruzada
    """
    # Obtener pacientes únicos de cada dataset
    sddb_metadata = temporal_data['sddb'].get('metadata', [])
    nsrdb_metadata = temporal_data['nsrdb'].get('metadata', [])
    
    # Extraer nombres de pacientes únicos
    sddb_patients = set()
    for meta in sddb_metadata:
        if 'record' in meta:
            sddb_patients.add(meta['record'])
    
    nsrdb_patients = set()
    for meta in nsrdb_metadata:
        if 'record' in meta:
            nsrdb_patients.add(meta['record'])
    
    # Convertir nombres de pacientes a números de registro
    sddb_patients_list = sorted(list(sddb_patients))
    nsrdb_patients_list = sorted(list(nsrdb_patients))
    
    # Mapear nombres a números (simplificado: usar índice)
    sddb_records = [i for i in range(len(sddb_patients_list))]
    nsrdb_records = [i + 100 for i in range(len(nsrdb_patients_list))]  # Offset para distinguir
    
    # Dividir pacientes en train/test
    n_test_sddb = max(1, int(len(sddb_records) * test_patients_ratio))
    n_test_nsrdb = max(1, int(len(nsrdb_records) * test_patients_ratio))
    
    test_sddb_records = sddb_records[:n_test_sddb]
    test_nsrdb_records = nsrdb_records[:n_test_nsrdb]
    
    train_sddb_records = sddb_records[n_test_sddb:]
    train_nsrdb_records = nsrdb_records[n_test_nsrdb:]
    
    # Crear split único (fold 0)
    split = InterPatientSplit(
        fold_id=0,
        train_records=train_sddb_records + train_nsrdb_records,
        test_records=test_sddb_records + test_nsrdb_records,
        n_train=len(train_sddb_records) + len(train_nsrdb_records),
        n_test=len(test_sddb_records) + len(test_nsrdb_records)
    )
    
    return [split]

def prepare_data_by_patient(temporal_data: Dict, split: InterPatientSplit) -> tuple:
    """
    Preparar datos según división por paciente
    (Esta función ya no se usa, pero se mantiene por compatibilidad)
    """
    pass

def train_inter_patient_model(model_type: str, X_train: List[np.ndarray], y_train: np.ndarray,
                              X_test: List[np.ndarray], y_test: np.ndarray, fs: float = 128.0) -> Dict:
    """Entrenar modelo con validación inter-paciente"""
    print(f"\n🔄 Entrenando modelo {model_type} (validación inter-paciente)...")
    
    if model_type == 'sparse':
        model = SparseRepresentationClassifier(
            n_atoms=30,
            n_nonzero_coefs=3,
            svm_kernel='rbf',
            multi_class=False
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
    
    elif model_type == 'hierarchical':
        model = HierarchicalFusionClassifier(
            tcn_filters=32,
            fusion_dim=64,
            n_classes=2
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
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            auc = 0.0
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
        'n_samples': len(y_test)
    }

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validación inter-paciente')
    parser.add_argument('--temporal-data', type=str, default='results/temporal_intervals_data.pkl',
                       help='Archivo con datos temporales')
    parser.add_argument('--output', type=str, default='results/inter_patient_results.pkl',
                       help='Archivo de salida')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['sparse', 'hierarchical', 'hybrid'],
                       default=['hierarchical'],
                       help='Modelos a entrenar')
    parser.add_argument('--test-ratio', type=float, default=0.3,
                       help='Proporción de pacientes para prueba')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("👥 VALIDACIÓN INTER-PACIENTE")
    print("=" * 70)
    
    # Cargar datos temporales
    print(f"\n📂 Cargando datos desde: {args.temporal_data}")
    temporal_data = load_temporal_data(args.temporal_data)
    
    # Dividir por paciente
    print(f"\n🔀 Dividiendo datos por paciente (test ratio: {args.test_ratio})")
    splits = split_by_patient(temporal_data, test_patients_ratio=args.test_ratio)
    split = splits[0]  # Usar el primer split
    
    print(f"   Train records: {len(split.train_records)}")
    print(f"   Test records: {len(split.test_records)}")
    
    # Preparar datos basándose en los records
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Procesar SDDB
    sddb_segments = temporal_data['sddb']['segments']
    sddb_labels = temporal_data['sddb']['labels']
    sddb_metadata = temporal_data['sddb'].get('metadata', [])
    
    # Mapear nombres de pacientes a números de registro
    sddb_patients_list = sorted(list(set(meta.get('record', 'unknown') for meta in sddb_metadata)))
    patient_to_record = {patient: i for i, patient in enumerate(sddb_patients_list)}
    
    for seg, label, meta in zip(sddb_segments, sddb_labels, sddb_metadata):
        patient = meta.get('record', 'unknown')
        record_id = patient_to_record.get(patient, 0)
        if record_id in split.train_records:
            X_train.append(seg)
            y_train.append(1)  # SCD
        elif record_id in split.test_records:
            X_test.append(seg)
            y_test.append(1)  # SCD
    
    # Procesar NSRDB
    nsrdb_segments = temporal_data['nsrdb']['segments']
    nsrdb_labels = temporal_data['nsrdb']['labels']
    nsrdb_metadata = temporal_data['nsrdb'].get('metadata', [])
    
    nsrdb_patients_list = sorted(list(set(meta.get('record', 'unknown') for meta in nsrdb_metadata)))
    patient_to_record_nsrdb = {patient: i + 100 for i, patient in enumerate(nsrdb_patients_list)}
    
    for seg, label, meta in zip(nsrdb_segments, nsrdb_labels, nsrdb_metadata):
        patient = meta.get('record', 'unknown')
        record_id = patient_to_record_nsrdb.get(patient, 100)
        if record_id in split.train_records:
            X_train.append(seg)
            y_train.append(0)  # Normal
        elif record_id in split.test_records:
            X_test.append(seg)
            y_test.append(0)  # Normal
    
    X_train = np.array(X_train) if isinstance(X_train[0], np.ndarray) else X_train
    y_train = np.array(y_train)
    X_test = np.array(X_test) if isinstance(X_test[0], np.ndarray) else X_test
    y_test = np.array(y_test)
    
    print(f"\n📊 Datos preparados:")
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    print(f"   Clases train: {np.bincount(y_train)}")
    print(f"   Clases test: {np.bincount(y_test)}")
    
    # Crear objeto de resultados
    inter_patient_results = InterPatientValidationResults(splits=splits)
    
    # Entrenar modelos
    for model_type in args.models:
        metrics = train_inter_patient_model(
            model_type, X_train, y_train, X_test, y_test, fs=128.0
        )
        
        inter_patient_results.add_fold_result(0, model_type, {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc']
        })
        
        print(f"\n✅ {model_type}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc_roc']:.4f}")
    
    # Calcular promedios
    inter_patient_results.calculate_averages()
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inter_patient_results.save(str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ VALIDACIÓN INTER-PACIENTE COMPLETADA")
    print("=" * 70)
    print(f"📁 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()

