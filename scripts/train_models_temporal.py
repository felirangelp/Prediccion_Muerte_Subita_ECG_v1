"""
Script para entrenar modelos por intervalo temporal pre-SCD
Entrena modelos separados para cada intervalo temporal (5, 10, 15, 20, 25, 30 min)
o un modelo multi-clase que distingue entre intervalos
"""

import numpy as np
import sys
from pathlib import Path
import pickle
import argparse
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.analysis_data_structures import TemporalAnalysisResults, TemporalIntervalResult


def prepare_temporal_training_data(temporal_data: Dict, 
                                   scheme: str = 'sensors',
                                   binary_classification: bool = False) -> tuple:
    """
    Preparar datos para entrenamiento por intervalos temporales
    
    Args:
        temporal_data: Datos extraídos por analyze_temporal_intervals.py
        scheme: 'sensors' o 'symmetry'
        binary_classification: Si True, clasificación binaria (SCD vs Normal)
                              Si False, multi-clase por intervalos
    
    Returns:
        X, y preparados para entrenamiento
    """
    all_segments = []
    all_labels = []
    
    # Procesar datos SDDB (SCD)
    sddb_segments = temporal_data['sddb']['segments']
    sddb_labels = temporal_data['sddb']['labels']
    
    # Procesar datos NSRDB (Normal)
    nsrdb_segments = temporal_data['nsrdb']['segments']
    nsrdb_labels = temporal_data['nsrdb']['labels']
    
    if binary_classification:
        # Clasificación binaria: SCD (1) vs Normal (0)
        for segment in sddb_segments:
            all_segments.append(segment)
            all_labels.append(1)  # SCD
        
        for segment in nsrdb_segments:
            all_segments.append(segment)
            all_labels.append(0)  # Normal
    else:
        # Clasificación multi-clase: Normal, 5min, 10min, 15min, 20min, 25min, 30min
        # Mapear intervalos a clases numéricas
        if scheme == 'sensors':
            interval_to_class = {
                'Normal': 0,
                5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6
            }
        else:  # symmetry
            # Para symmetry, usar promedio del intervalo como clase
            interval_to_class = {
                'Normal': 0,
                5: 1, 15: 2, 25: 3, 35: 4, 45: 5, 55: 6  # Promedios de intervalos
            }
        
        for segment, label in zip(sddb_segments, sddb_labels):
            all_segments.append(segment)
            if isinstance(label, (int, float)):
                # Redondear a intervalo más cercano
                if scheme == 'sensors':
                    class_label = interval_to_class.get(int(label), 1)
                else:
                    class_label = interval_to_class.get(int(label), 1)
            else:
                class_label = interval_to_class.get(label, 1)
            all_labels.append(class_label)
        
        for segment, label in zip(nsrdb_segments, nsrdb_labels):
            all_segments.append(segment)
            all_labels.append(0)  # Normal siempre es clase 0
    
    return all_segments, np.array(all_labels)


def train_model_by_interval(model_type: str,
                            X_train: List[np.ndarray],
                            y_train: np.ndarray,
                            X_test: List[np.ndarray],
                            y_test: np.ndarray,
                            interval_label: Optional[str] = None,
                            fs: float = 128.0) -> Dict:
    """
    Entrenar un modelo para un intervalo específico
    
    Args:
        model_type: 'sparse', 'hierarchical', o 'hybrid'
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        interval_label: Etiqueta del intervalo (para logging)
        fs: Frecuencia de muestreo
    
    Returns:
        Diccionario con resultados del entrenamiento
    """
    print(f"\n🔄 Entrenando modelo {model_type} para intervalo {interval_label}...")
    
    if model_type == 'sparse':
        model = SparseRepresentationClassifier(
            n_atoms=30,
            n_nonzero_coefs=3,
            svm_kernel='rbf',
            multi_class=(len(np.unique(y_train)) > 2)
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
    
    elif model_type == 'hierarchical':
        model = HierarchicalFusionClassifier(
            tcn_filters=32,
            fusion_dim=64,
            n_classes=len(np.unique(y_train))
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
    
    # AUC-ROC
    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            auc = roc_auc_score(y_test, probabilities, multi_class='ovo')
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_test, predictions)
    
    results = {
        'model': model,
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
    
    print(f"   ✅ Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return results


def train_models_by_interval(temporal_data: Dict,
                             model_types: List[str] = ['sparse', 'hierarchical', 'hybrid'],
                             scheme: str = 'sensors',
                             binary_classification: bool = False,
                             test_size: float = 0.2,
                             min_test_samples: int = 10,
                             use_cross_validation: bool = False) -> TemporalAnalysisResults:
    """
    Entrenar modelos para cada intervalo temporal
    
    Args:
        temporal_data: Datos extraídos por analyze_temporal_intervals.py
        model_types: Lista de tipos de modelos a entrenar
        scheme: 'sensors' o 'symmetry'
        binary_classification: Si True, entrenar modelos binarios por intervalo
                              Si False, entrenar un modelo multi-clase
        test_size: Proporción de datos para prueba
    
    Returns:
        TemporalAnalysisResults con todos los resultados
    """
    print("=" * 70)
    print("🎯 ENTRENAMIENTO DE MODELOS POR INTERVALO TEMPORAL")
    print("=" * 70)
    print(f"Esquema: {scheme}")
    print(f"Clasificación: {'Binaria por intervalo' if binary_classification else 'Multi-clase'}")
    print()
    
    # Preparar datos
    X, y = prepare_temporal_training_data(temporal_data, scheme, binary_classification)
    
    print(f"📊 Total de muestras: {len(X)}")
    print(f"   Clases: {np.unique(y)}")
    print()
    
    if binary_classification:
        # Entrenar modelo separado para cada intervalo
        intervals = [5, 10, 15, 20, 25, 30] if scheme == 'sensors' else [(0,10), (10,20), (20,30), (30,40), (40,50), (50,60)]
        temporal_results = TemporalAnalysisResults(intervals=intervals)
        
        # Filtrar datos por intervalo y entrenar
        sddb_data = temporal_data['sddb']
        nsrdb_data = temporal_data['nsrdb']
        
        for interval in intervals:
            print(f"\n{'='*70}")
            print(f"📅 Procesando intervalo: {interval} minutos antes de SCD")
            print(f"{'='*70}")
            
            # Filtrar segmentos de este intervalo
            interval_segments = []
            interval_labels = []
            
            # Añadir segmentos SCD de este intervalo
            for seg, label, meta in zip(sddb_data['segments'], sddb_data['labels'], sddb_data['metadata']):
                if meta['interval'] == interval:
                    interval_segments.append(seg)
                    interval_labels.append(1)  # SCD
            
            # Añadir segmentos Normal
            for seg in nsrdb_data['segments']:
                interval_segments.append(seg)
                interval_labels.append(0)  # Normal
            
            if len(interval_segments) == 0:
                print(f"⚠️  No hay datos para intervalo {interval}, omitiendo...")
                continue
            
            # Dividir datos
            X_interval = interval_segments
            y_interval = np.array(interval_labels)
            
            # Verificar distribución de clases
            unique_classes, class_counts = np.unique(y_interval, return_counts=True)
            print(f"   Distribución de clases: {dict(zip(unique_classes, class_counts))}")
            
            # Ajustar test_size dinámicamente para asegurar suficientes muestras en test
            min_test_per_class = 3  # Mínimo de muestras por clase en test
            
            # Calcular test_size adaptativo
            total_samples = len(y_interval)
            min_samples_needed = min_test_samples
            for count in class_counts:
                min_samples_needed = max(min_samples_needed, min_test_per_class * 2)  # Asegurar balance
            
            if total_samples < min_samples_needed:
                print(f"   ⚠️  ADVERTENCIA: Solo {total_samples} muestras disponibles (mínimo recomendado: {min_samples_needed})")
                print(f"   ⚠️  Los resultados pueden no ser confiables")
                # Usar un test_size más pequeño pero asegurar al menos 2 muestras por clase
                adaptive_test_size = max(0.1, min_test_per_class * len(unique_classes) / total_samples)
                adaptive_test_size = min(adaptive_test_size, 0.3)  # No más del 30%
            else:
                # Calcular test_size que asegure suficientes muestras
                adaptive_test_size = max(test_size, min_samples_needed / total_samples)
                adaptive_test_size = min(adaptive_test_size, 0.3)  # No más del 30%
            
            # Verificar que stratify sea posible
            can_stratify = all(count >= 2 for count in class_counts) and len(unique_classes) > 1
            
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_interval, y_interval, 
                    test_size=adaptive_test_size, 
                    random_state=42, 
                    stratify=y_interval
                )
            else:
                print(f"   ⚠️  No se puede usar stratify (muy pocas muestras por clase), usando división aleatoria")
            X_train, X_test, y_train, y_test = train_test_split(
                    X_interval, y_interval, 
                    test_size=adaptive_test_size, 
                    random_state=42
                )
            
            # Verificar distribución final
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            test_classes, test_counts = np.unique(y_test, return_counts=True)
            
            print(f"   Entrenamiento: {len(X_train)} muestras (distribución: {dict(zip(train_classes, train_counts))})")
            print(f"   Prueba: {len(X_test)} muestras (distribución: {dict(zip(test_classes, test_counts))})")
            
            # Advertencia si hay muy pocas muestras en test
            if len(X_test) < min_test_samples:
                print(f"   ⚠️  ADVERTENCIA: Solo {len(X_test)} muestras en test (mínimo recomendado: {min_test_samples})")
            if any(count < min_test_per_class for count in test_counts):
                print(f"   ⚠️  ADVERTENCIA: Algunas clases tienen menos de {min_test_per_class} muestras en test")
            
            # Entrenar cada tipo de modelo
            for model_type in model_types:
                try:
                    # Verificar balance de clases antes de entrenar sparse
                    if model_type == 'sparse':
                        unique_classes_train, counts = np.unique(y_train, return_counts=True)
                        min_class_count = min(counts)
                        if min_class_count < 10:
                            print(f"   ⚠️  Omitiendo modelo sparse para intervalo {interval}: clase minoritaria tiene solo {min_class_count} muestras")
                            continue
                    
                    if use_cross_validation and len(X_interval) >= 20:
                        # Usar validación cruzada para obtener resultados más confiables
                        print(f"   🔄 Usando validación cruzada (5-fold) para {model_type}...")
                        n_folds = min(5, min(class_counts) // 2)  # Asegurar al menos 2 muestras por clase por fold
                        n_folds = max(3, n_folds)  # Mínimo 3 folds
                        
                        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                        fold_accuracies = []
                        fold_precisions = []
                        fold_recalls = []
                        fold_f1s = []
                        fold_aucs = []
                        all_test_predictions = []
                        all_test_probs = []
                        all_test_labels = []
                        
                        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_interval, y_interval)):
                            X_train_fold = [X_interval[i] for i in train_idx]
                            X_test_fold = [X_interval[i] for i in test_idx]
                            y_train_fold = y_interval[train_idx]
                            y_test_fold = y_interval[test_idx]
                            
                            fold_results = train_model_by_interval(
                                model_type, X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                                interval_label=f"{interval}_fold{fold_idx}", fs=128.0
                            )
                            
                            fold_accuracies.append(fold_results['accuracy'])
                            fold_precisions.append(fold_results['precision'])
                            fold_recalls.append(fold_results['recall'])
                            fold_f1s.append(fold_results['f1_score'])
                            fold_aucs.append(fold_results['auc_roc'])
                            all_test_predictions.extend(fold_results['predictions'])
                            all_test_probs.extend(fold_results['probabilities'].tolist() if hasattr(fold_results['probabilities'], 'tolist') else fold_results['probabilities'])
                            all_test_labels.extend(y_test_fold)
                        
                        # Promediar resultados de todos los folds
                        avg_accuracy = np.mean(fold_accuracies)
                        avg_precision = np.mean(fold_precisions)
                        avg_recall = np.mean(fold_recalls)
                        avg_f1 = np.mean(fold_f1s)
                        avg_auc = np.mean(fold_aucs)
                        
                        # Calcular métricas agregadas sobre todas las predicciones
                        all_test_predictions = np.array(all_test_predictions)
                        all_test_labels = np.array(all_test_labels)
                        aggregated_accuracy = accuracy_score(all_test_labels, all_test_predictions)
                        
                        print(f"   ✅ CV Results - Accuracy: {avg_accuracy:.4f} ± {np.std(fold_accuracies):.4f}")
                        print(f"      (Agregado: {aggregated_accuracy:.4f})")
                        
                        # Crear matriz de confusión agregada
                        aggregated_cm = confusion_matrix(all_test_labels, all_test_predictions)
                        
                        interval_result = TemporalIntervalResult(
                            interval_minutes=interval if isinstance(interval, int) else interval[0],
                            accuracy=aggregated_accuracy,  # Usar accuracy agregado
                            precision=avg_precision,
                            recall=avg_recall,
                            f1_score=avg_f1,
                            auc_roc=avg_auc,
                            confusion_matrix=aggregated_cm,
                            predictions=all_test_predictions,
                            probabilities=np.array(all_test_probs),
                            n_samples=len(all_test_labels)
                        )
                    else:
                        # División simple train/test
                    results = train_model_by_interval(
                        model_type, X_train, y_train, X_test, y_test,
                        interval_label=str(interval), fs=128.0
                    )
                    
                    # Crear TemporalIntervalResult
                    interval_result = TemporalIntervalResult(
                        interval_minutes=interval if isinstance(interval, int) else interval[0],
                        accuracy=results['accuracy'],
                        precision=results['precision'],
                        recall=results['recall'],
                        f1_score=results['f1_score'],
                        auc_roc=results['auc_roc'],
                        confusion_matrix=results['confusion_matrix'],
                        predictions=results['predictions'],
                        probabilities=results['probabilities'],
                        n_samples=results['n_samples']
                    )
                    
                    temporal_results.add_result(model_type, interval, interval_result)
                    
                except Exception as e:
                    print(f"   ⚠️  Error entrenando {model_type} para intervalo {interval}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    else:
        # Entrenar modelo multi-clase único
        print("\n🔄 Entrenando modelo multi-clase único...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Entrenamiento: {len(X_train)} muestras")
        print(f"   Prueba: {len(X_test)} muestras")
        
        intervals = [0, 5, 10, 15, 20, 25, 30]  # Incluir Normal (0)
        temporal_results = TemporalAnalysisResults(intervals=intervals)
        
        for model_type in model_types:
            try:
                results = train_model_by_interval(
                    model_type, X_train, y_train, X_test, y_test,
                    interval_label='multi-class', fs=128.0
                )
                
                # Para multi-clase, guardar resultados para cada clase
                for class_label in np.unique(y_test):
                    if class_label == 0:
                        continue  # Normal se maneja por separado
                    
                    # Filtrar resultados para esta clase
                    class_mask = (y_test == class_label)
                    class_accuracy = accuracy_score(
                        y_test[class_mask], 
                        results['predictions'][class_mask]
                    )
                    
                    interval_result = TemporalIntervalResult(
                        interval_minutes=int(class_label),
                        accuracy=class_accuracy,
                        precision=results['precision'],
                        recall=results['recall'],
                        f1_score=results['f1_score'],
                        auc_roc=results['auc_roc'],
                        confusion_matrix=results['confusion_matrix'],
                        predictions=results['predictions'],
                        probabilities=results['probabilities'],
                        n_samples=np.sum(class_mask)
                    )
                    
                    temporal_results.add_result(model_type, int(class_label), interval_result)
                    
            except Exception as e:
                print(f"   ⚠️  Error entrenando {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return temporal_results


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenar modelos por intervalo temporal')
    parser.add_argument('--temporal-data', type=str, 
                       default='results/temporal_intervals_data.pkl',
                       help='Archivo con datos temporales extraídos')
    parser.add_argument('--output', type=str,
                       default='results/temporal_results.pkl',
                       help='Archivo de salida para resultados')
    parser.add_argument('--scheme', type=str, choices=['sensors', 'symmetry'],
                       default='sensors',
                       help='Esquema de intervalos')
    parser.add_argument('--binary', action='store_true',
                       help='Usar clasificación binaria por intervalo (en lugar de multi-clase)')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['sparse', 'hierarchical', 'hybrid'],
                       default=['sparse', 'hierarchical', 'hybrid'],
                       help='Modelos a entrenar')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporción de datos para prueba (se ajusta automáticamente si hay pocas muestras)')
    parser.add_argument('--min-test-samples', type=int, default=10,
                       help='Mínimo de muestras requeridas en conjunto de prueba')
    parser.add_argument('--use-cross-validation', action='store_true',
                       help='Usar validación cruzada en lugar de división simple (recomendado para pocas muestras)')
    
    args = parser.parse_args()
    
    # Cargar datos temporales
    temporal_data_path = Path(args.temporal_data)
    if not temporal_data_path.exists():
        print(f"❌ Archivo de datos temporales no encontrado: {args.temporal_data}")
        print("   Ejecuta primero: python scripts/analyze_temporal_intervals.py")
        return
    
    print(f"📂 Cargando datos temporales desde: {args.temporal_data}")
    with open(temporal_data_path, 'rb') as f:
        temporal_data = pickle.load(f)
    
    # Entrenar modelos
    temporal_results = train_models_by_interval(
        temporal_data,
        model_types=args.models,
        scheme=args.scheme,
        binary_classification=args.binary,
        test_size=args.test_size,
        min_test_samples=args.min_test_samples,
        use_cross_validation=args.use_cross_validation
    )
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporal_results.save(str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"📁 Resultados guardados en: {args.output}")
    print()
    print("📊 Resumen de resultados:")
    for model_name in args.models:
        if model_name in temporal_results.results_by_model:
            print(f"\n  {model_name.upper()}:")
            for interval, result in temporal_results.results_by_model[model_name].items():
                print(f"    {interval} min: Accuracy={result.accuracy:.4f}, AUC={result.auc_roc:.4f}")


if __name__ == "__main__":
    main()

