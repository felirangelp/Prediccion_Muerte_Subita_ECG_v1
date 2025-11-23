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
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.utils import load_ecg_record, list_available_records
from src.analysis_data_structures import CrossValidationResults

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

def cross_validation_evaluation(model, X, y, model_name: str, fs: float = 128.0, cv_folds: int = 10):
    """
    Evaluación con validación cruzada y cálculo de intervalos de confianza
    
    Args:
        model: Modelo a evaluar (puede ser None, se entrenará en cada fold)
        X: Datos completos
        y: Etiquetas
        model_name: Nombre del modelo
        fs: Frecuencia de muestreo
        cv_folds: Número de folds para validación cruzada
    
    Returns:
        CrossValidationResults con todos los resultados
    """
    print(f"\n📊 Validación cruzada ({cv_folds}-fold) para {model_name}...")
    
    from scripts.train_models import prepare_training_data
    
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Métricas a calcular
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    scores_per_fold = {metric: [] for metric in metrics}
    
    fold = 0
    for train_idx, val_idx in kf.split(X, y):
        fold += 1
        print(f"   Fold {fold}/{cv_folds}...", end=' ')
        
        X_train_fold = [X[i] for i in train_idx]
        X_val_fold = [X[i] for i in val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Entrenar modelo en este fold
        try:
            if 'sparse' in model_name.lower():
                model_fold = SparseRepresentationClassifier(n_atoms=50, n_nonzero_coefs=5)
                model_fold.fit(X_train_fold, y_train_fold)
                predictions = model_fold.predict(X_val_fold)
                probabilities = model_fold.predict_proba(X_val_fold)
            elif 'hierarchical' in model_name.lower():
                model_fold = HierarchicalFusionClassifier(
                    tcn_filters=32, fusion_dim=64, n_classes=len(np.unique(y))
                )
                model_fold.fit(X_train_fold, y_train_fold, fs=fs, epochs=10, batch_size=8, verbose=0)
                predictions = model_fold.predict(X_val_fold, fs=fs)
                probabilities = model_fold.predict_proba(X_val_fold, fs=fs)
            elif 'hybrid' in model_name.lower():
                model_fold = HybridSCDClassifier(
                    n_atoms=50, n_nonzero_coefs=5, tcn_filters=32, fusion_dim=64
                )
                model_fold.fit(X_train_fold, y_train_fold, fs=fs, epochs_hierarchical=10, batch_size=8, verbose=0)
                predictions = model_fold.predict(X_val_fold, fs=fs)
                probabilities = model_fold.predict_proba(X_val_fold, fs=fs)
            else:
                print("⚠️  Modelo desconocido, saltando fold")
                continue
            
            # Calcular métricas
            acc = accuracy_score(y_val_fold, predictions)
            prec = precision_score(y_val_fold, predictions, average='weighted', zero_division=0)
            rec = recall_score(y_val_fold, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_val_fold, predictions, average='weighted', zero_division=0)
            
            try:
                if probabilities.shape[1] == 2:
                    auc = roc_auc_score(y_val_fold, probabilities[:, 1])
                else:
                    auc = roc_auc_score(y_val_fold, probabilities, multi_class='ovo')
            except:
                auc = 0.0
            
            scores_per_fold['accuracy'].append(acc)
            scores_per_fold['precision'].append(prec)
            scores_per_fold['recall'].append(rec)
            scores_per_fold['f1_score'].append(f1)
            scores_per_fold['auc_roc'].append(auc)
            
            print(f"Accuracy: {acc:.4f}")
            
        except Exception as e:
            print(f"⚠️  Error en fold {fold}: {e}")
            continue
    
    # Calcular estadísticas
    mean_scores = {}
    std_scores = {}
    ci_95 = {}
    
    for metric in metrics:
        if scores_per_fold[metric]:
            mean_scores[metric] = np.mean(scores_per_fold[metric])
            std_scores[metric] = np.std(scores_per_fold[metric])
            # Intervalo de confianza 95% usando t-distribution
            n = len(scores_per_fold[metric])
            t_critical = stats.t.ppf(0.975, n - 1)  # 95% CI
            margin = t_critical * std_scores[metric] / np.sqrt(n)
            ci_95[metric] = (mean_scores[metric] - margin, mean_scores[metric] + margin)
        else:
            mean_scores[metric] = 0.0
            std_scores[metric] = 0.0
            ci_95[metric] = (0.0, 0.0)
    
    print(f"   Media ± Desv. Est.:")
    for metric in metrics:
        if scores_per_fold[metric]:
            print(f"   {metric.capitalize()}: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f} "
                  f"(95% CI: [{ci_95[metric][0]:.4f}, {ci_95[metric][1]:.4f}])")
    
    return CrossValidationResults(
        model_name=model_name,
        cv_folds=cv_folds,
        scores_per_fold=scores_per_fold,
        mean_scores=mean_scores,
        std_scores=std_scores,
        ci_95=ci_95
    )

def main():
    """Función principal"""
    # Configurar GPU al inicio
    try:
        from src.config_m1 import configure_tensorflow_m1
        tf_config = configure_tensorflow_m1()
        if tf_config['gpu_available']:
            print(f"✅ GPU Metal detectada: {tf_config['gpu_device']}")
            print(f"   Backend: {tf_config['tensorflow_backend']}")
        else:
            print("⚠️  GPU no disponible, usando CPU")
    except Exception as e:
        print(f"⚠️  Error configurando GPU: {e}")
    
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
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Número de folds para validación cruzada (0 para desactivar)')
    parser.add_argument('--cross-validation-only', action='store_true',
                       help='Solo realizar validación cruzada, no evaluación estándar')
    
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
    
    # Validación cruzada si está habilitada
    cv_results_dict = {}
    if args.cv_folds > 0:
        print("\n" + "="*60)
        print("📊 VALIDACIÓN CRUZADA")
        print("="*60)
        
        if 'sparse' in models:
            cv_results = cross_validation_evaluation(
                models['sparse'], X, y, 'Sparse Representations', fs=128.0, cv_folds=args.cv_folds
            )
            cv_results_dict['sparse'] = cv_results
        
        if 'hierarchical' in models:
            cv_results = cross_validation_evaluation(
                models['hierarchical'], X, y, 'Hierarchical Fusion', fs=128.0, cv_folds=args.cv_folds
            )
            cv_results_dict['hierarchical'] = cv_results
        
        if 'hybrid' in models:
            cv_results = cross_validation_evaluation(
                models['hybrid'], X, y, 'Hybrid Model', fs=128.0, cv_folds=args.cv_folds
            )
            cv_results_dict['hybrid'] = cv_results
        
        # Guardar resultados de validación cruzada
        cv_output = Path(args.output).parent / 'cross_validation_results.pkl'
        with open(cv_output, 'wb') as f:
            pickle.dump(cv_results_dict, f)
        print(f"\n✅ Resultados de validación cruzada guardados en: {cv_output}")
    
    # Evaluación estándar (si no es solo CV)
    all_results = {}
    if not args.cross_validation_only:
        # Dividir en test
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\n📊 Datos de prueba: {len(X_test)} muestras")
        
        # Evaluar cada modelo
        if 'sparse' in models:
            results = evaluate_model(
                models['sparse'], X_test, y_test, 'Sparse Representations', fs=128.0
            )
            if results:
                all_results['sparse'] = results
                # Agregar intervalos de confianza si hay CV
                if 'sparse' in cv_results_dict:
                    cv_res = cv_results_dict['sparse']
                    all_results['sparse']['accuracy_ci'] = cv_res.ci_95.get('accuracy', (0, 0))
                    all_results['sparse']['precision_ci'] = cv_res.ci_95.get('precision', (0, 0))
                    all_results['sparse']['recall_ci'] = cv_res.ci_95.get('recall', (0, 0))
                    all_results['sparse']['f1_score_ci'] = cv_res.ci_95.get('f1_score', (0, 0))
                    all_results['sparse']['auc_roc_ci'] = cv_res.ci_95.get('auc_roc', (0, 0))
                    all_results['sparse']['scores_per_fold'] = cv_res.scores_per_fold
    
        if 'hierarchical' in models:
            results = evaluate_model(
                models['hierarchical'], X_test, y_test, 'Hierarchical Fusion', fs=128.0
            )
            if results:
                all_results['hierarchical'] = results
                # Agregar intervalos de confianza si hay CV
                if 'hierarchical' in cv_results_dict:
                    cv_res = cv_results_dict['hierarchical']
                    all_results['hierarchical']['accuracy_ci'] = cv_res.ci_95.get('accuracy', (0, 0))
                    all_results['hierarchical']['precision_ci'] = cv_res.ci_95.get('precision', (0, 0))
                    all_results['hierarchical']['recall_ci'] = cv_res.ci_95.get('recall', (0, 0))
                    all_results['hierarchical']['f1_score_ci'] = cv_res.ci_95.get('f1_score', (0, 0))
                    all_results['hierarchical']['auc_roc_ci'] = cv_res.ci_95.get('auc_roc', (0, 0))
                    all_results['hierarchical']['scores_per_fold'] = cv_res.scores_per_fold
        
        if 'hybrid' in models:
            results = evaluate_model(
                models['hybrid'], X_test, y_test, 'Hybrid Model', fs=128.0
            )
            if results:
                all_results['hybrid'] = results
                # Agregar intervalos de confianza si hay CV
                if 'hybrid' in cv_results_dict:
                    cv_res = cv_results_dict['hybrid']
                    all_results['hybrid']['accuracy_ci'] = cv_res.ci_95.get('accuracy', (0, 0))
                    all_results['hybrid']['precision_ci'] = cv_res.ci_95.get('precision', (0, 0))
                    all_results['hybrid']['recall_ci'] = cv_res.ci_95.get('recall', (0, 0))
                    all_results['hybrid']['f1_score_ci'] = cv_res.ci_95.get('f1_score', (0, 0))
                    all_results['hybrid']['auc_roc_ci'] = cv_res.ci_95.get('auc_roc', (0, 0))
                    all_results['hybrid']['scores_per_fold'] = cv_res.scores_per_fold
    
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

