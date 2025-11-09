"""
Script para optimización de hiperparámetros usando Grid Search o Random Search
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from itertools import product
from tqdm import tqdm
import random

sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.analysis_data_structures import HyperparameterSearchResults
from scripts.train_models import load_dataset, prepare_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def optimize_sparse_hyperparameters(X_train, y_train, X_val, y_val, search_type='grid', max_combinations=50):
    """
    Optimizar hiperparámetros del modelo Sparse
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_val: Datos de validación
        y_val: Etiquetas de validación
        search_type: 'grid' o 'random'
        max_combinations: Máximo de combinaciones a probar (para random search)
    """
    print("\n🔍 Optimizando hiperparámetros para Sparse Representations...")
    
    # Grid de parámetros
    param_grid = {
        'n_atoms': [20, 30, 50, 70],
        'n_nonzero_coefs': [3, 5, 7, 10],
        'svm_kernel': ['rbf', 'linear', 'poly']
    }
    
    # Generar combinaciones
    if search_type == 'grid':
        combinations = list(product(
            param_grid['n_atoms'],
            param_grid['n_nonzero_coefs'],
            param_grid['svm_kernel']
        ))
    else:  # random search
        combinations = []
        for _ in range(min(max_combinations, len(list(product(*param_grid.values()))))):
            combinations.append((
                random.choice(param_grid['n_atoms']),
                random.choice(param_grid['n_nonzero_coefs']),
                random.choice(param_grid['svm_kernel'])
            ))
    
    best_score = 0.0
    best_params = None
    search_results = []
    
    print(f"   Probando {len(combinations)} combinaciones...")
    
    for n_atoms, n_nonzero_coefs, svm_kernel in tqdm(combinations):
        try:
            model = SparseRepresentationClassifier(
                n_atoms=n_atoms,
                n_nonzero_coefs=n_nonzero_coefs,
                svm_kernel=svm_kernel
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = accuracy_score(y_val, predictions)
            
            search_results.append({
                'n_atoms': n_atoms,
                'n_nonzero_coefs': n_nonzero_coefs,
                'svm_kernel': svm_kernel,
                'accuracy': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {
                    'n_atoms': n_atoms,
                    'n_nonzero_coefs': n_nonzero_coefs,
                    'svm_kernel': svm_kernel
                }
        except Exception as e:
            print(f"   ⚠️  Error con parámetros {n_atoms}, {n_nonzero_coefs}, {svm_kernel}: {e}")
            continue
    
    print(f"   ✅ Mejor accuracy: {best_score:.4f}")
    print(f"   ✅ Mejores parámetros: {best_params}")
    
    return HyperparameterSearchResults(
        model_name='sparse',
        best_params=best_params,
        best_score=best_score,
        search_results=search_results,
        param_grid=param_grid
    )


def optimize_hierarchical_hyperparameters(X_train, y_train, X_val, y_val, fs=128.0, search_type='grid', max_combinations=30):
    """
    Optimizar hiperparámetros del modelo Hierarchical
    """
    print("\n🔍 Optimizando hiperparámetros para Hierarchical Fusion...")
    
    param_grid = {
        'tcn_filters': [16, 32, 64],
        'fusion_dim': [32, 64, 128],
        'epochs': [20, 30, 50],
        'batch_size': [4, 8, 16]
    }
    
    if search_type == 'grid':
        # Grid completo es muy grande, usar subset
        combinations = [
            (16, 32, 20, 4), (16, 64, 20, 8), (32, 32, 20, 4),
            (32, 64, 30, 8), (32, 128, 30, 8), (64, 64, 30, 16),
            (64, 128, 50, 16)
        ]
    else:  # random search
        combinations = []
        for _ in range(max_combinations):
            combinations.append((
                random.choice(param_grid['tcn_filters']),
                random.choice(param_grid['fusion_dim']),
                random.choice(param_grid['epochs']),
                random.choice(param_grid['batch_size'])
            ))
    
    best_score = 0.0
    best_params = None
    search_results = []
    
    print(f"   Probando {len(combinations)} combinaciones...")
    
    for tcn_filters, fusion_dim, epochs, batch_size in tqdm(combinations):
        try:
            model = HierarchicalFusionClassifier(
                tcn_filters=tcn_filters,
                fusion_dim=fusion_dim,
                n_classes=len(np.unique(y_train))
            )
            model.fit(X_train, y_train, fs=fs, epochs=epochs, batch_size=batch_size, verbose=0)
            predictions = model.predict(X_val, fs=fs)
            score = accuracy_score(y_val, predictions)
            
            search_results.append({
                'tcn_filters': tcn_filters,
                'fusion_dim': fusion_dim,
                'epochs': epochs,
                'batch_size': batch_size,
                'accuracy': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {
                    'tcn_filters': tcn_filters,
                    'fusion_dim': fusion_dim,
                    'epochs': epochs,
                    'batch_size': batch_size
                }
        except Exception as e:
            print(f"   ⚠️  Error con parámetros {tcn_filters}, {fusion_dim}, {epochs}, {batch_size}: {e}")
            continue
    
    print(f"   ✅ Mejor accuracy: {best_score:.4f}")
    print(f"   ✅ Mejores parámetros: {best_params}")
    
    return HyperparameterSearchResults(
        model_name='hierarchical',
        best_params=best_params,
        best_score=best_score,
        search_results=search_results,
        param_grid=param_grid
    )


def optimize_hybrid_hyperparameters(X_train, y_train, X_val, y_val, fs=128.0, search_type='grid', max_combinations=20):
    """
    Optimizar hiperparámetros del modelo Hybrid
    """
    print("\n🔍 Optimizando hiperparámetros para Hybrid Model...")
    
    param_grid = {
        'n_atoms': [30, 50, 70],
        'n_nonzero_coefs': [3, 5, 7],
        'tcn_filters': [32, 64],
        'fusion_dim': [64, 128]
    }
    
    if search_type == 'grid':
        combinations = list(product(
            param_grid['n_atoms'],
            param_grid['n_nonzero_coefs'],
            param_grid['tcn_filters'],
            param_grid['fusion_dim']
        ))
        # Limitar a máximo 20 combinaciones
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
    else:  # random search
        combinations = []
        for _ in range(max_combinations):
            combinations.append((
                random.choice(param_grid['n_atoms']),
                random.choice(param_grid['n_nonzero_coefs']),
                random.choice(param_grid['tcn_filters']),
                random.choice(param_grid['fusion_dim'])
            ))
    
    best_score = 0.0
    best_params = None
    search_results = []
    
    print(f"   Probando {len(combinations)} combinaciones...")
    
    for n_atoms, n_nonzero_coefs, tcn_filters, fusion_dim in tqdm(combinations):
        try:
            model = HybridSCDClassifier(
                n_atoms=n_atoms,
                n_nonzero_coefs=n_nonzero_coefs,
                tcn_filters=tcn_filters,
                fusion_dim=fusion_dim
            )
            model.fit(X_train, y_train, fs=fs, epochs_hierarchical=20, batch_size=8, verbose=0)
            predictions = model.predict(X_val, fs=fs)
            score = accuracy_score(y_val, predictions)
            
            search_results.append({
                'n_atoms': n_atoms,
                'n_nonzero_coefs': n_nonzero_coefs,
                'tcn_filters': tcn_filters,
                'fusion_dim': fusion_dim,
                'accuracy': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {
                    'n_atoms': n_atoms,
                    'n_nonzero_coefs': n_nonzero_coefs,
                    'tcn_filters': tcn_filters,
                    'fusion_dim': fusion_dim
                }
        except Exception as e:
            print(f"   ⚠️  Error con parámetros: {e}")
            continue
    
    print(f"   ✅ Mejor accuracy: {best_score:.4f}")
    print(f"   ✅ Mejores parámetros: {best_params}")
    
    return HyperparameterSearchResults(
        model_name='hybrid',
        best_params=best_params,
        best_score=best_score,
        search_results=search_results,
        param_grid=param_grid
    )


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimizar hiperparámetros de modelos')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--output', type=str, default='results/hyperparameter_search_results.pkl',
                       help='Archivo de salida con resultados')
    parser.add_argument('--best-params-output', type=str, default='results/best_hyperparameters.pkl',
                       help='Archivo de salida con mejores parámetros')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    parser.add_argument('--search-type', type=str, default='random', choices=['grid', 'random'],
                       help='Tipo de búsqueda: grid o random')
    parser.add_argument('--max-combinations', type=int, default=30,
                       help='Máximo de combinaciones a probar (para random search)')
    parser.add_argument('--model', type=str, default='all', choices=['sparse', 'hierarchical', 'hybrid', 'all'],
                       help='Modelo a optimizar')
    
    args = parser.parse_args()
    
    # Crear directorio de resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # Cargar datos
    print("📂 Cargando datos...")
    sddb_path = Path(args.data_dir) / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = Path(args.data_dir) / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'
    
    sddb_signals, sddb_labels, _ = load_dataset(str(sddb_path), 'sddb', args.max_records)
    nsrdb_signals, nsrdb_labels, _ = load_dataset(str(nsrdb_path), 'nsrdb', args.max_records)
    
    all_signals = sddb_signals + nsrdb_signals
    all_labels = np.concatenate([sddb_labels, nsrdb_labels])
    
    X, y = prepare_training_data(all_signals, all_labels, fs=128.0, window_size=30.0)
    
    # Dividir en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Datos: {len(X_train)} entrenamiento, {len(X_val)} validación")
    
    # Optimizar modelos
    all_results = {}
    best_params_dict = {}
    
    if args.model in ['sparse', 'all']:
        result = optimize_sparse_hyperparameters(
            X_train, y_train, X_val, y_val,
            search_type=args.search_type,
            max_combinations=args.max_combinations
        )
        all_results['sparse'] = result
        best_params_dict['sparse'] = result.best_params
    
    if args.model in ['hierarchical', 'all']:
        result = optimize_hierarchical_hyperparameters(
            X_train, y_train, X_val, y_val, fs=128.0,
            search_type=args.search_type,
            max_combinations=args.max_combinations
        )
        all_results['hierarchical'] = result
        best_params_dict['hierarchical'] = result.best_params
    
    if args.model in ['hybrid', 'all']:
        result = optimize_hybrid_hyperparameters(
            X_train, y_train, X_val, y_val, fs=128.0,
            search_type=args.search_type,
            max_combinations=args.max_combinations
        )
        all_results['hybrid'] = result
        best_params_dict['hybrid'] = result.best_params
    
    # Guardar resultados
    with open(args.output, 'wb') as f:
        pickle.dump(all_results, f)
    
    with open(args.best_params_output, 'wb') as f:
        pickle.dump(best_params_dict, f)
    
    print(f"\n✅ Resultados guardados en: {args.output}")
    print(f"✅ Mejores parámetros guardados en: {args.best_params_output}")
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE OPTIMIZACIÓN")
    print("="*60)
    for model_name, result in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"   Mejor accuracy: {result.best_score:.4f}")
        print(f"   Mejores parámetros: {result.best_params}")

if __name__ == "__main__":
    main()

