#!/usr/bin/env python3
"""
Script principal para ejecutar el pipeline completo mejorado:
1. Entrenamiento con todos los registros
2. Optimización de hiperparámetros
3. Evaluación con validación cruzada 10-fold
4. Análisis de características
5. Análisis de errores
6. Comparación con baselines
7. Generación de dashboard actualizado
"""

import sys
from pathlib import Path
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Ejecutar pipeline completo mejorado')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio para modelos')
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Directorio para resultados')
    parser.add_argument('--skip-training', action='store_true',
                       help='Omitir entrenamiento (usar modelos existentes)')
    parser.add_argument('--skip-hyperparams', action='store_true',
                       help='Omitir optimización de hiperparámetros')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Omitir análisis profundos')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    parser.add_argument('--cv-folds', type=int, default=10,
                       help='Número de folds para validación cruzada')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 PIPELINE COMPLETO MEJORADO - Predicción de Muerte Súbita Cardíaca")
    print("=" * 70)
    
    # 1. Entrenamiento con todos los registros
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("📚 FASE 1: Entrenamiento de Modelos")
        print("=" * 70)
        
        cmd = [
            sys.executable, 'scripts/train_models.py',
            '--data-dir', args.data_dir,
            '--models-dir', args.models_dir,
            '--train-all'
        ]
        
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("⚠️  Error en entrenamiento. Continuando...")
    else:
        print("\n⏭️  Omitiendo entrenamiento (usando modelos existentes)")
    
    # 2. Optimización de hiperparámetros
    if not args.skip_hyperparams:
        print("\n" + "=" * 70)
        print("⚙️  FASE 2: Optimización de Hiperparámetros")
        print("=" * 70)
        
        cmd = [
            sys.executable, 'scripts/hyperparameter_optimization.py',
            '--data-dir', args.data_dir,
            '--output', str(Path(args.results_dir) / 'hyperparameter_search_results.pkl'),
            '--best-params-output', str(Path(args.results_dir) / 'best_hyperparameters.pkl'),
            '--search-type', 'random',
            '--max-combinations', '30'
        ]
        
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("⚠️  Error en optimización de hiperparámetros. Continuando...")
    else:
        print("\n⏭️  Omitiendo optimización de hiperparámetros")
    
    # 3. Evaluación con validación cruzada 10-fold
    print("\n" + "=" * 70)
    print("📊 FASE 3: Evaluación con Validación Cruzada")
    print("=" * 70)
    
    cmd = [
        sys.executable, 'scripts/evaluate_models.py',
        '--models-dir', args.models_dir,
        '--data-dir', args.data_dir,
        '--output', str(Path(args.results_dir) / 'evaluation_results.pkl'),
        '--cv-folds', str(args.cv_folds)
    ]
    
    if args.max_records:
        cmd.extend(['--max-records', str(args.max_records)])
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("⚠️  Error en evaluación. Continuando...")
    
    # 4. Análisis profundos
    if not args.skip_analysis:
        print("\n" + "=" * 70)
        print("🔬 FASE 4: Análisis Profundos")
        print("=" * 70)
        
        # 4.1 Análisis de características
        print("\n📊 4.1: Análisis de Importancia de Características")
        cmd = [
            sys.executable, 'scripts/feature_importance_analysis.py',
            '--models-dir', args.models_dir,
            '--data-dir', args.data_dir,
            '--output', str(Path(args.results_dir) / 'feature_importance_results.pkl')
        ]
        
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("⚠️  Error en análisis de características. Continuando...")
        
        # 4.2 Análisis de errores
        print("\n🔍 4.2: Análisis de Errores")
        cmd = [
            sys.executable, 'scripts/error_analysis.py',
            '--models-dir', args.models_dir,
            '--data-dir', args.data_dir,
            '--output', str(Path(args.results_dir) / 'error_analysis_results.pkl')
        ]
        
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("⚠️  Error en análisis de errores. Continuando...")
        
        # 4.3 Comparación con baselines
        print("\n📊 4.3: Comparación con Métodos Baseline")
        cmd = [
            sys.executable, 'scripts/baseline_comparison.py',
            '--models-dir', args.models_dir,
            '--data-dir', args.data_dir,
            '--output', str(Path(args.results_dir) / 'baseline_comparison_results.pkl')
        ]
        
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("⚠️  Error en comparación con baselines. Continuando...")
    else:
        print("\n⏭️  Omitiendo análisis profundos")
    
    # 5. Generación de dashboard actualizado
    print("\n" + "=" * 70)
    print("📈 FASE 5: Generación de Dashboard")
    print("=" * 70)
    
    cmd = [
        sys.executable, 'scripts/generate_dashboard.py',
        '--output', str(Path(args.results_dir) / 'dashboard_scd_prediction.html'),
        '--models-dir', args.models_dir,
        '--results-file', str(Path(args.results_dir) / 'evaluation_results.pkl')
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("⚠️  Error generando dashboard. Continuando...")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETO FINALIZADO")
    print("=" * 70)
    print(f"\n📊 Dashboard disponible en: {Path(args.results_dir) / 'dashboard_scd_prediction.html'}")
    print(f"📁 Todos los resultados guardados en: {args.results_dir}")

if __name__ == "__main__":
    main()

