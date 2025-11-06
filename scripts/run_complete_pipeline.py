#!/usr/bin/env python3
"""
Script principal para ejecutar el pipeline completo:
1. Entrenar modelos
2. Evaluar modelos
3. Generar dashboard
4. Análisis completo
"""

import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Ejecutar pipeline completo')
    parser.add_argument('--data-dir', type=str, default='datasets/',
                       help='Directorio con datasets')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio para modelos')
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Directorio para resultados')
    parser.add_argument('--skip-training', action='store_true',
                       help='Omitir entrenamiento (usar modelos existentes)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Omitir evaluación')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Número máximo de registros por dataset')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 PIPELINE COMPLETO - Predicción de Muerte Súbita Cardíaca")
    print("=" * 70)
    
    # 1. Entrenar modelos
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("📚 FASE 1: Entrenamiento de Modelos")
        print("=" * 70)
        
        import subprocess
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
    
    # 2. Evaluar modelos
    if not args.skip_evaluation:
        print("\n" + "=" * 70)
        print("📊 FASE 2: Evaluación de Modelos")
        print("=" * 70)
        
        import subprocess
        cmd = [
            sys.executable, 'scripts/evaluate_models.py',
            '--models-dir', args.models_dir,
            '--data-dir', args.data_dir,
            '--output', str(Path(args.results_dir) / 'evaluation_results.pkl')
        ]
        
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("⚠️  Error en evaluación. Continuando...")
    else:
        print("\n⏭️  Omitiendo evaluación")
    
    # 3. Generar dashboard
    print("\n" + "=" * 70)
    print("📈 FASE 3: Generación de Dashboard")
    print("=" * 70)
    
    import subprocess
    cmd = [
        sys.executable, 'scripts/generate_dashboard.py',
        '--output', str(Path(args.results_dir) / 'dashboard_scd_prediction.html'),
        '--models-dir', args.models_dir,
        '--results-file', str(Path(args.results_dir) / 'evaluation_results.pkl')
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("⚠️  Error generando dashboard. Continuando...")
    
    # 4. Análisis completo
    print("\n" + "=" * 70)
    print("🔬 FASE 4: Análisis Completo")
    print("=" * 70)
    
    import subprocess
    cmd = [
        sys.executable, 'scripts/comprehensive_analysis.py',
        '--models-dir', args.models_dir,
        '--data-dir', args.data_dir,
        '--output-dir', args.results_dir
    ]
    
    if args.max_records:
        cmd.extend(['--max-records', str(args.max_records)])
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("⚠️  Error en análisis completo. Continuando...")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 70)
    print(f"\n📁 Resultados guardados en:")
    print(f"   - Dashboard: {args.results_dir}/dashboard_scd_prediction.html")
    print(f"   - Resultados: {args.results_dir}/")
    print(f"   - Modelos: {args.models_dir}/")
    print(f"\n💡 Para ver el dashboard, abre {args.results_dir}/dashboard_scd_prediction.html en tu navegador")

if __name__ == "__main__":
    main()

