#!/usr/bin/env python3
"""
Monitor de entrenamiento en tiempo real
Muestra el progreso de los modelos que se están entrenando
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/feliperangel/Javeriana/Mestría IA/Procesamiento de señales biológicas/Proyecto_final/Prediccion_Muerte_Subita_ECG_v1")
os.chdir(BASE_DIR)

def check_training_processes():
    """Verificar si hay procesos de entrenamiento activos"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        train_processes = [
            line for line in result.stdout.split('\n')
            if 'train_models.py' in line.lower() and 'grep' not in line.lower()
        ]
        return len(train_processes), train_processes
    except:
        return 0, []

def check_model_files():
    """Verificar qué modelos se han generado"""
    models_dir = BASE_DIR / 'models'
    if not models_dir.exists():
        return {}
    
    models_status = {
        'sparse': {
            'files': [],
            'exists': False,
            'size': 0
        },
        'hierarchical': {
            'files': [],
            'exists': False,
            'size': 0
        },
        'hybrid': {
            'files': [],
            'exists': False,
            'size': 0
        }
    }
    
    # Verificar modelo sparse
    sparse_file = models_dir / 'sparse_classifier.pkl'
    if sparse_file.exists():
        models_status['sparse']['exists'] = True
        models_status['sparse']['size'] = sparse_file.stat().st_size / (1024 * 1024)  # MB
        models_status['sparse']['files'].append('sparse_classifier.pkl')
    
    # Verificar modelo hierarchical
    hierarchical_files = [
        'hierarchical_classifier_fusion.h5',
        'hierarchical_classifier_sparse.pkl',
        'hierarchical_classifier_linear.pkl',
        'hierarchical_classifier_nonlinear.pkl'
    ]
    for fname in hierarchical_files:
        fpath = models_dir / fname
        if fpath.exists():
            models_status['hierarchical']['files'].append(fname)
            models_status['hierarchical']['size'] += fpath.stat().st_size / (1024 * 1024)
            models_status['hierarchical']['exists'] = True
    
    # Verificar modelo hybrid
    hybrid_files = [
        'hybrid_model_sparse.pkl',
        'hybrid_model_hierarchical.h5',
        'hybrid_model_ensemble.pkl'
    ]
    for fname in hybrid_files:
        fpath = models_dir / fname
        if fpath.exists():
            models_status['hybrid']['files'].append(fname)
            models_status['hybrid']['size'] += fpath.stat().st_size / (1024 * 1024)
            models_status['hybrid']['exists'] = True
    
    return models_status

def get_training_log():
    """Obtener últimas líneas del log de entrenamiento si existe"""
    log_files = [
        BASE_DIR / 'logs' / 'entrenamiento.log',
        BASE_DIR / 'entrenamiento.log',
        BASE_DIR / 'train.log'
    ]
    
    for log_file in log_files:
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    return lines[-20:] if len(lines) > 20 else lines
            except:
                pass
    
    return []

def print_progress_bar(percentage, width=40):
    """Imprimir barra de progreso"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def main():
    """Función principal del monitor"""
    print("=" * 70)
    print("🔄 MONITOR DE ENTRENAMIENTO EN TIEMPO REAL")
    print("=" * 70)
    print("Actualizando cada 5 segundos...")
    print("Presiona Ctrl+C para salir")
    print()
    
    previous_models = {}
    iteration = 0
    
    try:
        while True:
            # Limpiar pantalla
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 70)
            print("🔄 MONITOR DE ENTRENAMIENTO EN TIEMPO REAL")
            print("=" * 70)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Iteración: {iteration}")
            print()
            
            # Verificar procesos activos
            num_processes, process_lines = check_training_processes()
            
            if num_processes > 0:
                print(f"✅ Procesos de entrenamiento activos: {num_processes}")
                print()
                for line in process_lines[:3]:  # Mostrar solo primeras 3 líneas
                    parts = line.split()
                    if len(parts) > 10:
                        pid = parts[1]
                        cpu = parts[2]
                        mem = parts[3]
                        print(f"   PID: {pid} | CPU: {cpu}% | Mem: {mem}%")
            else:
                print("⚠️  No hay procesos de entrenamiento activos")
                print("   El entrenamiento puede haber terminado o no haber iniciado")
            
            print()
            
            # Verificar estado de modelos
            models_status = check_model_files()
            
            print("📊 ESTADO DE MODELOS:")
            print()
            
            models_info = {
                'sparse': {
                    'name': 'Representaciones Dispersas',
                    'icon': '📊',
                    'status': models_status['sparse']
                },
                'hierarchical': {
                    'name': 'Fusión Jerárquica',
                    'icon': '🔗',
                    'status': models_status['hierarchical']
                },
                'hybrid': {
                    'name': 'Modelo Híbrido',
                    'icon': '🎯',
                    'status': models_status['hybrid']
                }
            }
            
            for model_key, model_info in models_info.items():
                status = model_info['status']
                icon = model_info['icon']
                name = model_info['name']
                
                if status['exists']:
                    print(f"{icon} {name}: ✅ COMPLETADO")
                    print(f"   Archivos: {len(status['files'])}")
                    print(f"   Tamaño total: {status['size']:.2f} MB")
                    if status['files']:
                        print(f"   Archivos generados:")
                        for fname in status['files'][:3]:
                            print(f"      - {fname}")
                        if len(status['files']) > 3:
                            print(f"      ... y {len(status['files']) - 3} más")
                else:
                    # Verificar si hay cambios desde la última iteración
                    if model_key in previous_models:
                        if previous_models[model_key] != status:
                            print(f"{icon} {name}: 🔄 ENTRENANDO...")
                        else:
                            print(f"{icon} {name}: ⏳ PENDIENTE")
                    else:
                        print(f"{icon} {name}: ⏳ PENDIENTE")
                
                print()
            
            previous_models = models_status.copy()
            
            # Mostrar log reciente si existe
            log_lines = get_training_log()
            if log_lines:
                print("📝 ÚLTIMAS LÍNEAS DEL LOG:")
                print("-" * 70)
                for line in log_lines[-5:]:
                    print(f"   {line.rstrip()}")
                print()
            
            # Resumen
            total_completed = sum(1 for s in models_status.values() if s['exists'])
            total_models = len(models_status)
            
            print("=" * 70)
            print("📈 RESUMEN")
            print("=" * 70)
            print(f"Modelos completados: {total_completed}/{total_models}")
            print(f"Progreso general: {print_progress_bar((total_completed / total_models) * 100)}")
            print()
            
            if total_completed == total_models:
                print("🎉 ¡ENTRENAMIENTO COMPLETADO!")
                print("   Todos los modelos han sido entrenados exitosamente")
                print()
                print("💡 Próximos pasos:")
                print("   1. Evaluar modelos: python scripts/evaluate_models.py")
                print("   2. Generar dashboard: python scripts/generate_dashboard.py")
            else:
                print("💡 El entrenamiento continúa...")
                print("   Próxima actualización en 5 segundos...")
            
            iteration += 1
            time.sleep(5)
            
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("⏹️  Monitor detenido")
        print("=" * 70)
        print()
        
        # Estado final
        models_status = check_model_files()
        total_completed = sum(1 for s in models_status.values() if s['exists'])
        total_models = len(models_status)
        
        print("📊 Estado final:")
        print(f"   Modelos completados: {total_completed}/{total_models}")
        
        if total_completed > 0:
            print()
            print("✅ Modelos entrenados:")
            for model_key, model_info in models_info.items():
                if models_status[model_key]['exists']:
                    print(f"   - {model_info['name']}")

if __name__ == "__main__":
    main()

