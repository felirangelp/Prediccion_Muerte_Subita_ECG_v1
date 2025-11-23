#!/usr/bin/env python3
"""
Script para monitorear automáticamente el progreso del entrenamiento
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
import os
import signal

def signal_handler(sig, frame):
    """Manejar Ctrl+C"""
    print("\n\n⏸️  Monitoreo detenido por el usuario")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def check_training_progress():
    """Verificar progreso del entrenamiento"""
    print("=" * 70)
    print(f"📊 VERIFICACIÓN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Verificar proceso
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        process_lines = [line for line in result.stdout.split('\n') 
                        if 'train_models.py' in line and 'grep' not in line]
        
        if process_lines:
            parts = process_lines[0].split()
            pid = parts[1]
            cpu = parts[2]
            mem = parts[3]
            
            # Obtener tiempo transcurrido y estado
            etime_result = subprocess.run(['ps', '-p', pid, '-o', 'etime,state='], 
                                        capture_output=True, text=True, timeout=5)
            etime_data = etime_result.stdout.strip().split()
            if len(etime_data) >= 2:
                etime = etime_data[0]
                state = etime_data[1]
            else:
                etime = etime_result.stdout.strip()
                state = "?"
            
            # Determinar qué modelo está entrenando
            cmd = ' '.join(parts[10:])
            model_type = "Desconocido"
            if '--train-sparse' in cmd:
                model_type = "Sparse (K-SVD + OMP)"
            elif '--train-hierarchical' in cmd:
                model_type = "Hierarchical (TCN + Fusion)"
            elif '--train-hybrid' in cmd:
                model_type = "Hybrid"
            elif '--train-all' in cmd:
                model_type = "Todos los modelos"
            
            print(f"\n🔄 Proceso activo (PID: {pid}):")
            print(f"   Modelo: {model_type}")
            print(f"   Estado: {state} ({'✅ ACTIVO' if state == 'R' else '⏸️  EN ESPERA'})")
            print(f"   Tiempo transcurrido: {etime}")
            print(f"   CPU: {cpu}% {'🔥 ALTA ACTIVIDAD' if float(cpu) > 50 else '⚠️  BAJA ACTIVIDAD'}")
            print(f"   Memoria: {mem}%")
            
            # Verificar última actividad del log
            log_file = '/tmp/training_sparse_fixed.log'
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                print(f"\n📝 Última línea del log:")
                                print(f"   {last_line[:80]}...")
                except:
                    pass
            
            return True, pid
        else:
            print("\n⚠️  Proceso no encontrado (puede haber terminado)")
            return False, None
    except Exception as e:
        print(f"\n⚠️  Error verificando proceso: {e}")
        return None, None
    
    # Verificar modelos
    models_dir = Path('models')
    print(f"\n📁 Estado de modelos:")
    
    # Hierarchical
    h_files = list(models_dir.glob('hierarchical_classifier*'))
    if h_files:
        latest = max(h_files, key=lambda x: os.path.getmtime(x))
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest))
        print(f"   ✅ Hierarchical: {len(h_files)} archivos, actualizado hace {str(age).split('.')[0]}")
    else:
        print(f"   ⏳ Hierarchical: No encontrado")
    
    # Hybrid
    hy_files = list(models_dir.glob('hybrid_model*'))
    if hy_files:
        latest = max(hy_files, key=lambda x: os.path.getmtime(x))
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest))
        print(f"   ✅ Hybrid: {len(hy_files)} archivos, actualizado hace {str(age).split('.')[0]}")
    else:
        print(f"   ⏳ Hybrid: No encontrado")
    
    # Sparse
    sparse_file = models_dir / 'sparse_classifier.pkl'
    if sparse_file.exists():
        mtime = os.path.getmtime(sparse_file)
        age = datetime.now() - datetime.fromtimestamp(mtime)
        size = sparse_file.stat().st_size / (1024*1024)
        print(f"   ✅ Sparse: {size:.2f} MB, actualizado hace {str(age).split('.')[0]}")
        
        # Verificar si se está actualizando
        if age.total_seconds() < 3600:  # Menos de 1 hora
            print(f"      🔄 Posiblemente en entrenamiento (muy reciente)")
    else:
        print(f"   ⏳ Sparse: No encontrado")
    
    return None, None

def main():
    """Función principal"""
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print("❌ Error: El intervalo debe ser un número (segundos)")
            sys.exit(1)
    else:
        interval = 1800  # 30 minutos por defecto
    
    max_checks = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          MONITOREO AUTOMÁTICO DE ENTRENAMIENTO                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("")
    print("📊 Configuración:")
    print(f"   Intervalo: {interval} segundos ({interval // 60} minutos)")
    if max_checks == 0:
        print("   Verificaciones: Infinitas (Ctrl+C para detener)")
    else:
        print(f"   Verificaciones: {max_checks}")
    print("")
    print("🔄 Iniciando monitoreo...")
    print("")
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            
            is_running, pid = check_training_progress()
            
            if is_running is False:
                print("\n✅ El proceso de entrenamiento ha terminado!")
                print("   Verificando si todos los modelos se completaron...")
                
                # Verificar modelos finales
                models_dir = Path('models')
                sparse_exists = (models_dir / 'sparse_classifier.pkl').exists()
                hierarchical_exists = len(list(models_dir.glob('hierarchical_classifier*'))) > 0
                
                if sparse_exists and hierarchical_exists:
                    print("   ✅ Todos los modelos parecen estar completos")
                    print("")
                    print("📋 Próximos pasos sugeridos:")
                    print("   1. Evaluar modelos: python scripts/evaluate_models.py")
                    print("   2. Actualizar dashboard: python scripts/generate_dashboard.py")
                
                break
            
            if max_checks > 0 and check_count >= max_checks:
                print("\n⏸️  Límite de verificaciones alcanzado")
                break
            
            print("")
            print(f"⏳ Esperando {interval} segundos hasta la próxima verificación...")
            print("   (Presiona Ctrl+C para detener el monitoreo)")
            print("")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoreo detenido por el usuario")
    
    print("\n✅ Monitoreo finalizado")

if __name__ == "__main__":
    main()

