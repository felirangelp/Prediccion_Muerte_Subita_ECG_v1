#!/usr/bin/env python3
"""
Script que espera a que termine el entrenamiento y luego ejecuta automáticamente
la validación cruzada (10-fold)
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

def check_training_completed():
    """Verificar si el entrenamiento ha terminado"""
    # Verificar proceso
    result = subprocess.run(["pgrep", "-f", "train_models.py --train-all"], 
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        # Proceso no encontrado, verificar si hay modelos nuevos
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.h5"))
            if model_files:
                # Verificar si los modelos son recientes (últimas 2 horas)
                latest = max(model_files, key=lambda x: x.stat().st_mtime)
                age_hours = (time.time() - latest.stat().st_mtime) / 3600
                
                if age_hours < 2:
                    return True, "Modelos recientemente actualizados"
        
        return True, "Proceso terminado"
    
    return False, "Proceso activo"

def run_cross_validation():
    """Ejecutar validación cruzada"""
    print("\n" + "=" * 70)
    print("🚀 INICIANDO VALIDACIÓN CRUZADA (10-fold)")
    print("=" * 70)
    print(f"🕐 Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cmd = [
        "python3", "scripts/evaluate_models.py",
        "--cv-folds", "10",
        "--cross-validation-only",
        "--data-dir", "datasets/",
        "--models-dir", "models/"
    ]
    
    print(f"📋 Comando: {' '.join(cmd)}")
    print(f"📁 Log: logs/cross_validation.log")
    print("=" * 70)
    
    # Ejecutar en background
    with open("logs/cross_validation.log", "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd()
        )
    
    print(f"✅ Proceso iniciado (PID: {process.pid})")
    print(f"⏳ Tiempo estimado: 35-50 horas")
    print(f"💡 Monitorea con: tail -f logs/cross_validation.log")
    
    # Iniciar monitoreo automático de CV
    print("\n📊 Iniciando monitoreo automático de CV...")
    monitor_cmd = [
        "python3", "scripts/monitor_cv_evaluation.py"
    ]
    
    with open("logs/monitor_cv.log", "w") as monitor_log:
        monitor_process = subprocess.Popen(
            monitor_cmd,
            stdout=monitor_log,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd()
        )
    
    print(f"✅ Monitoreo iniciado (PID: {monitor_process.pid})")
    print(f"   Este monitoreo ejecutará automáticamente:")
    print(f"   - Generación de dashboard al completar")
    print(f"   - Actualización de GitHub Pages al completar")
    
    return process.pid

def main():
    """Función principal"""
    print("=" * 70)
    print("⏳ ESPERANDO A QUE TERMINE EL ENTRENAMIENTO")
    print("=" * 70)
    print("Este script monitoreará el entrenamiento y ejecutará")
    print("automáticamente la validación cruzada cuando termine.")
    print("=" * 70)
    
    check_interval = 300  # Verificar cada 5 minutos
    
    while True:
        completed, message = check_training_completed()
        
        if completed:
            print(f"\n✅ {message}")
            print("Iniciando validación cruzada...")
            pid = run_cross_validation()
            
            print("\n" + "=" * 70)
            print("✅ VALIDACIÓN CRUZADA INICIADA")
            print("=" * 70)
            print(f"PID: {pid}")
            print(f"El proceso continuará en background.")
            print("Usa 'scripts/monitor_cv_evaluation.py' para monitorear.")
            print("=" * 70)
            break
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")
        print(f"   Próxima verificación en {check_interval // 60} minutos...")
        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Script detenido por el usuario")
        sys.exit(0)

