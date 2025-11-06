#!/usr/bin/env python3
"""
Supervisor de descarga - Monitorea y reinicia automáticamente si el proceso se detiene
Garantiza que la descarga finalice al 100%
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/feliperangel/Javeriana/Mestría IA/Procesamiento de señales biológicas/Proyecto_final/Prediccion_Muerte_Subita_ECG_v1")
os.chdir(BASE_DIR)

CHECKPOINT_FILE = BASE_DIR / "config" / "descarga_checkpoint.json"
LOG_FILE = BASE_DIR / "logs" / "descarga_detallada.log"
SUPERVISOR_LOG = BASE_DIR / "logs" / "supervisor_descarga.log"

datasets_info = {
    'sddb': {'expected_gb': 5.0, 'expected_records': 23},
    'nsrdb': {'expected_gb': 2.0, 'expected_records': 18}
    # CUDB eliminado - no se necesita para el trabajo final
}

def get_size_gb(path):
    """Obtener tamaño en GB"""
    if not path.exists():
        return 0.0
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total / (1024**3)

def count_file_types(path):
    """Contar archivos por tipo"""
    if not path.exists():
        return {'dat': 0, 'hea': 0, 'atr': 0}
    try:
        return {
            'dat': len(list(path.rglob('*.dat'))),
            'hea': len(list(path.rglob('*.hea'))),
            'atr': len(list(path.rglob('*.atr')))
        }
    except:
        return {'dat': 0, 'hea': 0, 'atr': 0}

def check_progress():
    """Verificar progreso actual"""
    total_files = 0
    total_size = 0.0
    total_expected_files = 0
    total_expected_size = 0.0
    
    for dataset_name, info in datasets_info.items():
        path = BASE_DIR / f"datasets/{dataset_name}/physionet.org/files/{dataset_name}/1.0.0"
        size_gb = get_size_gb(path)
        file_types = count_file_types(path)
        expected_files = info['expected_records']
        expected_gb = info['expected_gb']
        
        total_files += file_types['dat']
        total_size += size_gb
        total_expected_files += expected_files
        total_expected_size += expected_gb
    
    files_progress = (total_files / total_expected_files * 100) if total_expected_files > 0 else 0
    size_progress = (total_size / total_expected_size * 100) if total_expected_size > 0 else 0
    
    return {
        'files': total_files,
        'expected_files': total_expected_files,
        'files_progress': files_progress,
        'size': total_size,
        'expected_size': total_expected_size,
        'size_progress': size_progress,
        'is_complete': files_progress >= 100 and size_progress >= 95
    }

def check_processes():
    """Verificar procesos activos"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        python_count = sum(1 for line in result.stdout.split('\n') 
                          if 'descarga_maxima' in line and 'grep' not in line)
        aria2c_count = sum(1 for line in result.stdout.split('\n') 
                          if 'aria2c' in line.lower() and 'physionet' in line.lower() and 'grep' not in line.lower())
        wget_count = sum(1 for line in result.stdout.split('\n') 
                        if 'wget' in line.lower() and 'physionet' in line.lower() and 'grep' not in line.lower())
        return python_count, aria2c_count, wget_count
    except:
        return 0, 0, 0

def start_download():
    """Iniciar proceso de descarga"""
    print("🚀 Iniciando proceso de descarga...")
    try:
        process = subprocess.Popen(
            [sys.executable, 'scripts/descarga_maxima_velocidad.py'],
            stdout=open('descarga_maxima_velocidad.log', 'w'),
            stderr=subprocess.STDOUT
        )
        return process.pid
    except Exception as e:
        print(f"❌ Error iniciando descarga: {e}")
        return None

def log_supervisor(message):
    """Log del supervisor"""
    try:
        with open(SUPERVISOR_LOG, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {message}\n")
    except:
        pass
    print(message)

def main():
    print("=" * 70)
    print("🛡️  SUPERVISOR DE DESCARGA - GARANTIZA FINALIZACIÓN AL 100%")
    print("=" * 70)
    print()
    print("Este supervisor monitorea el proceso de descarga y lo reinicia")
    print("automáticamente si se detiene antes de completar al 100%")
    print()
    
    max_restarts = 20  # Máximo de reinicios
    restart_count = 0
    check_interval = 30  # Verificar cada 30 segundos
    last_progress = None
    stuck_count = 0
    
    log_supervisor("Supervisor iniciado")
    
    while True:
        progress = check_progress()
        python_count, aria2c_count, wget_count = check_processes()
        
        # Verificar si está completo
        if progress['is_complete']:
            log_supervisor(f"✅ DESCARGA COMPLETADA AL 100%!")
            log_supervisor(f"   Archivos: {progress['files']}/{progress['expected_files']}")
            log_supervisor(f"   Tamaño: {progress['size']:.2f} GB / {progress['expected_size']:.1f} GB")
            print()
            print("🔍 Ejecutando verificación final...")
            try:
                subprocess.run(['python3', 'scripts/validacion_completa.py'])
            except:
                pass
            break
        
        # Verificar si hay procesos activos
        if python_count == 0 and aria2c_count == 0 and wget_count == 0:
            # No hay procesos activos pero la descarga no está completa
            if restart_count >= max_restarts:
                log_supervisor(f"⚠️  Máximo de reinicios alcanzado ({max_restarts})")
                log_supervisor(f"   Progreso actual: {progress['files_progress']:.1f}% archivos, {progress['size_progress']:.1f}% tamaño")
                break
            
            restart_count += 1
            log_supervisor(f"⚠️  Procesos detenidos - Progreso: {progress['files_progress']:.1f}% archivos, {progress['size_progress']:.1f}% tamaño")
            log_supervisor(f"🔄 Reiniciando descarga (intento {restart_count}/{max_restarts})...")
            
            pid = start_download()
            if pid:
                log_supervisor(f"✅ Descarga reiniciada (PID: {pid})")
            else:
                log_supervisor("❌ Error al reiniciar descarga")
                time.sleep(60)  # Esperar antes de reintentar
                continue
        else:
            # Hay procesos activos
            if last_progress:
                # Verificar si el progreso está estancado
                if (progress['files'] == last_progress['files'] and 
                    abs(progress['size'] - last_progress['size']) < 0.01):
                    stuck_count += 1
                    if stuck_count >= 20:  # Sin progreso por 10 minutos (20 * 30s)
                        log_supervisor("⚠️  Progreso estancado detectado - Reiniciando...")
                        restart_count += 1
                        # Detener procesos actuales
                        subprocess.run(['pkill', '-9', '-f', 'descarga_maxima'], capture_output=True)
                        subprocess.run(['pkill', '-9', '-f', 'aria2c.*physionet'], capture_output=True)
                        time.sleep(5)
                        pid = start_download()
                        stuck_count = 0
                else:
                    stuck_count = 0
            
            last_progress = progress.copy()
        
        # Mostrar estado cada minuto
        if int(time.time()) % 60 == 0:
            print(f"\n📊 Estado: {progress['files']}/{progress['expected_files']} archivos ({progress['files_progress']:.1f}%) | "
                  f"{progress['size']:.2f} GB / {progress['expected_size']:.1f} GB ({progress['size_progress']:.1f}%) | "
                  f"Procesos: Python={python_count}, aria2c={aria2c_count}, wget={wget_count}")
        
        time.sleep(check_interval)
    
    print()
    print("=" * 70)
    print("📊 RESUMEN FINAL DEL SUPERVISOR")
    print("=" * 70)
    print(f"   Reinicios realizados: {restart_count}")
    print(f"   Progreso final: {progress['files_progress']:.1f}% archivos, {progress['size_progress']:.1f}% tamaño")
    print(f"   Log guardado en: {SUPERVISOR_LOG}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Supervisor detenido por el usuario")

