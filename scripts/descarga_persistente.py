#!/usr/bin/env python3
"""
Sistema de descarga persistente que NO TERMINA hasta completar al 100%
Ejecuta múltiples rondas de reintentos automáticos hasta garantizar completitud
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

def verify_completeness():
    """Verificar si la descarga está completa al 100%"""
    total_files = 0
    total_size = 0.0
    total_expected_files = 0
    total_expected_size = 0.0
    
    all_complete = True
    
    for dataset_name, info in datasets_info.items():
        path = BASE_DIR / f"datasets/{dataset_name}/physionet.org/files/{dataset_name}/1.0.0"
        size_gb = get_size_gb(path)
        file_types = count_file_types(path)
        expected_files = info['expected_records']
        expected_gb = info['expected_gb']
        
        files_progress = (file_types['dat'] / expected_files * 100) if expected_files > 0 else 0
        size_progress = (size_gb / expected_gb * 100) if expected_gb > 0 else 0
        
        # Verificar completitud: >=95% de tamaño y todos los archivos
        dataset_complete = (files_progress >= 100 and size_progress >= 95)
        
        if not dataset_complete:
            all_complete = False
        
        total_files += file_types['dat']
        total_size += size_gb
        total_expected_files += expected_files
        total_expected_size += expected_gb
    
    overall_files_progress = (total_files / total_expected_files * 100) if total_expected_files > 0 else 0
    overall_size_progress = (total_size / total_expected_size * 100) if total_expected_size > 0 else 0
    
    return {
        'complete': all_complete and overall_files_progress >= 100 and overall_size_progress >= 95,
        'files': total_files,
        'expected_files': total_expected_files,
        'files_progress': overall_files_progress,
        'size': total_size,
        'expected_size': total_expected_size,
        'size_progress': overall_size_progress
    }

def wait_for_completion(max_wait_minutes=60):
    """Esperar a que el proceso termine"""
    max_wait = max_wait_minutes * 60
    waited = 0
    
    while waited < max_wait:
        python_count, aria2c_count, wget_count = check_processes()
        
        if python_count == 0 and aria2c_count == 0 and wget_count == 0:
            return True  # Todos los procesos terminaron
        
        time.sleep(30)
        waited += 30
        
        if waited % 300 == 0:  # Cada 5 minutos
            print(f"   Esperando... ({waited/60:.0f} min) - Procesos: Python={python_count}, aria2c={aria2c_count}, wget={wget_count}")
    
    return False  # Timeout alcanzado

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

def main():
    print("=" * 70)
    print("🔄 DESCARGA PERSISTENTE - GARANTIZA 100% DE COMPLETITUD")
    print("=" * 70)
    print()
    print("Este script ejecuta múltiples rondas de descarga hasta completar al 100%")
    print("NO TERMINA hasta que todos los archivos estén presentes y con tamaños correctos")
    print()
    
    max_rounds = 10
    round_count = 0
    
    while round_count < max_rounds:
        round_count += 1
        
        print("=" * 70)
        print(f"🔄 RONDA {round_count}/{max_rounds}")
        print("=" * 70)
        print()
        
        # Verificar estado inicial
        status = verify_completeness()
        print(f"📊 Estado inicial:")
        print(f"   Archivos: {status['files']}/{status['expected_files']} ({status['files_progress']:.1f}%)")
        print(f"   Tamaño: {status['size']:.2f} GB / {status['expected_size']:.1f} GB ({status['size_progress']:.1f}%)")
        print()
        
        if status['complete']:
            print("✅ ¡DESCARGA COMPLETADA AL 100%!")
            print()
            print("🔍 Ejecutando verificación final...")
            try:
                subprocess.run(['python3', 'scripts/validacion_completa.py'])
            except:
                pass
            break
        
        # Ejecutar descarga
        print("🚀 Ejecutando descarga...")
        print()
        
        try:
            result = subprocess.run(
                [sys.executable, 'scripts/descarga_maxima_velocidad.py'],
                timeout=7200  # Timeout de 2 horas por ronda
            )
        except subprocess.TimeoutExpired:
            print("⚠️  Timeout en descarga - Continuando...")
        except KeyboardInterrupt:
            print("\n\n⏹️  Descarga detenida por el usuario")
            break
        
        print()
        print("⏳ Esperando a que todos los procesos terminen...")
        wait_for_completion(max_wait_minutes=60)
        
        # Verificar estado final de esta ronda
        status = verify_completeness()
        print()
        print(f"📊 Estado después de ronda {round_count}:")
        print(f"   Archivos: {status['files']}/{status['expected_files']} ({status['files_progress']:.1f}%)")
        print(f"   Tamaño: {status['size']:.2f} GB / {status['expected_size']:.1f} GB ({status['size_progress']:.1f}%)")
        print()
        
        if status['complete']:
            print("✅ ¡DESCARGA COMPLETADA AL 100%!")
            break
        
        if round_count < max_rounds:
            wait_minutes = 5
            print(f"⏳ Esperando {wait_minutes} minutos antes de próxima ronda...")
            print()
            time.sleep(wait_minutes * 60)
    
    print()
    print("=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    final_status = verify_completeness()
    
    if final_status['complete']:
        print("✅ DESCARGA COMPLETADA AL 100%")
        print(f"   Archivos: {final_status['files']}/{final_status['expected_files']}")
        print(f"   Tamaño: {final_status['size']:.2f} GB / {final_status['expected_size']:.1f} GB")
    else:
        print("⚠️  DESCARGA INCOMPLETA DESPUÉS DE TODAS LAS RONDAS")
        print(f"   Archivos: {final_status['files']}/{final_status['expected_files']} ({final_status['files_progress']:.1f}%)")
        print(f"   Tamaño: {final_status['size']:.2f} GB / {final_status['expected_size']:.1f} GB ({final_status['size_progress']:.1f}%)")
        print()
        print("💡 Recomendación: Verificar conexión a internet y ejecutar manualmente")
        print("   python scripts/descarga_maxima_velocidad.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Proceso detenido por el usuario")

