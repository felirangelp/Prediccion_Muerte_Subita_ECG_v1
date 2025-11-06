#!/usr/bin/env python3
"""
Monitor específico para descarga con aria2c
"""

import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path("/Users/feliperangel/Javeriana/Mestría IA/Procesamiento de señales biológicas/Proyecto_final/Prediccion_Muerte_Subita_ECG_v1")

datasets_info = {
    'sddb': {'expected_gb': 5.0, 'expected_records': 23},
    'nsrdb': {'expected_gb': 2.0, 'expected_records': 18},
    'cudb': {'expected_gb': 9.5, 'expected_records': 35}
}

# Para calcular velocidad
previous_sizes = {}
previous_time = time.time()

def get_size_gb(path):
    """Obtener tamaño en GB"""
    if not path.exists():
        return 0.0
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                try:
                    total += os.path.getsize(fp)
                except:
                    pass
    return total / (1024**3)

def count_dat_files(path):
    """Contar archivos .dat"""
    if not path.exists():
        return 0
    try:
        return len(list(path.rglob("*.dat")))
    except:
        return 0

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

def get_download_speed(current_sizes, prev_sizes, time_diff):
    """Calcular velocidad de descarga en MB/s"""
    if time_diff <= 0:
        return 0.0
    
    total_diff = 0.0
    for dataset in current_sizes:
        current = current_sizes.get(dataset, 0.0)
        previous = prev_sizes.get(dataset, 0.0)
        total_diff += max(0, current - previous)
    
    # Convertir GB a MB y dividir por tiempo
    speed_mb_s = (total_diff * 1024) / time_diff
    return speed_mb_s

def print_progress_bar(percentage, width=40):
    """Generar barra de progreso visual"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def count_processes():
    """Contar procesos activos"""
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
    global previous_sizes, previous_time
    
    print("=" * 70)
    print("⚡ MONITOR DE DESCARGA CON ARIA2C - MÁXIMA VELOCIDAD")
    print("=" * 70)
    print()
    
    iteration = 0
    current_sizes = {}
    
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 70)
        print("⚡ MONITOR DE DESCARGA CON ARIA2C - MÁXIMA VELOCIDAD")
        print("=" * 70)
        print()
        print(f"⏱️  {time.strftime('%Y-%m-%d %H:%M:%S')} - Iteración: {iteration}")
        print()
        
        python_count, aria2c_count, wget_count = count_processes()
        
        # Calcular velocidad
        current_time = time.time()
        time_diff = current_time - previous_time if previous_time > 0 else 0
        
        total_files = 0
        total_size = 0.0
        total_expected_files = 0
        total_expected_size = 0.0
        
        current_sizes = {}
        
        print("📊 ESTADO POR DATASET:")
        print()
        
        for dataset_name, info in datasets_info.items():
            path = BASE_DIR / f"datasets/{dataset_name}/physionet.org/files/{dataset_name}/1.0.0"
            size_gb = get_size_gb(path)
            file_types = count_file_types(path)
            dat_files = file_types['dat']
            expected_files = info['expected_records']
            expected_gb = info['expected_gb']
            
            current_sizes[dataset_name] = size_gb
            
            progress_size = (size_gb / expected_gb * 100) if expected_gb > 0 else 0
            progress_files = (dat_files / expected_files * 100) if expected_files > 0 else 0
            
            if progress_size >= 95 and dat_files >= expected_files:
                status = "✅"
            elif size_gb > 0 or dat_files > 0:
                status = "🔄"
            else:
                status = "⏳"
            
            print(f"{status} {dataset_name.upper()}")
            print(f"   📄 Archivos:")
            print(f"      .dat: {file_types['dat']} / {expected_files} ({progress_files:.1f}%)")
            print(f"      .hea: {file_types['hea']} / {expected_files}")
            print(f"      .atr: {file_types['atr']} / {expected_files}")
            print(f"   📊 Tamaño: {size_gb:.2f} GB / {expected_gb:.0f} GB ({progress_size:.1f}%)")
            print(f"      {print_progress_bar(progress_size)}")
            print()
            
            total_files += dat_files
            total_size += size_gb
            total_expected_files += expected_files
            total_expected_size += expected_gb
        
        # Calcular velocidad
        speed_mb_s = 0.0
        if time_diff > 0 and previous_sizes:
            speed_mb_s = get_download_speed(current_sizes, previous_sizes, time_diff)
        
        previous_sizes = current_sizes.copy()
        previous_time = current_time
        
        overall_files_progress = (total_files / total_expected_files * 100) if total_expected_files > 0 else 0
        overall_size_progress = (total_size / total_expected_size * 100) if total_expected_size > 0 else 0
        
        # Calcular ETA
        eta_minutes = 0
        if speed_mb_s > 0:
            remaining_gb = total_expected_size - total_size
            remaining_mb = remaining_gb * 1024
            eta_seconds = remaining_mb / speed_mb_s
            eta_minutes = eta_seconds / 60
        
        print("=" * 70)
        print("📈 RESUMEN")
        print("=" * 70)
        print(f"   ⚡ Procesos Python activos: {python_count}")
        print(f"   ⚡ Procesos aria2c activos: {aria2c_count}")
        print(f"   📥 Procesos wget activos: {wget_count}")
        print()
        print(f"   📄 Archivos .dat: {total_files} / {total_expected_files} ({overall_files_progress:.1f}%)")
        print(f"      {print_progress_bar(overall_files_progress)}")
        print()
        print(f"   📊 Tamaño total: {total_size:.2f} GB / {total_expected_size:.0f} GB ({overall_size_progress:.1f}%)")
        print(f"      {print_progress_bar(overall_size_progress)}")
        print()
        
        if speed_mb_s > 0:
            print(f"   ⚡ Velocidad actual: {speed_mb_s:.2f} MB/s")
        if eta_minutes > 0 and eta_minutes < 10000:
            print(f"   ⏱️  Tiempo estimado restante: {eta_minutes:.1f} minutos")
        
        if aria2c_count > 0:
            print(f"   ⚡ Usando aria2c: {aria2c_count} procesos activos (MÁXIMA VELOCIDAD)")
            print(f"   📊 Conexiones totales estimadas: ~{aria2c_count * 32}")
        
        print()
        
        # Verificar si está completo
        if aria2c_count == 0 and wget_count == 0 and python_count == 0:
            if total_files >= total_expected_files:
                print("✅ ¡DESCARGA COMPLETADA!")
                print()
                print("🔍 Ejecutando verificación de integridad...")
                print()
                try:
                    subprocess.run(['python3', 'scripts/validacion_completa.py'])
                except:
                    print("⚠️  Ejecutar manualmente: python scripts/validacion_completa.py")
                break
            else:
                print("⚠️  Procesos detenidos pero descarga incompleta")
                print("   Verificar logs: tail -f descarga_aria2c.log")
        
        print("💡 Presiona Ctrl+C para salir")
        print()
        
        iteration += 1
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitor detenido por el usuario")

