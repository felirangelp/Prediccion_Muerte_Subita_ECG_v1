#!/usr/bin/env python3
"""
Monitor continuo de progreso de descarga
Muestra el progreso actualizado cada 10 segundos
"""
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/feliperangel/Javeriana/Mestría IA/Procesamiento de señales biológicas/Proyecto_final/Prediccion_Muerte_Subita_ECG_v1")
os.chdir(BASE_DIR)

datasets_info = {
    'sddb': {'expected_gb': 5.0, 'expected_records': 23},
    'nsrdb': {'expected_gb': 2.0, 'expected_records': 18}
}

def get_size_gb(path):
    """Calcular tamaño total en GB"""
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

def count_files(path, ext):
    """Contar archivos por extensión"""
    if not path.exists():
        return 0
    try:
        return len(list(path.rglob(f'*.{ext}')))
    except:
        return 0

def check_processes():
    """Verificar procesos activos"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        python_count = sum(1 for line in result.stdout.split('\n') 
                          if 'descarga' in line.lower() and 'grep' not in line.lower() and 'monitor' not in line.lower())
        aria2c_count = sum(1 for line in result.stdout.split('\n') 
                          if 'aria2c' in line.lower() and 'physionet' in line.lower() and 'grep' not in line.lower())
        return python_count, aria2c_count
    except:
        return 0, 0

def print_progress_bar(percentage, width=40):
    """Imprimir barra de progreso"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def get_skipped_count():
    """Obtener número de archivos omitidos"""
    skip_file = BASE_DIR / 'config' / 'archivos_omitidos.json'
    if skip_file.exists():
        try:
            with open(skip_file, 'r') as f:
                skipped = json.load(f)
            return len(skipped)
        except:
            pass
    return 0

def main():
    print("=" * 70)
    print("🔄 MONITOR CONTINUO DE DESCARGA")
    print("=" * 70)
    print("Actualizando cada 10 segundos...")
    print("Presiona Ctrl+C para salir")
    print()
    
    previous_size = {}
    previous_time = {}
    
    try:
        iteration = 0
        while True:
            # Limpiar pantalla (compatible con diferentes terminales)
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 70)
            print("🔄 MONITOR CONTINUO DE DESCARGA")
            print("=" * 70)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Iteración: {iteration}")
            print()
            
            python_count, aria2c_count = check_processes()
            
            print(f"🔹 Procesos Python (descarga): {python_count}")
            print(f"🔹 Procesos aria2c: {aria2c_count}")
            print()
            
            total_size = 0.0
            total_dat = 0
            total_hea = 0
            total_atr = 0
            total_expected_dat = 0
            
            for dataset_name, info in datasets_info.items():
                path = BASE_DIR / f'datasets/{dataset_name}/physionet.org/files/{dataset_name}/1.0.0'
                
                size = get_size_gb(path)
                dat_count = count_files(path, 'dat')
                hea_count = count_files(path, 'hea')
                atr_count = count_files(path, 'atr')
                
                total_size += size
                total_dat += dat_count
                total_hea += hea_count
                total_atr += atr_count
                total_expected_dat += info['expected_records']
                
                progress_pct = (size / info['expected_gb']) * 100 if info['expected_gb'] > 0 else 0
                dat_pct = (dat_count / info['expected_records']) * 100 if info['expected_records'] > 0 else 0
                
                # Calcular velocidad si hay datos previos
                speed_str = ""
                if dataset_name in previous_size:
                    time_diff = time.time() - previous_time.get(dataset_name, time.time())
                    if time_diff > 0:
                        size_diff = size - previous_size[dataset_name]
                        speed_mb_s = (size_diff / time_diff) * 1024
                        if speed_mb_s > 0:
                            speed_str = f" | ⚡ {speed_mb_s:.2f} MB/s"
                
                previous_size[dataset_name] = size
                previous_time[dataset_name] = time.time()
                
                print(f"📦 {dataset_name.upper()}:")
                print(f"   📄 .dat: {dat_count}/{info['expected_records']} ({dat_pct:.1f}%)")
                print(f"   📄 .hea: {hea_count}/{info['expected_records']}")
                print(f"   📄 .atr: {atr_count}/{info['expected_records']}")
                print(f"   📊 Tamaño: {size:.2f} GB / {info['expected_gb']:.1f} GB ({progress_pct:.1f}%){speed_str}")
                print(f"   {print_progress_bar(progress_pct)}")
                print()
            
            total_expected_size = 7.0
            progress_pct = (total_size / total_expected_size) * 100 if total_expected_size > 0 else 0
            dat_pct = (total_dat / total_expected_dat) * 100 if total_expected_dat > 0 else 0
            
            # Calcular velocidad total
            total_speed_str = ""
            if 'total' in previous_size:
                time_diff = time.time() - previous_time.get('total', time.time())
                if time_diff > 0:
                    size_diff = total_size - previous_size['total']
                    speed_mb_s = (size_diff / time_diff) * 1024
                    if speed_mb_s > 0:
                        remaining_gb = total_expected_size - total_size
                        eta_minutes = (remaining_gb * 1024) / speed_mb_s / 60 if speed_mb_s > 0 else 0
                        total_speed_str = f" | ⚡ {speed_mb_s:.2f} MB/s | ⏱️  {eta_minutes:.0f} min restantes"
            
            previous_size['total'] = total_size
            previous_time['total'] = time.time()
            
            print("=" * 70)
            print("📈 RESUMEN GENERAL")
            print("=" * 70)
            print(f"📄 Archivos .dat (CRÍTICOS): {total_dat}/{total_expected_dat} ({dat_pct:.1f}%)")
            print(f"📄 Archivos .hea: {total_hea}/{total_expected_dat}")
            print(f"📄 Archivos .atr: {total_atr}/{total_expected_dat}")
            print(f"📊 Tamaño total: {total_size:.2f} GB / {total_expected_size:.1f} GB ({progress_pct:.1f}%){total_speed_str}")
            print(f"   {print_progress_bar(progress_pct)}")
            print()
            
            # Archivos omitidos
            skipped_count = get_skipped_count()
            if skipped_count > 0:
                print(f"⚠️  Archivos omitidos: {skipped_count} archivos .atr (no críticos)")
                print("   ✅ Sistema de omisión funcionando correctamente")
                print()
            
            # Estado de procesos
            if python_count > 0 or aria2c_count > 0:
                print("✅ PROCESOS ACTIVOS - La descarga está avanzando")
            else:
                print("⚠️  NO HAY PROCESOS ACTIVOS")
                print("   El script debería reiniciarse automáticamente")
            
            print()
            print("💡 Presiona Ctrl+C para salir")
            print("   Próxima actualización en 10 segundos...")
            
            iteration += 1
            time.sleep(10)
            
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("⏹️  Monitor detenido")
        print("=" * 70)
        print()
        print("📊 Estado final:")
        print(f"   Archivos .dat: {total_dat}/{total_expected_dat} ({dat_pct:.1f}%)")
        print(f"   Tamaño total: {total_size:.2f} GB / {total_expected_size:.1f} GB ({progress_pct:.1f}%)")
        if skipped_count > 0:
            print(f"   Archivos omitidos: {skipped_count}")
        print()

if __name__ == "__main__":
    main()

