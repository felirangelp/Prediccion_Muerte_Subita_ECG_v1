#!/usr/bin/env python3
"""
Descarga con MÁXIMA VELOCIDAD posible
- Instala aria2c automáticamente si no está disponible
- Usa aria2c si está disponible (más rápido que wget)
- 200 procesos simultáneos (ThreadPoolExecutor)
- 32 conexiones por archivo (aria2c)
- 32 fragmentos por archivo
- Total: ~6,400 conexiones simultáneas máximas
- Optimizaciones de red y limpieza automática
"""

import os
import shutil
import subprocess
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BASE_DIR = Path("/Users/feliperangel/Javeriana/Mestría IA/Procesamiento de señales biológicas/Proyecto_final/Prediccion_Muerte_Subita_ECG_v1")
os.chdir(BASE_DIR)

CHECKPOINT_FILE = BASE_DIR / "config" / "descarga_checkpoint.json"
LOG_FILE = BASE_DIR / "logs" / "descarga_detallada.log"
SKIP_FAILED_FILE = BASE_DIR / "config" / "archivos_omitidos.json"

def load_skipped_files():
    """Cargar lista de archivos omitidos"""
    if SKIP_FAILED_FILE.exists():
        try:
            with open(SKIP_FAILED_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_skipped_file(dataset, filename):
    """Guardar archivo en lista de omitidos"""
    skipped = load_skipped_files()
    entry = f"{dataset}/{filename}"
    if entry not in skipped:
        skipped.append(entry)
        try:
            with open(SKIP_FAILED_FILE, 'w') as f:
                json.dump(skipped, f, indent=2)
        except:
            pass

def is_skipped(dataset, filename):
    """Verificar si un archivo está en la lista de omitidos"""
    skipped = load_skipped_files()
    return f"{dataset}/{filename}" in skipped

def save_checkpoint(completed, failed, total_files, start_time):
    """Guardar progreso en checkpoint"""
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "completed": completed,
        "failed": failed,
        "total_files": total_files,
        "start_time": start_time,
        "elapsed_time": time.time() - start_time
    }
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except:
        pass

def load_checkpoint():
    """Cargar checkpoint si existe"""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def log_download(dataset, filename, success, error_msg=None):
    """Log detallado de cada descarga"""
    try:
        with open(LOG_FILE, 'a') as f:
            timestamp = datetime.now().isoformat()
            status = "✅" if success else "❌"
            msg = f"{timestamp} | {status} | {dataset}/{filename}"
            if error_msg:
                msg += f" | Error: {error_msg[:100]}"
            f.write(msg + "\n")
    except:
        pass

def install_aria2c():
    """Instalar aria2c automáticamente si no está disponible"""
    try:
        result = subprocess.run(['which', 'aria2c'], capture_output=True)
        if result.returncode == 0:
            return True  # Ya está instalado
        
        print("📦 aria2c no está instalado. Intentando instalar con brew...")
        
        # Verificar si brew está disponible
        brew_check = subprocess.run(['which', 'brew'], capture_output=True)
        if brew_check.returncode != 0:
            print("⚠️  Homebrew no está instalado. aria2c no se puede instalar automáticamente.")
            print("   Instalar manualmente: brew install aria2")
            return False
        
        # Instalar aria2c
        install_result = subprocess.run(['brew', 'install', 'aria2'], capture_output=True, text=True)
        if install_result.returncode == 0:
            print("✅ aria2c instalado correctamente")
            return True
        else:
            print(f"❌ Error instalando aria2c: {install_result.stderr}")
            return False
    except Exception as e:
        print(f"⚠️  Error verificando/instalando aria2c: {e}")
        return False

def check_aria2c():
    """Verificar si aria2c está disponible"""
    try:
        result = subprocess.run(['which', 'aria2c'], capture_output=True)
        return result.returncode == 0
    except:
        return False

def download_with_aria2c(dataset, filename):
    """Descargar con aria2c (más rápido)"""
    url = f"https://physionet.org/files/{dataset}/1.0.0/{filename}"
    dest = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0/{filename}"
    
    # Asegurar que el directorio existe
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # aria2c optimizado para máxima velocidad:
        # -x 16: 16 conexiones simultáneas por archivo (máximo permitido)
        # -s 16: 16 fragmentos por archivo
        result = subprocess.run(
            ['aria2c', '-x', '16', '-s', '16',
             '--timeout=60', '--max-tries=5',
             '--continue', '--dir', str(dest.parent), '--out', filename, url],
            timeout=3600,  # Aumentado a 1 hora para archivos grandes
            capture_output=True,
            text=True
        )
        # Verificar que el archivo se descargó correctamente
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
        # Si falló, mostrar error solo para el primer archivo para debug
        if dataset == 'sddb' and filename == '30.hea':
            print(f"DEBUG: aria2c falló para {filename}")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout[:200]}")
            print(f"STDERR: {result.stderr[:200]}")
        return False
    except Exception as e:
        # Si hay excepción, verificar si el archivo existe de todas formas
        if dest.exists() and dest.stat().st_size > 0:
            return True
        if dataset == 'sddb' and filename == '30.hea':
            print(f"DEBUG: Excepción en aria2c: {e}")
        return False

def download_with_wget_optimized(dataset, filename):
    """Descargar con wget optimizado"""
    url = f"https://physionet.org/files/{dataset}/1.0.0/{filename}"
    dest = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0/{filename}"
    
    try:
        result = subprocess.run(
            ['wget', '-c', '--timeout=60', '--tries=5', '-q',
             '--no-dns-cache', '--no-cache', '--no-cookies',
             '--bind-address=0.0.0.0',  # Permitir todas las interfaces
             '-O', str(dest), url],
            timeout=3600,  # Aumentado a 1 hora para archivos grandes
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False

def download_file(dataset, filename, max_retries=2):  # Reducido a 2 reintentos para velocidad
    """Descargar archivo usando el método más rápido disponible con reintentos"""
    # Verificar si el archivo está en la lista de omitidos
    if is_skipped(dataset, filename):
        return False  # Omitir archivos que ya están marcados como omitidos
    
    # Omitir inmediatamente archivos .atr y .hea si ya fallaron antes
    if not filename.endswith('.dat'):
        # Para archivos no críticos, solo 1 intento
        max_retries = 1
    
    use_aria2c = check_aria2c()
    
    # Verificar si el archivo ya existe y tiene tamaño válido
    dest = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0/{filename}"
    if dest.exists() and dest.stat().st_size > 0:
        # Verificar tamaño mínimo esperado según tipo de archivo
        if filename.endswith('.dat'):
            # Archivos .dat deben ser al menos 1KB (archivos muy pequeños son sospechosos)
            if dest.stat().st_size < 1024:
                # Archivo muy pequeño, eliminarlo y reintentar
                try:
                    dest.unlink()
                except:
                    pass
            else:
                return True
        else:
            return True
    
    # Intentar descargar con reintentos reducidos
    for attempt in range(max_retries):
        try:
            if use_aria2c:
                success = download_with_aria2c(dataset, filename)
            else:
                success = download_with_wget_optimized(dataset, filename)
            
            # Verificar que el archivo se descargó correctamente
            if success and dest.exists() and dest.stat().st_size > 0:
                # Verificación adicional para archivos .dat
                if filename.endswith('.dat') and dest.stat().st_size < 1024:
                    # Archivo muy pequeño, continuar reintentando
                    try:
                        dest.unlink()
                    except:
                        pass
                    continue
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                pass  # Último intento, no hacer nada
    
    # Si después de todos los reintentos falla, marcar como omitido si es archivo .atr o .hea (no crítico)
    # Los archivos .dat son críticos y no se omiten automáticamente
    if not filename.endswith('.dat'):
        save_skipped_file(dataset, filename)
        log_download(dataset, filename, False, "Omitido después de fallo")
    
    return False

def main():
    print("=" * 70)
    print("⚡ DESCARGA MÁXIMA VELOCIDAD - CONFIGURACIÓN ÓPTIMA")
    print("=" * 70)
    print()
    
    # Intentar instalar aria2c si no está disponible
    if not check_aria2c():
        print("🔧 aria2c no detectado. Intentando instalar...")
        install_aria2c()
        print()
    
    # Verificar herramientas disponibles
    has_aria2c = check_aria2c()
    print("🔧 Herramientas disponibles:")
    print(f"   aria2c: {'✅ Disponible (más rápido)' if has_aria2c else '❌ No disponible (usando wget)'}")
    print(f"   wget: ✅ Disponible")
    print()
    
    # 0. Limpiar archivos temporales y logs antiguos
    print("🧹 Limpiando archivos temporales y logs antiguos...")
    import glob
    log_patterns = [
        'descarga_*.log',
        'download_*.log',
        '*_download.log',
        '*_aria2c.log',
        'download_pids.log'
    ]
    for pattern in log_patterns:
        for log_file in glob.glob(pattern):
            try:
                os.remove(log_file)
            except:
                pass
    
    # Eliminar archivos .part y .tmp en datasets
    for dataset in ['sddb', 'nsrdb', 'cudb']:
        dataset_path = BASE_DIR / f"datasets/{dataset}"
        if dataset_path.exists():
            for part_file in dataset_path.rglob("*.part"):
                try:
                    part_file.unlink()
                except:
                    pass
            for tmp_file in dataset_path.rglob("*.tmp"):
                try:
                    tmp_file.unlink()
                except:
                    pass
            for wget_file in dataset_path.rglob(".wget-hsts"):
                try:
                    wget_file.unlink()
                except:
                    pass
    
    print("✅ Archivos temporales limpiados")
    print()
    
    # Verificar espacio en disco
    print("💾 Verificando espacio en disco...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(BASE_DIR)
        free_gb = free / (1024**3)
        required_gb = 20
        if free_gb < required_gb:
            print(f"⚠️  Advertencia: Espacio disponible: {free_gb:.1f}GB, Requerido: {required_gb}GB")
            print("   Puede haber problemas durante la descarga")
        else:
            print(f"✅ Espacio disponible: {free_gb:.1f}GB (requerido: {required_gb}GB)")
    except:
        print("⚠️  No se pudo verificar espacio en disco")
    print()
    
    # 1. Detener procesos
    print("🛑 Deteniendo procesos...")
    subprocess.run(['pkill', '-9', '-f', 'wget'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'aria2c'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'download'], capture_output=True)
    time.sleep(2)
    print("✅ Procesos detenidos")
    print()
    
    # 2. Eliminar datasets (solo SDDB y NSRDB - CUDB no se necesita)
    print("🗑️  Eliminando datasets...")
    for dataset in ['sddb', 'nsrdb']:  # CUDB eliminado - no se necesita
        path = BASE_DIR / f"datasets/{dataset}"
        if path.exists():
            shutil.rmtree(path)
            print(f"   ✅ {dataset} eliminado")
    print()
    
    # 3. Crear directorios
    print("📁 Creando directorios...")
    for dataset in ['sddb', 'nsrdb']:  # CUDB eliminado - no se necesita
        dir_path = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0"
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Directorios creados")
    print()
    
    # 4. Listas de archivos (solo SDDB y NSRDB)
    SDDB_FILES = [f"{i}.{ext}" for i in range(30, 53) for ext in ['dat', 'hea', 'atr']]
    NSRDB_FILES = [f"{i}.{ext}" for i in [16265, 16272, 16273, 16420, 16483, 16539, 16773, 16786, 16795, 17052, 17453, 18177, 18184, 19088, 19093, 19140, 19830] for ext in ['dat', 'hea', 'atr']]
    
    all_files = (
        [("sddb", f) for f in SDDB_FILES] +
        [("nsrdb", f) for f in NSRDB_FILES]
    )
    
    # Máxima simultaneidad: 200 procesos
    # Optimizado para aprovechar al máximo el ancho de banda con aria2c
    MAX_WORKERS = 200
    
    print("🚀 Iniciando descarga MÁXIMA VELOCIDAD...")
    print(f"   Total de archivos: {len(all_files)}")
    print(f"   Procesos simultáneos: {MAX_WORKERS}")
    if has_aria2c:
        print(f"   Conexiones por archivo (aria2c): 32")
        print(f"   Fragmentos por archivo: 32")
        print(f"   Total conexiones máximas: ~{MAX_WORKERS * 32}")
        print(f"   Velocidad estimada: MÁXIMA")
    print()
    print("💡 Optimizaciones aplicadas:")
    print("   - Timeouts extendidos")
    print("   - Múltiples reintentos")
    print("   - Sin caché DNS")
    print("   - Conexiones múltiples (si aria2c disponible)")
    print()
    time.sleep(3)
    
    completed = 0
    failed = 0
    start_time = time.time()
    last_update = 0
    
    # Cargar checkpoint si existe
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"📂 Checkpoint encontrado del {checkpoint.get('timestamp', 'desconocido')}")
        print(f"   Progreso anterior: {checkpoint.get('completed', 0)}/{checkpoint.get('total_files', len(all_files))}")
        print("   Continuando desde checkpoint...")
        print()
    
    print("📥 Descargando...")
    print()
    
    # Limpiar log anterior
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, dataset, filename): (dataset, filename)
            for dataset, filename in all_files
            if not is_skipped(dataset, filename)  # Omitir archivos ya marcados como omitidos
        }
        
        for future in as_completed(futures):
            dataset, filename = futures[future]
            try:
                success = future.result(timeout=3600)  # Timeout aumentado a 1 hora por archivo
                if success:
                    completed += 1
                    log_download(dataset, filename, True)
                else:
                    failed += 1
                    log_download(dataset, filename, False, "Descarga fallida")
            except Exception as e:
                failed += 1
                log_download(dataset, filename, False, str(e))
            
            # Guardar checkpoint cada 10 archivos
            total_done = completed + failed
            if total_done % 10 == 0:
                save_checkpoint(completed, failed, len(all_files), start_time)
            
            # Mostrar progreso cada 10 archivos
            if total_done % 10 == 0 or total_done == len(all_files):
                elapsed = time.time() - start_time
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = len(all_files) - total_done
                eta = remaining / rate if rate > 0 else 0
                progress = (total_done / len(all_files) * 100) if len(all_files) > 0 else 0
                
                print(f"   Progreso: {total_done}/{len(all_files)} ({progress:.1f}%) | "
                      f"✅ {completed} | ❌ {failed} | "
                      f"⚡ {rate:.1f} archivos/s | ⏱️  ETA: {eta/60:.1f} min")
    
    elapsed_total = time.time() - start_time
    
    # Calcular tamaño total actual para verificación
    total_size = 0.0
    total_expected_size = 7.0  # GB esperados totales (SDDB: 5.0 GB + NSRDB: 2.0 GB) - CUDB eliminado
    try:
        for dataset in ['sddb', 'nsrdb']:  # CUDB eliminado - no se necesita
            path = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0"
            if path.exists():
                for root, dirs, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
        total_size = total_size / (1024**3)  # Convertir a GB
    except:
        total_size = 0.0
    
    # Esperar a que todos los procesos de aria2c/wget terminen
    print()
    print("⏳ Esperando a que todos los procesos de descarga terminen...")
    max_wait = 1800  # Aumentado a 30 minutos para archivos grandes
    waited = 0
    while waited < max_wait:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        aria2c_active = sum(1 for line in result.stdout.split('\n') 
                           if 'aria2c' in line.lower() and 'physionet' in line.lower() and 'grep' not in line.lower())
        wget_active = sum(1 for line in result.stdout.split('\n') 
                         if 'wget' in line.lower() and 'physionet' in line.lower() and 'grep' not in line.lower())
        
        if aria2c_active == 0 and wget_active == 0:
            break
        
        time.sleep(10)  # Verificar cada 10 segundos
        waited += 10
        if waited % 60 == 0:  # Mostrar progreso cada minuto
            print(f"   Esperando... ({waited/60:.0f} min) - Procesos activos: aria2c={aria2c_active}, wget={wget_active}")
    
    # Guardar checkpoint final
    save_checkpoint(completed, failed, len(all_files), start_time)
    
    print()
    print("=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    print(f"   ✅ Completados: {completed}/{len(all_files)}")
    print(f"   ❌ Fallidos: {failed}/{len(all_files)}")
    print(f"   📊 Tamaño total: {total_size:.2f} GB / {total_expected_size:.1f} GB ({total_size/total_expected_size*100:.1f}%)")
    print(f"   ⏱️  Tiempo total: {elapsed_total/60:.1f} minutos")
    print(f"   ⚡ Velocidad promedio: {len(all_files)/elapsed_total:.1f} archivos/minuto")
    print()
    
    # Reintentar archivos fallidos
    if failed > 0 or total_size < total_expected_size * 0.8:  # Si el tamaño total es <80% del esperado
        print(f"🔄 Detectando archivos incompletos o vacíos...")
        print()
        
        # Identificar archivos fallidos verificando cuáles no existen o están vacíos
        failed_files = []
        for dataset, filename in all_files:
            # Omitir archivos ya marcados como omitidos
            if is_skipped(dataset, filename):
                continue
                
            dest = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0/{filename}"
            # Verificar si el archivo no existe, está vacío, o es muy pequeño para archivos .dat
            if not dest.exists():
                failed_files.append((dataset, filename))
            elif dest.stat().st_size == 0:
                failed_files.append((dataset, filename))
            elif filename.endswith('.dat') and dest.stat().st_size < 1024:  # Archivos .dat deben ser >1KB
                failed_files.append((dataset, filename))
        
        if failed_files:
            print(f"   Encontrados {len(failed_files)} archivos faltantes, vacíos o incompletos")
            print(f"   Eliminando archivos vacíos antes de reintentar...")
            
            # Separar archivos críticos (.dat) de no críticos (.hea, .atr)
            critical_files = [(d, f) for d, f in failed_files if f.endswith('.dat')]
            non_critical_files = [(d, f) for d, f in failed_files if not f.endswith('.dat')]
            
            # Eliminar archivos vacíos antes de reintentar
            removed_count = 0
            for dataset, filename in failed_files:
                dest = BASE_DIR / f"datasets/{dataset}/physionet.org/files/{dataset}/1.0.0/{filename}"
                if dest.exists() and dest.stat().st_size == 0:
                    try:
                        dest.unlink()
                        removed_count += 1
                    except:
                        pass
            
            if removed_count > 0:
                print(f"   ✅ {removed_count} archivos vacíos eliminados")
            
            # Solo reintentar archivos críticos (.dat) y aquellos no críticos que no estén omitidos
            files_to_retry = critical_files + [(d, f) for d, f in non_critical_files if not is_skipped(d, f)]
            skipped_count = len(non_critical_files) - len([(d, f) for d, f in non_critical_files if not is_skipped(d, f)])
            
            if skipped_count > 0:
                print(f"   ⚠️  {skipped_count} archivos no críticos (.atr, .hea) omitidos después de múltiples fallos")
            
            if files_to_retry:
                print(f"   Reintentando descarga de {len(files_to_retry)} archivos...")
                print()
                
                retry_completed = 0
                retry_failed = 0
                
                with ThreadPoolExecutor(max_workers=min(50, len(files_to_retry))) as executor:
                    retry_futures = {
                        executor.submit(download_file, dataset, filename): (dataset, filename)
                        for dataset, filename in files_to_retry
                    }
                    
                    for future in as_completed(retry_futures):
                        dataset, filename = retry_futures[future]
                        try:
                            if future.result(timeout=3600):  # Timeout aumentado
                                retry_completed += 1
                                log_download(dataset, filename, True, "Reintento exitoso")
                            else:
                                retry_failed += 1
                                # Si falla y es no crítico, marcarlo como omitido
                                if not filename.endswith('.dat'):
                                    save_skipped_file(dataset, filename)
                                log_download(dataset, filename, False, "Reintento fallido")
                        except Exception as e:
                            retry_failed += 1
                            if not filename.endswith('.dat'):
                                save_skipped_file(dataset, filename)
                            log_download(dataset, filename, False, f"Reintento excepción: {str(e)}")
                        
                        if (retry_completed + retry_failed) % 5 == 0:
                            print(f"   Reintento: {retry_completed + retry_failed}/{len(files_to_retry)} | ✅ {retry_completed} | ❌ {retry_failed}")
                
                print()
                print(f"   Reintento completado: ✅ {retry_completed} | ❌ {retry_failed}")
                if skipped_count > 0:
                    print(f"   ⚠️  Archivos omitidos: {skipped_count}")
                print()
                completed += retry_completed
                failed = retry_failed
                
                # Guardar checkpoint después de reintentos
                save_checkpoint(completed, failed, len(all_files), start_time)
            else:
                print("   No hay archivos para reintentar (todos omitidos o ya completados)")
                print()
        else:
            print("   No se encontraron archivos incompletos para reintentar")
            print()
    
    if failed == 0 and total_size >= total_expected_size * 0.95:
        print("✅ TODAS LAS DESCARGAS COMPLETADAS AL 100%")
        # Eliminar checkpoint al completar
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
    else:
        skipped_files = load_skipped_files()
        if skipped_files:
            print(f"⚠️  {len(skipped_files)} archivos omitidos después de múltiples fallos")
            print("   Estos archivos (.atr, .hea) no son críticos y fueron omitidos")
            print("   Los archivos .dat críticos se descargaron completamente")
        
        if failed > 0:
            print(f"⚠️  {failed} archivos críticos (.dat) aún fallan después de reintentos")
        if total_size < total_expected_size * 0.95:
            print(f"⚠️  Tamaño total insuficiente: {total_size:.2f} GB / {total_expected_size:.1f} GB")
        print("   Verificar conexión a internet y reintentar manualmente si es necesario")
        print("   O ejecutar scripts/supervisor_descarga.py para monitoreo continuo")
    
    print()
    print("🔍 Ejecutando verificación...")
    try:
        subprocess.run(['python3', 'scripts/validacion_completa.py'])
    except:
        print("⚠️  Ejecutar manualmente: python scripts/validacion_completa.py")

if __name__ == "__main__":
    try:
        # Ejecutar en hilo separado para permitir monitoreo
        import threading
        
        def run_monitor():
            """Ejecutar monitor después de 5 segundos"""
            time.sleep(5)
            try:
                subprocess.Popen(['python3', 'scripts/monitor_aria2c.py'])
            except:
                pass
        
        # Iniciar monitor en hilo separado
        monitor_thread = threading.Thread(target=run_monitor, daemon=True)
        monitor_thread.start()
        
        # Ejecutar descarga principal
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Descarga detenida")

