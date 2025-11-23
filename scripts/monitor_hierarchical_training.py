#!/usr/bin/env python3
"""
Script para monitorear el progreso del entrenamiento del modelo Hierarchical
y reiniciar automáticamente si hay fallas.
"""
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import signal


def check_model_files_exist(models_dir="models"):
    """Verifica si los archivos del modelo existen"""
    model_path = Path(models_dir) / "hierarchical_classifier"
    files = [
        f"{model_path}_fusion.h5",
        f"{model_path}_tcn.h5",
        f"{model_path}_scalers.pkl",
        f"{model_path}_metadata.pkl"
    ]
    return all(Path(f).exists() for f in files)


def get_model_file_sizes(models_dir="models"):
    """Obtiene el tamaño de los archivos del modelo"""
    model_path = Path(models_dir) / "hierarchical_classifier"
    sizes = {}
    for suffix in ["_fusion.h5", "_tcn.h5", "_scalers.pkl", "_metadata.pkl"]:
        file_path = Path(f"{model_path}{suffix}")
        if file_path.exists():
            sizes[suffix] = file_path.stat().st_size
        else:
            sizes[suffix] = 0
    return sizes


def check_process_running(pid):
    """Verifica si un proceso está corriendo"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid)], 
            capture_output=True, 
            text=True
        )
        return result.returncode == 0
    except:
        return False


def get_process_info(pid):
    """Obtiene información del proceso"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime,pcpu,pmem,rss"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                return lines[1].strip()
        return None
    except:
        return None


def monitor_training(script_path, models_dir="models", data_dir="datasets", 
                     check_interval=30, max_restarts=3):
    """
    Monitorea el entrenamiento del modelo Hierarchical
    
    Args:
        script_path: Ruta al script de entrenamiento
        models_dir: Directorio de modelos
        data_dir: Directorio de datos
        check_interval: Intervalo de verificación en segundos
        max_restarts: Número máximo de reinicios
    """
    print("="*70)
    print("🔍 MONITOR DE ENTRENAMIENTO - MODELO HIERARCHICAL")
    print("="*70)
    print(f"📁 Directorio de modelos: {models_dir}")
    print(f"📁 Directorio de datos: {data_dir}")
    print(f"⏱️  Intervalo de verificación: {check_interval} segundos")
    print(f"🔄 Máximo de reinicios: {max_restarts}")
    print("="*70)
    
    restart_count = 0
    process = None
    start_time = datetime.now()
    
    while restart_count <= max_restarts:
        try:
            # Verificar estado inicial
            if check_model_files_exist(models_dir):
                print(f"\n✅ Modelo existente detectado")
                sizes = get_model_file_sizes(models_dir)
                for file, size in sizes.items():
                    if size > 0:
                        print(f"   {file}: {size / 1024:.1f} KB")
            
            # Iniciar entrenamiento
            print(f"\n🚀 Iniciando entrenamiento (intento {restart_count + 1}/{max_restarts + 1})...")
            print(f"⏰ Hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            cmd = [
                sys.executable,
                script_path,
                "--train-hierarchical",
                "--data-dir", data_dir,
                "--models-dir", models_dir
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"📊 PID del proceso: {process.pid}")
            print(f"💻 Comando: {' '.join(cmd)}")
            print("\n" + "-"*70)
            print("📝 SALIDA DEL ENTRENAMIENTO:")
            print("-"*70)
            
            # Monitorear salida en tiempo real
            last_check = time.time()
            last_sizes = get_model_file_sizes(models_dir)
            epoch_count = 0
            
            while True:
                # Leer línea de salida
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
                    sys.stdout.flush()
                    
                    # Detectar progreso de epochs
                    if "Epoch" in line and "/" in line:
                        try:
                            # Extraer número de epoch
                            parts = line.split("Epoch")
                            if len(parts) > 1:
                                epoch_part = parts[1].split("/")[0].strip()
                                epoch_count = max(epoch_count, int(epoch_part.split()[0]))
                        except:
                            pass
                    
                    # Detectar errores críticos
                    if any(keyword in line.lower() for keyword in 
                           ["error", "exception", "traceback", "failed", "killed"]):
                        print(f"\n⚠️  ERROR DETECTADO EN LA SALIDA")
                        break
                
                # Verificar si el proceso terminó
                if process.poll() is not None:
                    break
                
                # Verificar progreso periódicamente
                current_time = time.time()
                if current_time - last_check >= check_interval:
                    last_check = current_time
                    
                    # Verificar si el proceso sigue corriendo
                    if not check_process_running(process.pid):
                        print(f"\n⚠️  Proceso terminó inesperadamente (PID: {process.pid})")
                        break
                    
                    # Verificar cambios en archivos del modelo
                    current_sizes = get_model_file_sizes(models_dir)
                    if current_sizes != last_sizes:
                        print(f"\n📊 Progreso detectado - Archivos del modelo actualizados")
                        for file, size in current_sizes.items():
                            if size > last_sizes.get(file, 0):
                                print(f"   ✅ {file}: {size / 1024:.1f} KB (aumentó)")
                        last_sizes = current_sizes
                    
                    # Mostrar información del proceso
                    proc_info = get_process_info(process.pid)
                    if proc_info:
                        print(f"\n⏱️  Estado del proceso: {proc_info}")
                    if epoch_count > 0:
                        print(f"📈 Epochs completados: {epoch_count}/20")
                    
                    elapsed = datetime.now() - start_time
                    print(f"⏰ Tiempo transcurrido: {elapsed}")
            
            # Esperar a que termine completamente
            return_code = process.wait()
            
            print("\n" + "-"*70)
            print(f"🏁 Proceso terminado con código: {return_code}")
            
            # Verificar si el entrenamiento fue exitoso
            if return_code == 0 and check_model_files_exist(models_dir):
                print("\n✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
                sizes = get_model_file_sizes(models_dir)
                print("\n📦 Archivos del modelo generados:")
                for file, size in sizes.items():
                    print(f"   {file}: {size / 1024:.1f} KB")
                
                total_time = datetime.now() - start_time
                print(f"\n⏱️  Tiempo total: {total_time}")
                print("="*70)
                return True
            
            elif return_code != 0:
                print(f"\n❌ ERROR: El proceso terminó con código {return_code}")
                if restart_count < max_restarts:
                    restart_count += 1
                    wait_time = 10
                    print(f"\n⏳ Esperando {wait_time} segundos antes de reiniciar...")
                    print(f"🔄 Reinicio {restart_count}/{max_restarts}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"\n❌ MÁXIMO DE REINICIOS ALCANZADO")
                    return False
            
            else:
                print(f"\n⚠️  El proceso terminó pero los archivos del modelo no se generaron")
                if restart_count < max_restarts:
                    restart_count += 1
                    wait_time = 10
                    print(f"\n⏳ Esperando {wait_time} segundos antes de reiniciar...")
                    print(f"🔄 Reinicio {restart_count}/{max_restarts}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"\n❌ MÁXIMO DE REINICIOS ALCANZADO")
                    return False
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción por usuario (Ctrl+C)")
            if process:
                print(f"🛑 Terminando proceso {process.pid}...")
                try:
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
            return False
        
        except Exception as e:
            print(f"\n❌ ERROR INESPERADO: {e}")
            if restart_count < max_restarts:
                restart_count += 1
                wait_time = 10
                print(f"\n⏳ Esperando {wait_time} segundos antes de reiniciar...")
                print(f"🔄 Reinicio {restart_count}/{max_restarts}...")
                time.sleep(wait_time)
                continue
            else:
                return False
    
    return False


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitorear entrenamiento del modelo Hierarchical"
    )
    parser.add_argument(
        "--script",
        type=str,
        default="scripts/train_models.py",
        help="Script de entrenamiento a ejecutar"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directorio de modelos"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Directorio de datos"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Intervalo de verificación en segundos (default: 30)"
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=3,
        help="Número máximo de reinicios (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Verificar que el script existe
    if not Path(args.script).exists():
        print(f"❌ Error: Script no encontrado: {args.script}")
        sys.exit(1)
    
    # Ejecutar monitoreo
    success = monitor_training(
        args.script,
        args.models_dir,
        args.data_dir,
        args.check_interval,
        args.max_restarts
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

