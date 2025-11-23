#!/usr/bin/env python3
"""
Script para monitorear el progreso de la validación cruzada en tiempo real
con porcentaje y barra visual
"""
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime


def clear_screen():
    """Limpiar pantalla"""
    os.system('clear' if os.name != 'nt' else 'cls')


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
                parts = lines[1].strip().split()
                if len(parts) >= 4:
                    return {
                        'etime': parts[0],
                        'cpu': float(parts[1]),
                        'mem': float(parts[2]),
                        'rss': int(parts[3]) / 1024  # MB
                    }
        return None
    except:
        return None


def parse_time(etime_str):
    """Parsea el tiempo transcurrido a horas"""
    try:
        parts = etime_str.split('-')
        if len(parts) == 2:
            days = int(parts[0])
            time_part = parts[1]
        else:
            days = 0
            time_part = parts[0]
        
        time_parts = time_part.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
        
        total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
        return total_hours
    except:
        return 0


def count_folds_in_logs():
    """Cuenta los folds completados en los logs"""
    fold_count = 0
    model_folds = {'sparse': 0, 'hierarchical': 0, 'hybrid': 0}
    
    # Buscar en evaluation.log
    log_files = [
        Path("logs/evaluation.log"),
        Path("logs/cross_validation.log"),
    ]
    
    for log_file in log_files:
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    # Buscar patrones de folds
                    # "Fold X/10" o "Fold X/10..."
                    import re
                    fold_matches = re.findall(r'Fold\s+(\d+)/(\d+)', content)
                    if fold_matches:
                        # Obtener el último fold mencionado
                        last_fold = max([int(m[0]) for m in fold_matches])
                        fold_count = max(fold_count, last_fold)
                    
                    # Buscar por modelo
                    if 'Sparse' in content or 'sparse' in content.lower():
                        sparse_folds = re.findall(r'Fold\s+(\d+)/(\d+).*?Sparse', content, re.IGNORECASE)
                        if sparse_folds:
                            model_folds['sparse'] = max([int(f[0]) for f in sparse_folds])
                    
                    if 'Hierarchical' in content or 'hierarchical' in content.lower():
                        hier_folds = re.findall(r'Fold\s+(\d+)/(\d+).*?Hierarchical', content, re.IGNORECASE)
                        if hier_folds:
                            model_folds['hierarchical'] = max([int(f[0]) for f in hier_folds])
                    
                    if 'Hybrid' in content or 'hybrid' in content.lower():
                        hybrid_folds = re.findall(r'Fold\s+(\d+)/(\d+).*?Hybrid', content, re.IGNORECASE)
                        if hybrid_folds:
                            model_folds['hybrid'] = max([int(f[0]) for f in hybrid_folds])
            except Exception as e:
                pass
    
    return fold_count, model_folds


def calculate_progress(total_hours, fold_count, total_folds=10, models_active=1):
    """Calcula el progreso basado en tiempo y folds"""
    # Estimación: 12 horas por fold por modelo
    estimated_hours_per_fold = 12 * models_active
    estimated_total = estimated_hours_per_fold * total_folds
    
    # Progreso basado en tiempo
    time_progress = min(100, (total_hours / estimated_total) * 100) if estimated_total > 0 else 0
    
    # Progreso basado en folds
    fold_progress = (fold_count / total_folds) * 100 if total_folds > 0 else 0
    
    # Si no detectamos folds en logs, usar principalmente el tiempo
    if fold_count == 0:
        # Si superó las estimaciones, asumir que está en fase final
        if total_hours >= estimated_total:
            # Está en fase final (consolidación/guardado)
            # Mostrar 95% para indicar que está muy cerca de terminar
            combined_progress = 95.0
        else:
            # Usar tiempo como indicador principal
            combined_progress = time_progress * 0.9  # 90% basado en tiempo
    else:
        # Si tenemos información de folds, combinar ambos
        combined_progress = (fold_progress * 0.7) + (time_progress * 0.3)
    
    return min(100, combined_progress), time_progress, fold_progress


def create_progress_bar(progress, width=50):
    """Crea una barra de progreso visual"""
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress:.1f}%"


def main():
    """Función principal"""
    # Obtener PID del proceso
    try:
        result = subprocess.run(
            ["pgrep", "-f", "evaluate_models.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            if pids:
                pid = int(pids[0])
            else:
                print("❌ No se encontró el proceso de evaluación")
                sys.exit(1)
        else:
            print("❌ No se encontró el proceso de evaluación")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error al buscar el proceso: {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("📊 MONITOREO EN TIEMPO REAL - VALIDACIÓN CRUZADA")
    print("=" * 80)
    print(f"PID: {pid}")
    print(f"Actualizando cada 5 segundos... (Ctrl+C para salir)")
    print("=" * 80)
    time.sleep(2)
    
    last_fold_count = 0
    iteration = 0
    
    try:
        while True:
            iteration += 1
            clear_screen()
            
            print("=" * 80)
            print("📊 MONITOREO EN TIEMPO REAL - VALIDACIÓN CRUZADA")
            print("=" * 80)
            print(f"PID: {pid} | Iteración: {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Verificar si el proceso está corriendo
            process_info = get_process_info(pid)
            if not process_info:
                print("\n❌ El proceso ya no está corriendo")
                break
            
            # Información del proceso
            total_hours = parse_time(process_info['etime'])
            cpu = process_info['cpu']
            mem = process_info['mem']
            rss = process_info['rss']
            
            # Contar folds
            fold_count, model_folds = count_folds_in_logs()
            
            # Detectar modelos activos (solo Sparse según análisis anterior)
            models_active = 1  # Solo Sparse está funcionando
            
            # Calcular progreso
            progress, time_progress, fold_progress = calculate_progress(
                total_hours, fold_count, total_folds=10, models_active=models_active
            )
            
            # Información general
            print(f"\n📈 PROGRESO GENERAL:")
            print(f"   {create_progress_bar(progress, width=60)}")
            print(f"   Progreso basado en folds: {fold_progress:.1f}%")
            print(f"   Progreso basado en tiempo: {time_progress:.1f}%")
            
            # Folds
            print(f"\n📋 FOLDS:")
            print(f"   Completados: {fold_count}/10")
            print(f"   Restantes: {max(0, 10 - fold_count)}")
            
            # Modelos
            print(f"\n🤖 MODELOS:")
            print(f"   Sparse: {model_folds['sparse']}/10 folds")
            if model_folds['hierarchical'] > 0:
                print(f"   Hierarchical: {model_folds['hierarchical']}/10 folds")
            if model_folds['hybrid'] > 0:
                print(f"   Hybrid: {model_folds['hybrid']}/10 folds")
            
            # Estado del proceso
            print(f"\n💻 ESTADO DEL PROCESO:")
            print(f"   Tiempo transcurrido: {process_info['etime']} (~{total_hours:.1f} horas)")
            print(f"   CPU: {cpu:.1f}%")
            print(f"   Memoria: {mem:.1f}% ({rss:.1f} MB)")
            
            # Estimación de tiempo restante
            if fold_count > 0:
                # Basado en folds
                avg_time_per_fold = total_hours / fold_count if fold_count > 0 else 12
                remaining_folds = 10 - fold_count
                estimated_remaining = avg_time_per_fold * remaining_folds
            else:
                # Basado en tiempo estimado
                estimated_remaining = max(0, (12 * 10) - total_hours)
            
            print(f"\n⏱️  ESTIMACIÓN:")
            if estimated_remaining > 0:
                if estimated_remaining < 1:
                    print(f"   Tiempo restante: ~{estimated_remaining * 60:.0f} minutos")
                elif estimated_remaining < 24:
                    print(f"   Tiempo restante: ~{estimated_remaining:.1f} horas")
                else:
                    print(f"   Tiempo restante: ~{estimated_remaining/24:.1f} días")
            else:
                print(f"   ⚠️  Superó estimaciones - Debería terminar pronto")
            
            # Verificar si completó
            cv_file = Path("results/cross_validation_results.pkl")
            if cv_file.exists():
                print(f"\n✅ VALIDACIÓN CRUZADA COMPLETADA!")
                print(f"   Archivo generado: {cv_file}")
                stat = cv_file.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"   Fecha: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            
            # Detectar cambios
            if fold_count > last_fold_count:
                print(f"\n🎉 ¡Nuevo fold completado! (Fold {fold_count})")
                last_fold_count = fold_count
            
            print(f"\n" + "=" * 80)
            print("Actualizando en 5 segundos... (Ctrl+C para salir)")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoreo detenido por el usuario")
        print("=" * 80)
    
    # Verificación final
    cv_file = Path("results/cross_validation_results.pkl")
    if cv_file.exists():
        print("\n✅ ¡VALIDACIÓN CRUZADA COMPLETADA!")
        print(f"   El archivo {cv_file} fue generado")
    else:
        print("\n⏳ El proceso continúa ejecutándose")
        print("   Puedes ejecutar este script de nuevo para monitorear")


if __name__ == "__main__":
    main()

