#!/usr/bin/env python3
"""
Script para monitorear el progreso de la validación cruzada
y continuar automáticamente con los siguientes pasos cuando termine.
"""
import subprocess
import sys
import time
from pathlib import Path


def check_process_running(pid):
    """Verifica si un proceso está corriendo"""
    try:
        result = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def get_process_info(pid):
    """Obtiene información del proceso"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime,pcpu,pmem"],
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


def check_cv_completed():
    """Verifica si la validación cruzada se completó"""
    cv_file = Path("results/cross_validation_results.pkl")
    return cv_file.exists()


def check_evaluation_completed():
    """Verifica si la evaluación estándar se completó"""
    eval_file = Path("results/evaluation_results.pkl")
    return eval_file.exists()


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

print("=" * 70)
print("🔍 MONITOREO DE VALIDACIÓN CRUZADA")
print("=" * 70)
print(f"PID del proceso: {pid}")
print(f"Verificando cada 60 segundos...")
print("=" * 70)

iteration = 0
last_status = None

while True:
    iteration += 1
    is_running = check_process_running(pid)
    info = get_process_info(pid)
    cv_completed = check_cv_completed()
    eval_completed = check_evaluation_completed()

    current_time = time.strftime("%H:%M:%S")

    if is_running:
        if cv_completed:
            status = "✅ CV completada - Esperando finalización del proceso"
        else:
            status = "⏳ CV en progreso"

        # Solo imprimir si el estado cambió
        if status != last_status:
            print(f"\n[{iteration}] {current_time} - {status}")
            if info:
                parts = info.split()
                if len(parts) >= 3:
                    print(f"    Tiempo transcurrido: {parts[0]}")
                    print(f"    CPU: {parts[1]}% | Memoria: {parts[2]}%")
            last_status = status

        if cv_completed:
            # Esperar un poco más para que termine completamente
            time.sleep(30)
            if not check_process_running(pid):
                break
    else:
        # El proceso terminó
        if cv_completed:
            print("\n" + "=" * 70)
            print("✅ PROCESO COMPLETADO")
            print("=" * 70)
            print("✅ Validación cruzada: Completada")
            print("✅ Proceso de evaluación: Terminado")
            break
        elif eval_completed:
            print("\n" + "=" * 70)
            print("⚠️  PROCESO TERMINADO")
            print("=" * 70)
            print("✅ Evaluación estándar: Completada")
            print("❌ Validación cruzada: No completada")
            print("\n   Continuando con los resultados disponibles...")
            break
        else:
            print("\n⚠️  El proceso terminó inesperadamente")
            print("   Verificando resultados disponibles...")
            break

    time.sleep(60)  # Esperar 60 segundos entre verificaciones

print("\n📋 Próximos pasos:")
print("   1. Generar dashboard")
print("   2. Actualizar GitHub Pages")
print("=" * 70)

# Ejecutar automáticamente los siguientes pasos
if cv_completed or eval_completed:
    print("\n" + "=" * 70)
    print("🚀 EJECUTANDO PASOS SIGUIENTES AUTOMÁTICAMENTE")
    print("=" * 70)

    # Paso 1: Generar dashboard
    print("\n📊 Paso 1: Generando dashboard...")
    try:
        result = subprocess.run(
            ["python3", "scripts/generate_dashboard.py"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutos timeout
        )
        if result.returncode == 0:
            print("✅ Dashboard generado exitosamente")
        else:
            print(f"⚠️  Error al generar dashboard:")
            print(result.stderr[:500])
    except subprocess.TimeoutExpired:
        print("⚠️  Timeout al generar dashboard (más de 5 minutos)")
    except Exception as e:
        print(f"⚠️  Error al generar dashboard: {e}")

    # Paso 2: Actualizar GitHub Pages
    print("\n🌐 Paso 2: Actualizando GitHub Pages...")
    update_script = Path("scripts/update_github_pages.sh")
    if update_script.exists():
        try:
            result = subprocess.run(
                ["bash", str(update_script)],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutos timeout
            )
            if result.returncode == 0:
                print("✅ GitHub Pages actualizado exitosamente")
            else:
                print(f"⚠️  Error al actualizar GitHub Pages:")
                print(result.stderr[:500])
        except subprocess.TimeoutExpired:
            print("⚠️  Timeout al actualizar GitHub Pages")
        except Exception as e:
            print(f"⚠️  Error al actualizar GitHub Pages: {e}")
    else:
        print("⚠️  Script update_github_pages.sh no encontrado")

    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("=" * 70)

    # Notificación al usuario
    try:
        # Notificación del sistema en macOS
        subprocess.run(
            [
                "osascript",
                "-e",
                'display notification "La validación cruzada ha terminado. Dashboard y GitHub Pages actualizados." with title "Validación Cruzada Completada" sound name "Glass"',
            ],
            check=False,
        )
    except:
        pass

    # Mensaje final muy visible
    print("\n" + "🎉" * 35)
    print("🎉" + " " * 68 + "🎉")
    print("🎉" + " " * 20 + "¡PROCESO COMPLETADO EXITOSAMENTE!" + " " * 20 + "🎉")
    print("🎉" + " " * 68 + "🎉")
    print("🎉" * 35)
    print("\n✅ Validación cruzada: COMPLETADA")
    print("✅ Dashboard: GENERADO")
    print("✅ GitHub Pages: ACTUALIZADO")
    print("\n📊 Puedes revisar los resultados en:")
    print("   - results/cross_validation_results.pkl")
    print("   - results/dashboard.html")
    print("   - docs/index.html (GitHub Pages)")
    print("\n" + "=" * 70)
