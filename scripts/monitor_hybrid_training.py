#!/usr/bin/env python3
"""
Script de monitoreo en tiempo real del entrenamiento del modelo Hybrid
Muestra el porcentaje de avance basado en las etapas detectadas en el log
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Definir etapas del entrenamiento y sus porcentajes
STAGES = {
    "inicio": {
        "keywords": ["entrenando modelo híbrido", "cargando datos", "preparando datos"],
        "min_percent": 0,
        "max_percent": 5,
    },
    "sparse_wavelet": {
        "keywords": [
            "entrenando componente de representaciones dispersas",
            "wavelet",
            "diccionario wavelet",
        ],
        "min_percent": 5,
        "max_percent": 30,
    },
    "sparse_classifier": {
        "keywords": [
            "entrenando clasificador de representaciones dispersas",
            "clasificador disperso",
        ],
        "min_percent": 30,
        "max_percent": 50,
    },
    "hierarchical": {
        "keywords": [
            "entrenando componente de fusión jerárquica",
            "clasificador jerárquico",
        ],
        "min_percent": 50,
        "max_percent": 85,
    },
    "ensemble": {
        "keywords": [
            "creando ensemble",
            "combinando características",
            "entrenando ensemble",
        ],
        "min_percent": 85,
        "max_percent": 95,
    },
    "completado": {
        "keywords": [
            "modelo híbrido entrenado completamente",
            "completado",
            "entrenamiento completado",
        ],
        "min_percent": 100,
        "max_percent": 100,
    },
}


def detect_stage(log_content, process_etime=None):
    """Detectar la etapa actual basada en el contenido del log y indicadores indirectos"""
    log_lower = log_content.lower()

    # Verificar en orden inverso (más reciente primero) - mensajes directos
    for stage_name, stage_info in reversed(list(STAGES.items())):
        for keyword in stage_info["keywords"]:
            if keyword in log_lower:
                return stage_name, stage_info

    # Si no se encuentran mensajes directos, usar indicadores indirectos
    # Parsear tiempo transcurrido si está disponible
    elapsed_minutes = 0
    if process_etime:
        try:
            if "-" in process_etime:
                days_part, time_part = process_etime.split("-")
                days = int(days_part)
                time_parts = time_part.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                elapsed_minutes = days * 24 * 60 + hours * 60 + minutes
            else:
                time_parts = process_etime.split(":")
                if len(time_parts) == 3:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = int(time_parts[2])
                    elapsed_minutes = hours * 60 + minutes + seconds / 60
        except Exception:
            pass

    # Indicadores indirectos basados en el contenido del log
    if "hierarchical_fusion.py" in log_content or "polyfit" in log_lower:
        # Está en etapa Hierarchical
        return "hierarchical", STAGES["hierarchical"]
    elif "metal" in log_lower or "gpu" in log_lower or "tensorflow" in log_lower:
        # GPU inicializada, probablemente en etapas tempranas o Hierarchical
        if elapsed_minutes > 60:
            # Si lleva más de 1 hora, probablemente está en Hierarchical
            return "hierarchical", STAGES["hierarchical"]
        else:
            # Si es menos, puede estar en etapas iniciales
            return "sparse_wavelet", STAGES["sparse_wavelet"]
    elif "cargando" in log_lower or "loading" in log_lower:
        # Está cargando datos
        return "inicio", STAGES["inicio"]

    # Si no hay indicadores, inferir basándose en tiempo transcurrido
    if elapsed_minutes > 0:
        if elapsed_minutes > 300:  # Más de 5 horas
            return "hierarchical", STAGES["hierarchical"]
        elif elapsed_minutes > 120:  # Más de 2 horas
            return "sparse_classifier", STAGES["sparse_classifier"]
        elif elapsed_minutes > 30:  # Más de 30 minutos
            return "sparse_wavelet", STAGES["sparse_wavelet"]

    return None, None


def calculate_progress(
    current_stage, stage_info, log_lines, process_etime=None, process_cpu=None
):
    """Calcular porcentaje de progreso usando múltiples indicadores"""
    if current_stage is None:
        return 0.0, "Iniciando..."

    min_p = stage_info["min_percent"]
    max_p = stage_info["max_percent"]

    # Si es la etapa final, retornar 100%
    if current_stage == "completado":
        return 100.0, "Completado"

    # Calcular progreso dentro de la etapa usando múltiples indicadores
    base_progress = min_p
    stage_range = max_p - min_p

    # Indicador 1: Líneas del log relacionadas con la etapa
    stage_lines = [
        line
        for line in log_lines
        if any(kw in line.lower() for kw in stage_info["keywords"])
    ]
    lines_progress = min(0.3, len(stage_lines) * 0.05)  # Máximo 30% basado en líneas

    # Indicador 2: Tiempo transcurrido (si está disponible)
    time_progress = 0.0
    if process_etime:
        try:
            elapsed_minutes = 0
            if "-" in process_etime:
                days_part, time_part = process_etime.split("-")
                days = int(days_part)
                time_parts = time_part.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                elapsed_minutes = days * 24 * 60 + hours * 60 + minutes
            else:
                time_parts = process_etime.split(":")
                if len(time_parts) == 3:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = int(time_parts[2])
                    elapsed_minutes = hours * 60 + minutes + seconds / 60

            # Estimaciones de tiempo por etapa (en minutos)
            stage_estimates = {
                "inicio": (0, 5),
                "sparse_wavelet": (5, 60),
                "sparse_classifier": (60, 120),
                "hierarchical": (120, 480),  # 2-8 horas
                "ensemble": (480, 540),
            }

            if current_stage in stage_estimates:
                stage_min, stage_max = stage_estimates[current_stage]
                if elapsed_minutes >= stage_min:
                    # Calcular progreso dentro de la etapa basado en tiempo
                    time_in_stage = elapsed_minutes - stage_min
                    stage_duration = stage_max - stage_min
                    if stage_duration > 0:
                        time_progress = min(
                            0.5, (time_in_stage / stage_duration) * 0.5
                        )  # Máximo 50% basado en tiempo
        except Exception:
            pass

    # Indicador 3: Uso de CPU (si está disponible)
    cpu_progress = 0.0
    if process_cpu:
        try:
            cpu_usage = float(process_cpu)
            # Si CPU está alta (>50%), asumir que está procesando activamente
            if cpu_usage > 50:
                cpu_progress = 0.2  # 20% adicional si está procesando activamente
        except Exception:
            pass

    # Combinar indicadores (pesos: líneas 30%, tiempo 50%, CPU 20%)
    combined_progress = (
        (lines_progress * 0.3) + (time_progress * 0.5) + (cpu_progress * 0.2)
    )

    # Si no hay indicadores, usar un progreso mínimo conservador
    if combined_progress == 0:
        combined_progress = 0.1  # Al menos 10% dentro de la etapa

    # Calcular porcentaje total
    progress = base_progress + (stage_range * combined_progress)

    # Asegurar que no exceda el máximo de la etapa
    progress = min(progress, max_p)

    # Nombre de la etapa para mostrar
    stage_display = current_stage.replace("_", " ").title()
    if current_stage == "hierarchical":
        stage_display = "Hierarchical Fusion"
    elif current_stage == "sparse_wavelet":
        stage_display = "Sparse Wavelet"
    elif current_stage == "sparse_classifier":
        stage_display = "Sparse Classifier"

    return progress, stage_display


def get_process_info(pid):
    """Obtener información del proceso"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime,pcpu,pmem,state"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].strip().split()
                if len(parts) >= 4:
                    return {
                        "etime": parts[0],
                        "cpu": parts[1],
                        "mem": parts[2],
                        "state": parts[3],
                    }
    except:
        pass
    return None


def format_progress_bar(percent, width=50):
    """Crear barra de progreso visual"""
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent:.1f}%"


def main():
    """Función principal"""
    log_file = Path("logs/train_hybrid.log")

    print("=" * 70)
    print("📊 MONITOREO EN TIEMPO REAL - ENTRENAMIENTO HYBRID")
    print("=" * 70)
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Log: {log_file}")
    print("=" * 70)

    # Buscar proceso de entrenamiento
    result = subprocess.run(
        ["pgrep", "-f", "train_models.py.*--train-hybrid"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("⚠️  No se encontró proceso de entrenamiento Hybrid")
        print("   Verificando si el log existe...")
        if not log_file.exists():
            print("❌ El log no existe. El entrenamiento puede no haber iniciado.")
            return
    else:
        pid = int(result.stdout.strip().split("\n")[0])
        print(f"✅ Proceso encontrado (PID: {pid})")

    last_size = 0
    last_stage = None
    iteration = 0

    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")

            # Verificar si el proceso sigue activo
            process_info = None
            if result.returncode == 0:
                pid = int(result.stdout.strip().split("\n")[0])
                process_info = get_process_info(pid)
                if not process_info:
                    print(f"\n[{current_time}] ⚠️  Proceso terminado")
                    break
            else:
                # Si no hay proceso, verificar si el log sigue creciendo
                if log_file.exists():
                    current_size = log_file.stat().st_size
                    if current_size == last_size:
                        # Log no ha crecido, puede haber terminado
                        time.sleep(5)
                        if log_file.stat().st_size == last_size:
                            print(
                                f"\n[{current_time}] ⚠️  Log no ha crecido, proceso puede haber terminado"
                            )
                            break
                    last_size = current_size

            # Leer log
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    log_content = f.read()
                    log_lines = log_content.split("\n")

                # Obtener información del proceso para indicadores indirectos
                process_info = None
                if result.returncode == 0:
                    pid = int(result.stdout.strip().split("\n")[0])
                    process_info = get_process_info(pid)

                # Detectar etapa actual (usando indicadores indirectos si es necesario)
                current_stage, stage_info = detect_stage(
                    log_content,
                    process_etime=process_info["etime"] if process_info else None,
                )

                # Si no se detectó etapa, usar inicio por defecto
                if current_stage is None:
                    current_stage = "inicio"
                    stage_info = STAGES["inicio"]

                # Calcular progreso usando indicadores múltiples
                progress, stage_name = calculate_progress(
                    current_stage,
                    stage_info,
                    log_lines,
                    process_etime=process_info["etime"] if process_info else None,
                    process_cpu=process_info["cpu"] if process_info else None,
                )

                # Mostrar progreso solo si cambió
                if current_stage != last_stage or iteration % 10 == 0:
                    # Limpiar línea anterior
                    print(
                        f"\r[{current_time}] {format_progress_bar(progress)} - {stage_name}",
                        end="",
                        flush=True,
                    )

                    if process_info:
                        print(
                            f" | CPU: {process_info['cpu']}% | Mem: {process_info['mem']}% | Tiempo: {process_info['etime']}",
                            end="",
                            flush=True,
                        )

                    last_stage = current_stage

                # Si está completado, salir
                if current_stage == "completado":
                    print(f"\n\n[{current_time}] ✅ ¡ENTRENAMIENTO COMPLETADO!")
                    print("=" * 70)
                    break

                # Verificar errores
                if (
                    "error" in log_content.lower()
                    or "exception" in log_content.lower()
                    or "traceback" in log_content.lower()
                ):
                    error_lines = [
                        line
                        for line in log_lines
                        if any(
                            err in line.lower()
                            for err in ["error", "exception", "traceback"]
                        )
                    ]
                    if error_lines:
                        print(f"\n\n[{current_time}] ❌ ERROR DETECTADO:")
                        for line in error_lines[-3:]:
                            print(f"   {line[:100]}")
                        print("\n   Revisa el log completo para más detalles.")
                        break
            else:
                print(
                    f"\r[{current_time}] ⏳ Esperando inicio del entrenamiento...",
                    end="",
                    flush=True,
                )

            time.sleep(5)  # Actualizar cada 5 segundos

    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoreo detenido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error en monitoreo: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📋 RESUMEN FINAL")
    print("=" * 70)

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            final_content = f.read()
            final_stage, _ = detect_stage(final_content)

            if final_stage == "completado":
                print("✅ Entrenamiento completado exitosamente")
            else:
                print(f"⚠️  Estado final: {final_stage}")
                print("   Revisa el log para más detalles")

    print("=" * 70)


if __name__ == "__main__":
    main()
