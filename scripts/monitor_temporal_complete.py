#!/usr/bin/env python3
"""
Monitor mejorado para el análisis temporal completo
"""

import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pickle

def check_process_running(pid=None):
    """Verificar si el proceso está ejecutándose"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        processes = result.stdout
        return 'analyze_temporal_intervals' in processes
    except:
        return False

def check_results_file():
    """Verificar el archivo de resultados"""
    results_file = Path('results/temporal_intervals_data.pkl')
    if not results_file.exists():
        return None
    
    try:
        mod_time = datetime.fromtimestamp(results_file.stat().st_mtime)
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        return {
            'exists': True,
            'modified': mod_time,
            'total_segments': data.get('total_segments', 0),
            'sddb_segments': len(data.get('sddb', {}).get('segments', [])),
            'nsrdb_segments': len(data.get('nsrdb', {}).get('segments', []))
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}

def main():
    print("=" * 70)
    print("🔍 MONITOREO DE ANÁLISIS TEMPORAL COMPLETO")
    print("=" * 70)
    print("Monitoreando progreso cada 30 segundos...")
    print("Presiona Ctrl+C para detener el monitoreo (el proceso seguirá)")
    print()
    
    iteration = 0
    last_sddb_count = 0
    last_nsrdb_count = 0
    last_total = 0
    no_change_count = 0
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Verificar proceso
            process_running = check_process_running()
            
            # Verificar resultados
            results = check_results_file()
            
            print(f"[{current_time}] Verificación #{iteration}")
            
            if process_running:
                print("  ✅ Proceso ejecutándose...")
            else:
                print("  ⏸️  Proceso no encontrado")
            
            if results:
                if 'error' in results:
                    print(f"  ⚠️  Error leyendo resultados: {results['error']}")
                else:
                    total = results['total_segments']
                    sddb = results['sddb_segments']
                    nsrdb = results['nsrdb_segments']
                    time_since_update = (datetime.now() - results['modified']).total_seconds()
                    
                    print(f"  📊 Total segmentos: {total}")
                    print(f"     - SDDB: {sddb} segmentos")
                    print(f"     - NSRDB: {nsrdb} segmentos")
                    print(f"     Última actualización: hace {int(time_since_update)} segundos")
                    
                    # Detectar cambios
                    if sddb > last_sddb_count:
                        print(f"  🎉 ¡Nuevos datos de SDDB! (+{sddb - last_sddb_count})")
                        last_sddb_count = sddb
                        no_change_count = 0
                    elif nsrdb > last_nsrdb_count:
                        print(f"  🎉 ¡Nuevos datos de NSRDB! (+{nsrdb - last_nsrdb_count})")
                        last_nsrdb_count = nsrdb
                        no_change_count = 0
                    elif total > last_total:
                        print(f"  📈 Progreso: {total - last_total} nuevos segmentos")
                        last_total = total
                        no_change_count = 0
                    else:
                        no_change_count += 1
                        if no_change_count > 2:
                            print(f"  ⏳ Sin cambios recientes ({no_change_count} verificaciones)")
                    
                    # Si el proceso terminó y hay datos de SDDB
                    if not process_running:
                        print("\n" + "=" * 70)
                        if sddb > 0:
                            print("✅ ANÁLISIS COMPLETADO CON ÉXITO")
                            print("=" * 70)
                            print(f"📊 Total de segmentos extraídos: {total}")
                            print(f"   - SDDB: {sddb} segmentos")
                            print(f"   - NSRDB: {nsrdb} segmentos")
                            print("\n🚀 Listo para entrenar modelos temporales")
                            return True
                        else:
                            print("⚠️  ANÁLISIS COMPLETADO PERO SIN DATOS DE SDDB")
                            print("=" * 70)
                            return False
            else:
                print("  ⏳ Archivo de resultados aún no generado...")
            
            print()
            
            # Esperar antes de la siguiente verificación
            time.sleep(30)
            
            # Limitar número de iteraciones (máximo 120 = 60 minutos)
            if iteration >= 120:
                print("\n⏱️  Tiempo máximo de monitoreo alcanzado (60 minutos)")
                print("El proceso puede seguir ejecutándose en segundo plano")
                break
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoreo interrumpido por el usuario")
        print("El proceso de análisis continúa ejecutándose en segundo plano")
        print("Puedes verificar el progreso ejecutando:")
        print("  python3 scripts/monitor_temporal_analysis.py")
        return None

if __name__ == "__main__":
    main()

