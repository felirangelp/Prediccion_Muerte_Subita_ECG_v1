#!/usr/bin/env python3
"""
Script para monitorear el progreso del análisis temporal
"""

import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pickle

def check_process_running():
    """Verificar si el proceso de análisis temporal está ejecutándose"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        processes = result.stdout
        return 'analyze_temporal_intervals' in processes or 'temporal_intervals' in processes
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
    print("🔍 MONITOREO DE ANÁLISIS TEMPORAL")
    print("=" * 70)
    print()
    
    iteration = 0
    last_sddb_count = 0
    
    while True:
        iteration += 1
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Verificar proceso
        process_running = check_process_running()
        
        # Verificar resultados
        results = check_results_file()
        
        print(f"[{current_time}] Iteración {iteration}")
        
        if process_running:
            print("  ✅ Proceso ejecutándose...")
        else:
            print("  ⏸️  Proceso no encontrado")
        
        if results:
            if 'error' in results:
                print(f"  ⚠️  Error leyendo resultados: {results['error']}")
            else:
                print(f"  📊 Total segmentos: {results['total_segments']}")
                print(f"     - SDDB: {results['sddb_segments']} segmentos")
                print(f"     - NSRDB: {results['nsrdb_segments']} segmentos")
                
                if results['sddb_segments'] > last_sddb_count:
                    print(f"  🎉 ¡Nuevos datos de SDDB! (+{results['sddb_segments'] - last_sddb_count})")
                    last_sddb_count = results['sddb_segments']
                
                # Si el proceso terminó y hay datos de SDDB, avisar
                if not process_running and results['sddb_segments'] > 0:
                    print("\n" + "=" * 70)
                    print("✅ ANÁLISIS COMPLETADO CON ÉXITO")
                    print("=" * 70)
                    print(f"📊 Total de segmentos extraídos: {results['total_segments']}")
                    print(f"   - SDDB: {results['sddb_segments']} segmentos")
                    print(f"   - NSRDB: {results['nsrdb_segments']} segmentos")
                    print("\n🚀 Listo para entrenar modelos temporales")
                    return True
                
                # Si el proceso terminó pero no hay datos de SDDB
                if not process_running and results['sddb_segments'] == 0:
                    print("\n" + "=" * 70)
                    print("⚠️  ANÁLISIS COMPLETADO PERO SIN DATOS DE SDDB")
                    print("=" * 70)
                    print("El proceso terminó pero no se extrajeron segmentos de SDDB.")
                    print("Esto puede deberse a:")
                    print("  - Registros sin archivos .atr (anotaciones)")
                    print("  - Problemas con el preprocesamiento")
                    print("  - Señales inválidas después del procesamiento")
                    return False
        else:
            print("  ⏳ Archivo de resultados aún no generado...")
        
        print()
        
        # Esperar antes de la siguiente verificación
        time.sleep(10)
        
        # Limitar número de iteraciones (máximo 60 = 10 minutos)
        if iteration >= 60:
            print("\n⏱️  Tiempo máximo de monitoreo alcanzado")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoreo interrumpido por el usuario")
        sys.exit(0)

