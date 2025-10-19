#!/usr/bin/env python3
"""
Script para verificar la integridad y mostrar información de los datasets descargados
"""

import os
import wfdb
from pathlib import Path
import pandas as pd
from datetime import timedelta

def get_dataset_info(db_name, db_path):
    """Obtener información detallada de un dataset"""
    if not db_path.exists():
        return None
    
    # Buscar todos los archivos .hea
    hea_files = list(db_path.glob("*.hea"))
    
    if not hea_files:
        return None
    
    records_info = []
    
    for hea_file in hea_files:
        try:
            record_name = hea_file.stem
            
            # Leer metadatos del registro
            record = wfdb.rdheader(str(db_path / record_name))
            
            # Información básica
            info = {
                'record_name': record_name,
                'fs': record.fs,  # Frecuencia de muestreo
                'sig_len': record.sig_len,  # Longitud de la señal
                'n_sig': record.n_sig,  # Número de señales
                'sig_name': record.sig_name,  # Nombres de las señales
                'duration_hours': record.sig_len / record.fs / 3600 if record.fs > 0 else 0
            }
            
            records_info.append(info)
            
        except Exception as e:
            print(f"⚠️  Error leyendo {hea_file.name}: {str(e)}")
            continue
    
    return records_info

def format_duration(hours):
    """Formatear duración en horas a formato legible"""
    if hours < 1:
        minutes = hours * 60
        return f"{minutes:.1f} min"
    elif hours < 24:
        return f"{hours:.1f} h"
    else:
        days = hours / 24
        return f"{days:.1f} días"

def print_dataset_summary(db_name, db_description, records_info):
    """Imprimir resumen de un dataset"""
    if not records_info:
        print(f"❌ {db_description} ({db_name}): No disponible")
        return
    
    print(f"\n📊 {db_description} ({db_name})")
    print(f"{'='*60}")
    
    # Estadísticas generales
    total_records = len(records_info)
    total_duration = sum(r['duration_hours'] for r in records_info)
    avg_duration = total_duration / total_records if total_records > 0 else 0
    
    # Frecuencias de muestreo únicas
    unique_fs = list(set(r['fs'] for r in records_info))
    
    print(f"📁 Registros encontrados: {total_records}")
    print(f"⏱️  Duración total: {format_duration(total_duration)}")
    print(f"📈 Duración promedio: {format_duration(avg_duration)}")
    print(f"🔊 Frecuencias de muestreo: {unique_fs} Hz")
    
    # Información detallada por registro
    print(f"\n📋 Detalles por registro:")
    print(f"{'Registro':<12} {'Duración':<12} {'FS (Hz)':<8} {'Señales':<8}")
    print(f"{'-'*50}")
    
    for record in sorted(records_info, key=lambda x: x['record_name']):
        duration_str = format_duration(record['duration_hours'])
        signals_str = ', '.join(record['sig_name']) if record['sig_name'] else 'N/A'
        
        print(f"{record['record_name']:<12} {duration_str:<12} {record['fs']:<8} {record['n_sig']:<8}")
    
    return {
        'total_records': total_records,
        'total_duration_hours': total_duration,
        'avg_duration_hours': avg_duration,
        'sampling_frequencies': unique_fs
    }

def check_file_integrity(db_path):
    """Verificar integridad de archivos en un dataset"""
    if not db_path.exists():
        return False, "Directorio no existe"
    
    hea_files = list(db_path.glob("*.hea"))
    dat_files = list(db_path.glob("*.dat"))
    atr_files = list(db_path.glob("*.atr"))
    
    missing_files = []
    
    for hea_file in hea_files:
        record_name = hea_file.stem
        
        # Verificar archivo .dat correspondiente
        dat_file = db_path / f"{record_name}.dat"
        if not dat_file.exists():
            missing_files.append(f"{record_name}.dat")
        
        # Verificar archivo .atr correspondiente
        atr_file = db_path / f"{record_name}.atr"
        if not atr_file.exists():
            missing_files.append(f"{record_name}.atr")
    
    if missing_files:
        return False, f"Archivos faltantes: {', '.join(missing_files)}"
    
    return True, f"Integridad verificada: {len(hea_files)} registros completos"

def main():
    """Función principal para verificar todos los datasets"""
    print("🔍 Verificando datasets de Predicción de Muerte Súbita ECG")
    print("=" * 70)
    
    # Configuración de datasets
    datasets_config = [
        {
            "name": "sddb",
            "description": "MIT-BIH Sudden Cardiac Death Holter Database",
            "path": Path("datasets/sddb")
        },
        {
            "name": "nsrdb",
            "description": "MIT-BIH Normal Sinus Rhythm Database", 
            "path": Path("datasets/nsrdb")
        },
        {
            "name": "cudb",
            "description": "CU Ventricular Tachyarrhythmia Database",
            "path": Path("datasets/cudb")
        }
    ]
    
    summary_data = {}
    
    # Verificar cada dataset
    for dataset in datasets_config:
        print(f"\n🔍 Verificando {dataset['description']}...")
        
        # Verificar integridad de archivos
        integrity_ok, integrity_msg = check_file_integrity(dataset['path'])
        print(f"📁 Integridad: {integrity_msg}")
        
        # Obtener información detallada
        records_info = get_dataset_info(dataset['name'], dataset['path'])
        
        if records_info:
            summary = print_dataset_summary(
                dataset['name'], 
                dataset['description'], 
                records_info
            )
            summary_data[dataset['name']] = summary
        else:
            print(f"❌ No se pudo obtener información de {dataset['name']}")
    
    # Resumen general
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN GENERAL")
    print(f"{'='*70}")
    
    total_records = sum(data['total_records'] for data in summary_data.values())
    total_duration = sum(data['total_duration_hours'] for data in summary_data.values())
    
    print(f"📁 Total de registros: {total_records}")
    print(f"⏱️  Duración total: {format_duration(total_duration)}")
    
    print(f"\n📚 Datasets disponibles:")
    for dataset in datasets_config:
        if dataset['name'] in summary_data:
            data = summary_data[dataset['name']]
            print(f"   ✅ {dataset['name']}: {data['total_records']} registros")
        else:
            print(f"   ❌ {dataset['name']}: No disponible")
    
    print(f"\n💡 Para usar los datasets en Python:")
    print(f"   import wfdb")
    print(f"   record = wfdb.rdrecord('datasets/sddb/30')")
    print(f"   signal = record.p_signal")
    
    print(f"\n📚 Referencias:")
    print(f"   • Velázquez-González et al., Sensors 2021")
    print(f"   • Huang et al., Symmetry 2025")

if __name__ == "__main__":
    main()
