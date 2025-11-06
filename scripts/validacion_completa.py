#!/usr/bin/env python3
"""
Script unificado de validación completa de datasets
Incluye: tamaños, checksums SHA256, carga con wfdb
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# Agregar path del proyecto
BASE_DIR = Path("/Users/feliperangel/Javeriana/Mestría IA/Procesamiento de señales biológicas/Proyecto_final/Prediccion_Muerte_Subita_ECG_v1")
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    print("⚠️  wfdb no disponible - saltando verificación de carga")
    WFDB_AVAILABLE = False
    wfdb = None

def get_size_gb(path):
    """Obtener tamaño de directorio en GB"""
    total = 0
    if path.exists():
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    return total / (1024 ** 3)

def calculate_sha256(file_path):
    """Calcular checksum SHA256 de un archivo"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return None

def verify_checksums(dataset_path):
    """Verificar checksums SHA256"""
    sha_file = dataset_path / "SHA256SUMS.txt"
    results = {
        'sha_file_exists': sha_file.exists(),
        'total_files': 0,
        'verified': 0,
        'failed': 0,
        'missing': 0,
        'errors': []
    }
    
    if not sha_file.exists():
        return results
    
    # Leer SHA256SUMS.txt
    checksums = {}
    try:
        for line in sha_file.read_text().strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    checksum = parts[0]
                    filename = parts[1]
                    checksums[filename] = checksum
    except Exception as e:
        results['errors'].append(f"Error leyendo SHA256SUMS.txt: {str(e)}")
        return results
    
    # Verificar archivos
    for filename, expected_checksum in checksums.items():
        file_path = dataset_path / filename
        results['total_files'] += 1
        
        if not file_path.exists():
            results['missing'] += 1
            results['errors'].append(f"Archivo faltante: {filename}")
            continue
        
        try:
            actual_checksum = calculate_sha256(file_path)
            if actual_checksum == expected_checksum:
                results['verified'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(f"Checksum incorrecto: {filename}")
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Error calculando checksum de {filename}: {str(e)}")
    
    return results

def verify_wfdb_loading(dataset_path, expected_records):
    """Verificar que los registros se puedan cargar con wfdb"""
    if not WFDB_AVAILABLE:
        return {
            'available': False,
            'loaded': 0,
            'failed': 0,
            'errors': []
        }
    
    results = {
        'available': True,
        'loaded': 0,
        'failed': 0,
        'errors': [],
        'sample_records': []
    }
    
    hea_files = sorted(list(dataset_path.glob("*.hea")))
    
    for hea_file in hea_files[:expected_records]:  # Verificar todos los registros esperados
        record_name = hea_file.stem
        try:
            record = wfdb.rdrecord(str(dataset_path / record_name))
            
            if record.sig_len > 0:
                results['loaded'] += 1
                duration_hours = record.sig_len / record.fs / 3600
                if len(results['sample_records']) < 3:
                    results['sample_records'].append({
                        'name': record_name,
                        'samples': record.sig_len,
                        'fs': record.fs,
                        'duration_hours': duration_hours,
                        'channels': record.n_sig
                    })
            else:
                results['failed'] += 1
                results['errors'].append(f"{record_name}: Señal vacía")
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"{record_name}: {str(e)[:60]}")
    
    return results

def verify_dataset_complete(name, path, expected_gb, expected_records):
    """Verificación completa de un dataset"""
    print(f"\n{'='*70}")
    print(f"🔍 Verificando {name}")
    print(f"{'='*70}")
    
    path_obj = Path(path)
    results = {
        'name': name,
        'path': str(path),
        'exists': path_obj.exists(),
        'size_ok': False,
        'files_ok': False,
        'checksums_ok': False,
        'wfdb_ok': False,
        'overall_ok': False,
        'details': {}
    }
    
    if not path_obj.exists():
        print(f"❌ ERROR: Directorio no existe")
        return results
    
    # 1. Verificación de tamaño
    print(f"\n📊 1. Verificación de Tamaño:")
    total_size_gb = get_size_gb(path_obj)
    percentage = (total_size_gb / expected_gb) * 100 if expected_gb > 0 else 0
    size_ok = total_size_gb >= (expected_gb * 0.95)
    
    print(f"   Tamaño actual: {total_size_gb:.4f} GB")
    print(f"   Tamaño esperado: {expected_gb:.2f} GB")
    print(f"   Porcentaje: {percentage:.2f}%")
    print(f"   Estado: {'✅ CORRECTO' if size_ok else '⚠️  PARCIAL' if percentage >= 50 else '❌ MUY BAJO'}")
    
    results['details']['size'] = {
        'actual_gb': total_size_gb,
        'expected_gb': expected_gb,
        'percentage': percentage,
        'ok': size_ok
    }
    results['size_ok'] = size_ok
    
    # 2. Verificación de archivos
    print(f"\n📄 2. Verificación de Archivos:")
    dat_files = sorted(list(path_obj.glob("*.dat")))
    hea_files = sorted(list(path_obj.glob("*.hea")))
    atr_files = sorted(list(path_obj.glob("*.atr")))
    
    files_ok = len(dat_files) == expected_records and len(hea_files) == expected_records
    
    print(f"   Archivos .dat: {len(dat_files)}/{expected_records} {'✅' if len(dat_files) == expected_records else '❌'}")
    print(f"   Archivos .hea: {len(hea_files)}/{expected_records} {'✅' if len(hea_files) == expected_records else '❌'}")
    print(f"   Archivos .atr: {len(atr_files)}/{expected_records}")
    print(f"   Estado: {'✅ COMPLETO' if files_ok else '❌ INCOMPLETO'}")
    
    results['details']['files'] = {
        'dat_count': len(dat_files),
        'hea_count': len(hea_files),
        'atr_count': len(atr_files),
        'expected': expected_records,
        'ok': files_ok
    }
    results['files_ok'] = files_ok
    
    # 3. Verificación de checksums
    print(f"\n🔐 3. Verificación de Checksums SHA256:")
    checksum_results = verify_checksums(path_obj)
    
    if checksum_results['sha_file_exists']:
        print(f"   SHA256SUMS.txt: ✅ Presente")
        print(f"   Archivos verificados: {checksum_results['total_files']}")
        print(f"   Checksums correctos: {checksum_results['verified']} ✅")
        print(f"   Checksums incorrectos: {checksum_results['failed']}")
        print(f"   Archivos faltantes: {checksum_results['missing']}")
        
        if checksum_results['failed'] == 0 and checksum_results['missing'] == 0:
            checksums_ok = True
            print(f"   Estado: ✅ TODOS LOS CHECKSUMS CORRECTOS")
        else:
            checksums_ok = False
            print(f"   Estado: ⚠️  ALGUNOS CHECKSUMS INCORRECTOS")
            if checksum_results['errors']:
                print(f"   Primeros errores:")
                for error in checksum_results['errors'][:5]:
                    print(f"      - {error}")
    else:
        print(f"   SHA256SUMS.txt: ⚠️  No encontrado")
        checksums_ok = None  # No disponible
        print(f"   Estado: ⚠️  NO DISPONIBLE")
    
    results['details']['checksums'] = checksum_results
    results['checksums_ok'] = checksums_ok
    
    # 4. Verificación de carga con wfdb
    print(f"\n🔬 4. Verificación de Carga con wfdb:")
    wfdb_results = verify_wfdb_loading(path_obj, expected_records)
    
    if not wfdb_results['available']:
        print(f"   wfdb: ⚠️  No disponible")
        print(f"   Estado: ⚠️  NO DISPONIBLE")
        wfdb_ok = None
    else:
        print(f"   Registros cargados: {wfdb_results['loaded']}/{expected_records}")
        print(f"   Registros fallidos: {wfdb_results['failed']}")
        
        if wfdb_results['sample_records']:
            print(f"   Ejemplos de registros cargados:")
            for rec in wfdb_results['sample_records']:
                print(f"      ✅ {rec['name']}: {rec['samples']:,} muestras, {rec['fs']} Hz, {rec['duration_hours']:.2f}h, {rec['channels']} canales")
        
        if wfdb_results['loaded'] == expected_records:
            wfdb_ok = True
            print(f"   Estado: ✅ TODOS LOS REGISTROS SE CARGAN CORRECTAMENTE")
        elif wfdb_results['loaded'] >= expected_records * 0.9:
            wfdb_ok = False
            print(f"   Estado: ⚠️  LA MAYORÍA SE CARGA ({wfdb_results['loaded']}/{expected_records})")
        else:
            wfdb_ok = False
            print(f"   Estado: ❌ MUCHOS REGISTROS NO SE CARGAN")
            if wfdb_results['errors']:
                print(f"   Primeros errores:")
                for error in wfdb_results['errors'][:5]:
                    print(f"      - {error}")
    
    results['details']['wfdb'] = wfdb_results
    results['wfdb_ok'] = wfdb_ok
    
    # Resumen
    print(f"\n📋 Resumen de Verificación:")
    print(f"   Tamaño: {'✅' if size_ok else '❌'}")
    print(f"   Archivos: {'✅' if files_ok else '❌'}")
    print(f"   Checksums: {'✅' if checksums_ok else '⚠️ ' if checksums_ok is None else '❌'}")
    print(f"   Carga wfdb: {'✅' if wfdb_ok else '⚠️ ' if wfdb_ok is None else '❌'}")
    
    # Determinar si está completo
    # Consideramos completo si: tamaño OK, archivos OK, y (checksums OK o wfdb OK)
    overall_ok = size_ok and files_ok and (checksums_ok is True or (checksums_ok is None and wfdb_ok is True))
    
    if overall_ok:
        print(f"\n✅ {name}: INTEGRIDAD 100% VERIFICADA")
    else:
        print(f"\n⚠️  {name}: INTEGRIDAD PARCIAL")
        issues = []
        if not size_ok:
            issues.append(f"Tamaño ({percentage:.1f}%)")
        if not files_ok:
            issues.append(f"Archivos ({len(dat_files)}/{expected_records})")
        if checksums_ok is False:
            issues.append("Checksums")
        if wfdb_ok is False:
            issues.append("Carga wfdb")
        if issues:
            print(f"   Problemas: {', '.join(issues)}")
    
    results['overall_ok'] = overall_ok
    return results

def generate_report(all_results, output_file):
    """Generar reporte final consolidado"""
    report = f"""# Reporte de Validación Completa de Datasets

**Fecha de verificación**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumen Ejecutivo

"""
    
    total_datasets = len(all_results)
    complete_datasets = sum(1 for r in all_results if r['overall_ok'])
    
    report += f"**Total de datasets verificados**: {total_datasets}\n"
    report += f"**Datasets con integridad 100%**: {complete_datasets}/{total_datasets}\n\n"
    
    if complete_datasets == total_datasets:
        report += "✅ **CONCLUSIÓN**: Todos los datasets tienen integridad 100% verificada.\n\n"
    else:
        report += "⚠️  **CONCLUSIÓN**: Algunos datasets tienen problemas de integridad.\n\n"
    
    report += "---\n\n"
    
    # Detalles por dataset
    for result in all_results:
        name = result['name']
        details = result['details']
        
        report += f"## {name}\n\n"
        
        # Tamaño
        size_info = details.get('size', {})
        report += f"### Tamaño\n"
        report += f"- Actual: {size_info.get('actual_gb', 0):.4f} GB\n"
        report += f"- Esperado: {size_info.get('expected_gb', 0):.2f} GB\n"
        report += f"- Porcentaje: {size_info.get('percentage', 0):.2f}%\n"
        report += f"- Estado: {'✅' if size_info.get('ok') else '❌'}\n\n"
        
        # Archivos
        files_info = details.get('files', {})
        report += f"### Archivos\n"
        report += f"- .dat: {files_info.get('dat_count', 0)}/{files_info.get('expected', 0)}\n"
        report += f"- .hea: {files_info.get('hea_count', 0)}/{files_info.get('expected', 0)}\n"
        report += f"- .atr: {files_info.get('atr_count', 0)}/{files_info.get('expected', 0)}\n"
        report += f"- Estado: {'✅' if files_info.get('ok') else '❌'}\n\n"
        
        # Checksums
        checksum_info = details.get('checksums', {})
        if checksum_info.get('sha_file_exists'):
            report += f"### Checksums SHA256\n"
            report += f"- Archivos verificados: {checksum_info.get('total_files', 0)}\n"
            report += f"- Correctos: {checksum_info.get('verified', 0)}\n"
            report += f"- Incorrectos: {checksum_info.get('failed', 0)}\n"
            report += f"- Faltantes: {checksum_info.get('missing', 0)}\n"
            if checksum_info.get('failed', 0) == 0 and checksum_info.get('missing', 0) == 0:
                report += f"- Estado: ✅\n\n"
            else:
                report += f"- Estado: ❌\n\n"
        else:
            report += f"### Checksums SHA256\n"
            report += f"- SHA256SUMS.txt: No disponible\n\n"
        
        # wfdb
        wfdb_info = details.get('wfdb', {})
        if wfdb_info.get('available'):
            report += f"### Carga con wfdb\n"
            report += f"- Registros cargados: {wfdb_info.get('loaded', 0)}/{files_info.get('expected', 0)}\n"
            report += f"- Registros fallidos: {wfdb_info.get('failed', 0)}\n"
            if wfdb_info.get('loaded', 0) == files_info.get('expected', 0):
                report += f"- Estado: ✅\n\n"
            else:
                report += f"- Estado: ❌\n\n"
        else:
            report += f"### Carga con wfdb\n"
            report += f"- wfdb: No disponible\n\n"
        
        report += f"### Estado General\n"
        report += f"- {'✅ INTEGRIDAD 100%' if result['overall_ok'] else '⚠️  INTEGRIDAD PARCIAL'}\n\n"
        report += "---\n\n"
    
    # Recomendaciones
    report += "## Recomendaciones\n\n"
    
    if complete_datasets == total_datasets:
        report += "✅ Los datasets están completamente descargados y verificados.\n\n"
        report += "**Próximo paso**: Proceder con el entrenamiento de modelos:\n"
        report += "```bash\n"
        report += "python scripts/train_models.py --train-all --data-dir datasets/ --models-dir models/\n"
        report += "```\n"
    else:
        report += "⚠️  Algunos datasets requieren atención:\n\n"
        for result in all_results:
            if not result['overall_ok']:
                report += f"- **{result['name']}**: "
                issues = []
                if not result['size_ok']:
                    issues.append("tamaño")
                if not result['files_ok']:
                    issues.append("archivos")
                if result['checksums_ok'] is False:
                    issues.append("checksums")
                if result['wfdb_ok'] is False:
                    issues.append("carga wfdb")
                report += f"{', '.join(issues)}\n"
        
        report += "\n**Recomendación**: Reiniciar descargas completas:\n"
        report += "```bash\n"
        report += "bash scripts/download_completo_verificado.sh\n"
        report += "```\n"
    
    # Guardar reporte
    output_path = BASE_DIR / output_file
    output_path.write_text(report, encoding='utf-8')
    print(f"\n📄 Reporte guardado en: {output_path}")

def main():
    print("=" * 70)
    print("🔍 VALIDACIÓN COMPLETA DE DATASETS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    datasets = {
        'SDDB': {
            'path': 'datasets/sddb/physionet.org/files/sddb/1.0.0',
            'expected_gb': 5.0,
            'expected_records': 23,
            'description': 'Sudden Cardiac Death Holter Database'
        },
        'NSRDB': {
            'path': 'datasets/nsrdb/physionet.org/files/nsrdb/1.0.0',
            'expected_gb': 2.0,
            'expected_records': 18,
            'description': 'Normal Sinus Rhythm Database'
        },
        'CUDB': {
            'path': 'datasets/cudb/physionet.org/files/cudb/1.0.0',
            'expected_gb': 9.5,
            'expected_records': 35,
            'description': 'Ventricular Tachyarrhythmia Database'
        }
    }
    
    all_results = []
    
    for name, config in datasets.items():
        result = verify_dataset_complete(
            name,
            config['path'],
            config['expected_gb'],
            config['expected_records']
        )
        all_results.append(result)
    
    # Resumen final
    print(f"\n{'='*70}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*70}")
    
    complete_count = sum(1 for r in all_results if r['overall_ok'])
    total_count = len(all_results)
    
    print(f"\nDatasets con integridad 100%: {complete_count}/{total_count}")
    
    for result in all_results:
        status = "✅" if result['overall_ok'] else "⚠️ "
        print(f"  {status} {result['name']}")
    
    if complete_count == total_count:
        print("\n✅ TODOS LOS DATASETS TIENEN INTEGRIDAD 100%")
        print("\n💡 Los datasets están completamente descargados y verificados.")
        print("   Puedes proceder con el entrenamiento de modelos.")
        return_code = 0
    else:
        print("\n⚠️  ALGUNOS DATASETS NO TIENEN INTEGRIDAD 100%")
        print("\n💡 Revisa los detalles arriba y considera reiniciar descargas si es necesario.")
        return_code = 1
    
    # Generar reporte
    print("\n📄 Generando reporte final...")
    generate_report(all_results, BASE_DIR / "docs" / "validacion_completa_report.md")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())

