#!/usr/bin/env python3
"""
Script de verificación del sistema
Verifica que todos los componentes estén correctamente instalados
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def print_header():
    """Imprime el encabezado del script"""
    print("🔍 VERIFICACIÓN DEL SISTEMA ECG")
    print("=" * 50)
    print()

def check_python_version():
    """Verifica la versión de Python"""
    print("🐍 Verificando Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requerido: 3.8+)")
        return False

def check_packages():
    """Verifica que los paquetes requeridos estén instalados"""
    print("\n📦 Verificando paquetes Python...")
    
    required_packages = [
        ('wfdb', 'WFDB'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras')
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_directories():
    """Verifica que los directorios necesarios existan"""
    print("\n📁 Verificando estructura de directorios...")
    
    required_dirs = [
        'src',
        'scripts',
        'docs',
        'datasets',
        '.vscode'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0, missing_dirs

def check_scripts():
    """Verifica que los scripts necesarios existan"""
    print("\n🔧 Verificando scripts...")
    
    required_scripts = [
        'scripts/download_auto.sh',
        'scripts/show_progress.sh',
        'scripts/verify_datasets.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script}")
            missing_scripts.append(script)
    
    return len(missing_scripts) == 0, missing_scripts

def check_datasets():
    """Verifica el estado de los datasets"""
    print("\n📊 Verificando datasets...")
    
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("   ❌ Directorio datasets no encontrado")
        return False
    
    # Verificar subdirectorios
    subdirs = ['sddb', 'nsrdb', 'cudb']
    for subdir in subdirs:
        subdir_path = datasets_dir / subdir
        if subdir_path.exists():
            # Contar archivos .hea
            hea_files = list(subdir_path.glob("*.hea"))
            if hea_files:
                print(f"   ✅ {subdir}/ ({len(hea_files)} archivos)")
            else:
                print(f"   ⚠️  {subdir}/ (vacío)")
        else:
            print(f"   ❌ {subdir}/ (no encontrado)")
    
    return True

def check_git():
    """Verifica la configuración de Git"""
    print("\n🔗 Verificando Git...")
    
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Git no encontrado")
            return False
    except FileNotFoundError:
        print("   ❌ Git no instalado")
        return False

def check_wget():
    """Verifica la disponibilidad de wget"""
    print("\n🌐 Verificando wget...")
    
    try:
        result = subprocess.run(['wget', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ {version_line}")
            return True
        else:
            print("   ❌ wget no encontrado")
            return False
    except FileNotFoundError:
        print("   ❌ wget no instalado")
        return False

def print_summary(results):
    """Imprime el resumen de la verificación"""
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    
    all_ok = all(results.values())
    
    if all_ok:
        print("🎉 ¡Sistema configurado correctamente!")
        print("✅ Todos los componentes están listos")
        print("\n🚀 Próximos pasos:")
        print("   1. Descargar datasets: bash scripts/download_auto.sh")
        print("   2. Verificar descarga: python scripts/verify_datasets.py")
        print("   3. Monitorear progreso: bash scripts/show_progress.sh")
    else:
        print("⚠️  Hay problemas que resolver:")
        
        if not results['python']:
            print("   • Instalar Python 3.8+")
        
        if not results['packages'][0]:
            print(f"   • Instalar paquetes faltantes: {', '.join(results['packages'][1])}")
        
        if not results['directories'][0]:
            print(f"   • Crear directorios faltantes: {', '.join(results['directories'][1])}")
        
        if not results['scripts'][0]:
            print(f"   • Verificar scripts faltantes: {', '.join(results['scripts'][1])}")
        
        if not results['git']:
            print("   • Instalar Git")
        
        if not results['wget']:
            print("   • Instalar wget (o usar Git Bash en Windows)")

def main():
    """Función principal"""
    print_header()
    
    # Ejecutar verificaciones
    python_ok = check_python_version()
    packages_ok, missing_packages = check_packages()
    directories_ok, missing_dirs = check_directories()
    scripts_ok, missing_scripts = check_scripts()
    datasets_ok = check_datasets()
    git_ok = check_git()
    wget_ok = check_wget()
    
    # Compilar resultados
    results = {
        'python': python_ok,
        'packages': (packages_ok, missing_packages),
        'directories': (directories_ok, missing_dirs),
        'scripts': (scripts_ok, missing_scripts),
        'datasets': datasets_ok,
        'git': git_ok,
        'wget': wget_ok
    }
    
    # Imprimir resumen
    print_summary(results)
    
    # Código de salida
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
