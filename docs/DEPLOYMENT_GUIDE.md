# 🚀 Guía de Despliegue - Visual Studio Code

## 📋 Requisitos Previos

### Sistema Operativo
- **Windows**: Windows 10/11
- **macOS**: macOS 10.15 o superior
- **Linux**: Ubuntu 18.04+ / CentOS 7+ / Debian 10+

### Software Necesario
- **Python**: 3.8 o superior
- **Git**: Para clonar el repositorio
- **Visual Studio Code**: Última versión
- **Conexión a Internet**: Para descargar datasets

### Hardware Mínimo
- **RAM**: 8 GB (recomendado 16 GB)
- **Espacio en disco**: 25 GB libres
- **Procesador**: Intel i5 / AMD Ryzen 5 o superior

## 🔧 Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
# Clonar desde GitHub
git clone https://github.com/tu-usuario/Prediccion_Muerte_Subita_ECG_v1.git

# Entrar al directorio
cd Prediccion_Muerte_Subita_ECG_v1
```

### 2. Configurar Visual Studio Code

#### Instalar Extensiones Recomendadas
```bash
# Instalar extensiones esenciales
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.vscode-json
code --install-extension redhat.vscode-yaml
```

#### Configurar Python Interpreter
1. Abrir VS Code: `code .`
2. Presionar `Ctrl+Shift+P` (Windows/Linux) o `Cmd+Shift+P` (macOS)
3. Buscar "Python: Select Interpreter"
4. Seleccionar Python 3.8+

### 3. Configurar Ambiente Virtual

#### Windows (PowerShell)
```powershell
# Crear ambiente virtual
python -m venv venv

# Activar ambiente virtual
.\venv\Scripts\Activate.ps1

# Si hay problemas de ejecución:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS/Linux (Terminal)
```bash
# Crear ambiente virtual
python3 -m venv venv

# Activar ambiente virtual
source venv/bin/activate
```

### 4. Instalar Dependencias

```bash
# Instalar paquetes desde requirements.txt
pip install -r requirements.txt

# Verificar instalación
pip list
```

### 5. Configurar Variables de Entorno

#### Crear archivo .env
```bash
# Crear archivo de configuración
touch .env
```

#### Contenido del archivo .env
```env
# Configuración del proyecto
PROJECT_NAME=Prediccion_Muerte_Subita_ECG
DATASETS_DIR=datasets
LOG_LEVEL=INFO

# Configuración de PhysioNet
PHYSIONET_BASE_URL=https://physionet.org/files/

# Configuración de procesamiento
SAMPLE_RATE=250
WINDOW_SIZE=30
```

### 6. Descargar Datasets

#### Método Automático (Recomendado)
```bash
# Descargar todos los datasets automáticamente
bash scripts/download_auto.sh
```

#### Método Manual
```bash
# Descargar SCDH
cd datasets/sddb
wget -r -N -c -np https://physionet.org/files/sddb/1.0.0/

# Descargar NSRDB
cd ../nsrdb
wget -r -N -c -np https://physionet.org/files/nsrdb/1.0.0/

# Descargar CUDB
cd ../cudb
wget -r -N -c -np https://physionet.org/files/cudb/1.0.0/
```

### 7. Verificar Instalación

```bash
# Verificar datasets
python scripts/verify_datasets.py

# Verificar progreso
bash scripts/show_progress.sh

# Probar importaciones
python -c "import wfdb; import numpy; import scipy; print('✅ Todas las librerías funcionan')"
```

## 🎯 Configuración de VS Code

### Configuración del Workspace

#### Crear .vscode/settings.json
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "files.exclude": {
        "**/datasets/**": true,
        "**/venv/**": true,
        "**/__pycache__/**": true
    },
    "search.exclude": {
        "**/datasets/**": true,
        "**/venv/**": true
    }
}
```

#### Crear .vscode/launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Download Datasets",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/download_datasets.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### Extensiones Recomendadas

#### Lista completa de extensiones
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.flake8",
        "ms-vscode.vscode-github-actions",
        "ms-vscode.remote-containers"
    ]
}
```

## 🔍 Troubleshooting

### Problemas Comunes

#### Error: "Python not found"
```bash
# Verificar instalación de Python
python --version
python3 --version

# Instalar Python desde python.org o usar conda
conda install python=3.9
```

#### Error: "Permission denied" en scripts
```bash
# Dar permisos de ejecución
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

#### Error: "Module not found"
```bash
# Verificar ambiente virtual activo
which python
pip list

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

#### Error: "wget not found" (Windows)
```bash
# Instalar wget en Windows
# Opción 1: Usar Git Bash (incluye wget)
# Opción 2: Instalar con chocolatey
choco install wget

# Opción 3: Usar PowerShell equivalente
Invoke-WebRequest -Uri "URL" -OutFile "archivo"
```

### Verificación de Sistema

#### Script de verificación
```python
# Crear archivo: verify_setup.py
import sys
import subprocess
import importlib

def check_python_version():
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requerido: 3.8+)")
        return False

def check_packages():
    required_packages = [
        'wfdb', 'numpy', 'scipy', 'matplotlib', 
        'seaborn', 'pandas', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_datasets():
    import os
    datasets_dir = "datasets"
    if os.path.exists(datasets_dir):
        print(f"✅ Directorio datasets existe")
        return True
    else:
        print(f"❌ Directorio datasets no encontrado")
        return False

if __name__ == "__main__":
    print("🔍 Verificando configuración del sistema...")
    print("=" * 50)
    
    python_ok = check_python_version()
    packages_ok = check_packages()
    datasets_ok = check_datasets()
    
    print("=" * 50)
    if python_ok and packages_ok and datasets_ok:
        print("🎉 ¡Sistema configurado correctamente!")
    else:
        print("⚠️  Hay problemas que resolver")
```

## 📊 Monitoreo y Mantenimiento

### Comandos Útiles

#### Verificar estado del sistema
```bash
# Ver progreso de descarga
bash scripts/show_progress.sh

# Verificar integridad de datasets
python scripts/verify_datasets.py

# Verificar configuración
python verify_setup.py
```

#### Limpiar sistema
```bash
# Limpiar archivos temporales
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Limpiar logs antiguos
rm -f *.log
rm -f scripts/*.log
```

### Actualizaciones

#### Actualizar dependencias
```bash
# Actualizar pip
pip install --upgrade pip

# Actualizar paquetes
pip install -r requirements.txt --upgrade

# Verificar versiones
pip list --outdated
```

#### Actualizar código desde GitHub
```bash
# Obtener últimos cambios
git pull origin main

# Verificar cambios
git log --oneline -5
```

## 🎯 Próximos Pasos

### Una vez configurado el ambiente:

1. **Explorar datasets**: `python scripts/explore_datasets.py`
2. **Ejecutar análisis**: `python src/analysis.py`
3. **Entrenar modelos**: `python src/train_models.py`
4. **Visualizar resultados**: `python src/visualize_results.py`

### Recursos Adicionales

- **Documentación PhysioNet**: https://physionet.org/
- **Documentación WFDB**: https://wfdb.readthedocs.io/
- **Tutoriales ECG**: https://www.ecgwaves.com/

---

**Última actualización**: Diciembre 2024  
**Versión**: 2.0.0  
**Compatibilidad**: VS Code 1.80+, Python 3.8+
