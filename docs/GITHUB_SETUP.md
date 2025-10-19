# 🚀 Instrucciones para GitHub

## 📋 Pasos para Conectar con GitHub

### 1. Crear Repositorio en GitHub

1. Ir a [GitHub.com](https://github.com)
2. Hacer clic en "New repository"
3. Nombre: `Prediccion_Muerte_Subita_ECG_v1`
4. Descripción: `Proyecto de predicción de muerte súbita cardíaca usando ECG`
5. **NO** inicializar con README (ya tenemos uno)
6. Hacer clic en "Create repository"

### 2. Conectar Repositorio Local

```bash
# Agregar remote origin (reemplazar con tu URL)
git remote add origin https://github.com/TU-USUARIO/Prediccion_Muerte_Subita_ECG_v1.git

# Cambiar a rama main
git branch -M main

# Subir código inicial
git push -u origin main
```

### 3. Configurar GitHub Pages (Opcional)

1. Ir a Settings > Pages
2. Source: Deploy from a branch
3. Branch: main
4. Folder: /docs
5. Guardar

## 🔧 Configuración Adicional

### GitHub Actions (CI/CD)

Crear archivo `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Verify setup
      run: python scripts/verify_setup.py
```

### Issues Templates

Crear `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Windows 10, macOS 12, Ubuntu 20.04]
 - Python version: [e.g. 3.9.7]
 - VS Code version: [e.g. 1.80.0]

**Additional context**
Add any other context about the problem here.
```

## 📊 Estadísticas del Proyecto

### Archivos Incluidos en GitHub

✅ **Código fuente**:
- `src/` - Módulos principales
- `scripts/` - Scripts de automatización
- `docs/` - Documentación completa

✅ **Configuración**:
- `.vscode/` - Configuración de VS Code
- `.gitignore` - Exclusiones de Git
- `requirements.txt` - Dependencias Python
- `LICENSE` - Licencia MIT

✅ **Documentación**:
- `README.md` - Documentación principal
- `docs/DEPLOYMENT_GUIDE.md` - Guía de despliegue
- `docs/DATASETS_INFO.md` - Info técnica

### Archivos Excluidos (NO en GitHub)

❌ **Datasets** (16.5 GB):
- `datasets/sddb/` - SCDH Database
- `datasets/nsrdb/` - NSRDB Database  
- `datasets/cudb/` - CUDB Database

❌ **Archivos temporales**:
- `venv/` - Ambiente virtual
- `*.log` - Archivos de log
- `physionet.org/` - Archivos temporales de wget

## 🎯 Comandos para Usuarios

### Clonar y Configurar

```bash
# Clonar repositorio
git clone https://github.com/TU-USUARIO/Prediccion_Muerte_Subita_ECG_v1.git
cd Prediccion_Muerte_Subita_ECG_v1

# Configurar ambiente
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verificar sistema
python scripts/verify_setup.py

# Descargar datasets
bash scripts/download_auto.sh
```

### Actualizar desde GitHub

```bash
# Obtener últimos cambios
git pull origin main

# Verificar cambios
git log --oneline -5
```

## 🔗 Enlaces Útiles

- **Repositorio**: https://github.com/TU-USUARIO/Prediccion_Muerte_Subita_ECG_v1
- **Issues**: https://github.com/TU-USUARIO/Prediccion_Muerte_Subita_ECG_v1/issues
- **PhysioNet**: https://physionet.org/
- **Documentación**: Ver `docs/` en el repositorio

---

**Nota**: Reemplazar `TU-USUARIO` con tu nombre de usuario de GitHub
