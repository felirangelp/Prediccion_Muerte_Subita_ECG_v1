#!/bin/bash

# Script para configurar el ambiente de desarrollo para Predicción de Muerte Súbita ECG
# Basado en los papers de Velázquez-González et al. y Huang et al.

set -e  # Salir si hay algún error

echo "🚀 Configurando ambiente para Predicción de Muerte Súbita ECG"
echo "=============================================================="

# Verificar que Python 3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Por favor instala Python 3.8 o superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python ${PYTHON_VERSION} detectado"

# Crear ambiente virtual
echo ""
echo "📦 Creando ambiente virtual..."
if [ -d "venv" ]; then
    echo "⚠️  El ambiente virtual 'venv' ya existe"
    read -p "¿Deseas recrearlo? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Eliminando ambiente virtual existente..."
        rm -rf venv
    else
        echo "📁 Usando ambiente virtual existente"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Ambiente virtual creado"
fi

# Activar ambiente virtual
echo ""
echo "🔧 Activando ambiente virtual..."
source venv/bin/activate

# Actualizar pip
echo ""
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo ""
echo "📚 Instalando dependencias desde requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ Archivo requirements.txt no encontrado"
    exit 1
fi

pip install -r requirements.txt
echo "✅ Dependencias instaladas"

# Verificar instalación de wfdb
echo ""
echo "🔍 Verificando instalación de wfdb..."
python3 -c "import wfdb; print(f'✅ wfdb versión: {wfdb.__version__}')" || {
    echo "❌ Error verificando wfdb"
    exit 1
}

# Crear directorios necesarios
echo ""
echo "📁 Creando estructura de directorios..."
mkdir -p datasets/sddb datasets/nsrdb datasets/cudb
mkdir -p scripts src docs
echo "✅ Directorios creados"

# Hacer ejecutables los scripts
echo ""
echo "🔧 Configurando permisos de scripts..."
chmod +x scripts/download_datasets.py
chmod +x scripts/verify_datasets.py
echo "✅ Scripts configurados como ejecutables"

# Mostrar información del ambiente
echo ""
echo "📊 Información del ambiente configurado:"
echo "========================================"
echo "🐍 Python: $(python3 --version)"
echo "📦 Ambiente virtual: $(which python3)"
echo "📚 Ubicación: $(pwd)/venv"
echo "📁 Directorios creados:"
echo "   • datasets/ (para los datasets de PhysioNet)"
echo "   • scripts/ (scripts de utilidad)"
echo "   • src/ (código fuente)"
echo "   • docs/ (documentación)"

echo ""
echo "🎉 ¡Ambiente configurado exitosamente!"
echo ""
echo "📋 Próximos pasos:"
echo "   1. Para descargar los datasets:"
echo "      python scripts/download_datasets.py"
echo ""
echo "   2. Para verificar los datasets:"
echo "      python scripts/verify_datasets.py"
echo ""
echo "   3. Para activar el ambiente en el futuro:"
echo "      source venv/bin/activate"
echo ""
echo "📚 Referencias:"
echo "   • Velázquez-González et al., Sensors 2021"
echo "   • Huang et al., Symmetry 2025"
echo ""
echo "💡 Nota: Los datasets (~16.5 GB) se descargarán en la carpeta 'datasets/'"
echo "   cuando ejecutes el script de descarga."
