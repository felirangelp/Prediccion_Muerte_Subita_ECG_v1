#!/bin/bash

# Script para configurar ambiente optimizado para MacBook Pro M1
# Instala TensorFlow con soporte Metal (GPU) y optimizaciones específicas

set -e

echo "🚀 Configurando ambiente optimizado para MacBook Pro M1"
echo "========================================================"

# Verificar que estamos en Mac
if [[ "$(uname)" != "Darwin" ]]; then
    echo "⚠️  Este script está optimizado para macOS. Continuando de todas formas..."
fi

# Verificar arquitectura
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "✅ Arquitectura Apple Silicon (M1/M2/M3) detectada"
    IS_M1=true
else
    echo "⚠️  Arquitectura Intel detectada. Algunas optimizaciones no se aplicarán."
    IS_M1=false
fi

# Activar ambiente virtual si existe
if [ -d "venv" ]; then
    echo ""
    echo "🔧 Activando ambiente virtual..."
    source venv/bin/activate
else
    echo "❌ Ambiente virtual no encontrado. Ejecuta primero: ./setup_env.sh"
    exit 1
fi

# Actualizar pip
echo ""
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias base primero
echo ""
echo "📚 Instalando dependencias base..."
pip install numpy scipy pandas matplotlib seaborn

# Instalar TensorFlow optimizado para M1
echo ""
if [[ "$IS_M1" == true ]]; then
    echo "🤖 Instalando TensorFlow para Apple Silicon (Metal GPU)..."
    pip install tensorflow-macos tensorflow-metal
    echo "✅ TensorFlow con soporte Metal instalado"
else
    echo "🤖 Instalando TensorFlow estándar..."
    pip install tensorflow
fi

# Instalar librerías de procesamiento de señales
echo ""
echo "📊 Instalando librerías de procesamiento de señales..."
pip install PyWavelets scikit-sparse

# Instalar librerías de análisis biomédico
echo ""
echo "🏥 Instalando librerías de análisis biomédico..."
pip install heartpy nolds

# Instalar scikit-learn y otras dependencias
echo ""
echo "🧠 Instalando librerías de Machine Learning..."
pip install scikit-learn joblib

# Instalar Plotly para visualizaciones
echo ""
echo "📈 Instalando Plotly para visualizaciones interactivas..."
pip install plotly

# Instalar utilidades
echo ""
echo "🔧 Instalando utilidades adicionales..."
pip install tqdm requests cachetools

# Verificar instalación de TensorFlow
echo ""
echo "🔍 Verificando instalación de TensorFlow..."
python3 -c "
import sys
try:
    import tensorflow as tf
    print(f'✅ TensorFlow versión: {tf.__version__}')
    
    # Verificar GPU Metal
    if hasattr(tf.config, 'list_physical_devices'):
        devices = tf.config.list_physical_devices()
        gpu_devices = [d for d in devices if 'GPU' in d.name or 'Metal' in d.name]
        if gpu_devices:
            print(f'✅ GPU Metal detectada: {gpu_devices[0].name}')
        else:
            print('⚠️  GPU Metal no detectada (se usará CPU)')
    else:
        print('⚠️  No se pudo verificar dispositivos GPU')
except Exception as e:
    print(f'❌ Error verificando TensorFlow: {e}')
    sys.exit(1)
"

# Verificar otras instalaciones críticas
echo ""
echo "🔍 Verificando otras dependencias críticas..."
python3 -c "
import sys
dependencies = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'plotly': 'plotly',
    'PyWavelets': 'PyWavelets',
    'wfdb': 'wfdb'
}

failed = []
for name, module in dependencies.items():
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError:
        print(f'❌ {name} - NO INSTALADO')
        failed.append(name)

if failed:
    print(f'\n⚠️  Dependencias faltantes: {', '.join(failed)}')
    sys.exit(1)
"

# Ejecutar script de configuración Python
echo ""
echo "⚙️  Ejecutando configuración optimizada..."
python3 -c "
from src.config_m1 import print_system_info, optimize_numpy_scipy, setup_memory_management
optimize_numpy_scipy()
setup_memory_management()
print_system_info()
"

echo ""
echo "🎉 ¡Ambiente configurado exitosamente para MacBook Pro M1!"
echo ""
echo "📋 Próximos pasos:"
echo "   1. Verificar configuración:"
echo "      python -c 'from src.config_m1 import print_system_info; print_system_info()'"
echo ""
echo "   2. Para entrenar modelos:"
echo "      python scripts/train_models.py"
echo ""
echo "   3. Para generar dashboard:"
echo "      python scripts/generate_dashboard.py"
echo ""

