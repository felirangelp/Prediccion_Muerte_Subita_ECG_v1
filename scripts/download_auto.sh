#!/bin/bash
# Script para descarga automática uno tras otro

echo "🚀 DESCARGA AUTOMÁTICA UNO TRAS OTRO"
echo "===================================="
echo ""

# Función para verificar si wget está corriendo
is_wget_running() {
    ps aux | grep -v grep | grep "wget.*physionet" > /dev/null
    return $?
}

# Función para descargar un dataset
download_dataset() {
    local dataset=$1
    local name=$2
    local url=$3
    
    echo "📥 Descargando ${name}..."
    echo "   Destino: datasets/${dataset}/"
    echo "   URL: ${url}"
    echo ""
    
    cd "datasets/${dataset}"
    wget -r -N -c -np --progress=bar:force "${url}"
    local exit_code=$?
    cd ../..
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${name} descargado exitosamente"
        echo ""
        return 0
    else
        echo "❌ Error descargando ${name}"
        echo ""
        return 1
    fi
}

# Crear directorios
mkdir -p datasets/sddb datasets/nsrdb datasets/cudb

echo "📋 Secuencia automática:"
echo "1. SCDH (Sudden Cardiac Death) - ~5 GB"
echo "2. NSRDB (Normal Sinus Rhythm) - ~2 GB"
echo "3. CUDB (Ventricular Tachyarrhythmia) - ~9.5 GB"
echo ""

# Verificar si SCDH ya está descargando
if is_wget_running; then
    echo "🔄 SCDH ya está descargando..."
    echo "⏳ Esperando a que termine..."
    
    # Esperar a que termine SCDH
    while is_wget_running; do
        echo "   Progreso SCDH: $(du -sh datasets/sddb/ 2>/dev/null | cut -f1)"
        sleep 30
    done
    
    echo "✅ SCDH completado"
    echo ""
fi

# Descargar NSRDB
echo "🚀 Iniciando NSRDB..."
download_dataset "nsrdb" "NSRDB (Normal Sinus Rhythm)" "https://physionet.org/files/nsrdb/1.0.0/"

# Descargar CUDB
echo "🚀 Iniciando CUDB..."
download_dataset "cudb" "CUDB (Ventricular Tachyarrhythmia)" "https://physionet.org/files/cudb/1.0.0/"

echo "🎉 ¡TODOS LOS DATASETS DESCARGADOS!"
echo "=================================="
echo ""
echo "📊 Verificación final:"
bash scripts/show_progress.sh
