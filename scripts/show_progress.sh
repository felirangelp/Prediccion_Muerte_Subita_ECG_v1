#!/bin/bash
# Script simple para mostrar progreso de descarga

echo "🚀 PROGRESO DE DESCARGA ECG"
echo "=========================="
echo ""

# SCDH
if [ -d "datasets/sddb" ]; then
    current_size=$(du -s datasets/sddb/ | cut -f1)
    current_mb=$((current_size / 1024))
    target_mb=5000
    progress=$((current_mb * 100 / target_mb))
    
    echo "✅ SCDH (Sudden Cardiac Death):"
    echo "   📊 Progreso: [$progress%] - ${current_mb}MB / ${target_mb}MB"
    echo "   📄 Archivos: $(ls datasets/sddb/ | wc -l)"
else
    echo "❌ SCDH: No iniciado"
fi

echo ""

# NSRDB
if [ -d "datasets/nsrdb" ]; then
    current_size=$(du -s datasets/nsrdb/ | cut -f1)
    if [ $current_size -gt 0 ]; then
        current_mb=$((current_size / 1024))
        target_mb=2000
        progress=$((current_mb * 100 / target_mb))
        echo "🔄 NSRDB (Normal Sinus Rhythm):"
        echo "   📊 Progreso: [$progress%] - ${current_mb}MB / ${target_mb}MB"
    else
        echo "⏳ NSRDB (Normal Sinus Rhythm):"
        echo "   📊 Progreso: [0%] - Esperando..."
    fi
else
    echo "❌ NSRDB: No iniciado"
fi

echo ""

# CUDB
if [ -d "datasets/cudb" ]; then
    current_size=$(du -s datasets/cudb/ | cut -f1)
    if [ $current_size -gt 0 ]; then
        current_mb=$((current_size / 1024))
        target_mb=9500
        progress=$((current_mb * 100 / target_mb))
        echo "🔄 CUDB (Ventricular Tachyarrhythmia):"
        echo "   📊 Progreso: [$progress%] - ${current_mb}MB / ${target_mb}MB"
    else
        echo "⏳ CUDB (Ventricular Tachyarrhythmia):"
        echo "   📊 Progreso: [0%] - Esperando..."
    fi
else
    echo "❌ CUDB: No iniciado"
fi

echo ""
echo "🔄 Proceso activo: $(ps aux | grep 'python.*download' | grep -v grep | wc -l) proceso(s)"
echo "⏱️  Actualizado: $(date '+%H:%M:%S')"
echo ""
echo "💡 Para ver progreso: bash scripts/show_progress.sh"
