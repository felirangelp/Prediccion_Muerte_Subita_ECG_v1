#!/bin/bash
# Script para monitorear automáticamente el progreso del entrenamiento

cd "$(dirname "$0")/.."

INTERVAL=${1:-1800}  # Intervalo en segundos (default: 30 minutos = 1800 seg)
MAX_CHECKS=${2:-0}   # Número máximo de verificaciones (0 = infinito)

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          MONITOREO AUTOMÁTICO DE ENTRENAMIENTO                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Configuración:"
echo "   Intervalo: $INTERVAL segundos ($(($INTERVAL / 60)) minutos)"
if [ "$MAX_CHECKS" -eq 0 ]; then
    echo "   Verificaciones: Infinitas (Ctrl+C para detener)"
else
    echo "   Verificaciones: $MAX_CHECKS"
fi
echo ""
echo "🔄 Iniciando monitoreo..."
echo ""

CHECK_COUNT=0

while true; do
    CHECK_COUNT=$((CHECK_COUNT + 1))
    
    echo "======================================================================"
    echo "📊 VERIFICACIÓN #$CHECK_COUNT - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
    
    # Ejecutar script de verificación
    ./scripts/check_training_progress.sh
    
    # Verificar si el proceso aún está corriendo
    PROCESS_ID=$(ps aux | grep "train_models.py --train-all" | grep -v grep | awk '{print $2}' | head -1)
    
    if [ -z "$PROCESS_ID" ]; then
        echo ""
        echo "✅ El proceso de entrenamiento ha terminado!"
        echo "   Verificando si todos los modelos se completaron..."
        
        # Verificar modelos finales
        if [ -f "models/sparse_classifier.pkl" ] && [ -f "models/hierarchical_classifier_metadata.pkl" ]; then
            echo "   ✅ Todos los modelos parecen estar completos"
            echo ""
            echo "📋 Próximos pasos sugeridos:"
            echo "   1. Evaluar modelos: python scripts/evaluate_models.py"
            echo "   2. Actualizar dashboard: python scripts/generate_dashboard.py"
        fi
        
        break
    fi
    
    # Verificar límite de checks
    if [ "$MAX_CHECKS" -gt 0 ] && [ "$CHECK_COUNT" -ge "$MAX_CHECKS" ]; then
        echo ""
        echo "⏸️  Límite de verificaciones alcanzado ($MAX_CHECKS)"
        break
    fi
    
    echo ""
    echo "⏳ Esperando $INTERVAL segundos hasta la próxima verificación..."
    echo "   (Presiona Ctrl+C para detener el monitoreo)"
    echo ""
    
    sleep $INTERVAL
done

echo ""
echo "✅ Monitoreo finalizado"

