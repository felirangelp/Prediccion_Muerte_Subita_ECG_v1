#!/bin/bash
# Script para verificar el progreso del entrenamiento

cd "$(dirname "$0")/.."

echo "======================================================================"
echo "📊 VERIFICACIÓN DE PROGRESO DEL ENTRENAMIENTO"
echo "======================================================================"
echo ""

# Verificar proceso
PROCESS_ID=$(ps aux | grep "train_models.py --train-all" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PROCESS_ID" ]; then
    echo "⚠️  No se encontró proceso de entrenamiento activo"
    echo "   El entrenamiento puede haber terminado"
else
    echo "🔄 Proceso activo (PID: $PROCESS_ID):"
    ps -p "$PROCESS_ID" -o etime,pcpu,pmem,command 2>/dev/null | tail -1 | awk '{print "   Tiempo: " $1 ", CPU: " $2 "%, Memoria: " $3 "%"}'
fi

echo ""
echo "📁 Estado de modelos:"

# Verificar Hierarchical
if [ -d "models/hierarchical_classifier" ] || [ -f "models/hierarchical_classifier_metadata.pkl" ]; then
    LATEST=$(find models/ -name "hierarchical_classifier*" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST" ]; then
        MTIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST" 2>/dev/null)
        echo "   ✅ Hierarchical: Actualizado en $MTIME"
    fi
else
    echo "   ⏳ Hierarchical: No encontrado"
fi

# Verificar Hybrid
if [ -d "models/hybrid_model" ] || [ -f "models/hybrid_model_ensemble.pkl" ]; then
    LATEST=$(find models/ -name "hybrid_model*" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST" ]; then
        MTIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST" 2>/dev/null)
        echo "   ✅ Hybrid: Actualizado en $MTIME"
    fi
else
    echo "   ⏳ Hybrid: No encontrado"
fi

# Verificar Sparse
if [ -f "models/sparse_classifier.pkl" ]; then
    MTIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "models/sparse_classifier.pkl" 2>/dev/null)
    SIZE=$(stat -f "%z" "models/sparse_classifier.pkl" 2>/dev/null)
    SIZE_MB=$(echo "scale=2; $SIZE / 1024 / 1024" | bc)
    echo "   ✅ Sparse: ${SIZE_MB} MB, Actualizado en $MTIME"
else
    echo "   ⏳ Sparse: No encontrado"
fi

echo ""
echo "📊 Archivos más recientes en models/:"
find models/ -type f -name "*.pkl" -o -name "*.h5" | xargs ls -lth 2>/dev/null | head -5 | awk '{print "   " $9, "(" $5, "modificado", $6, $7, $8 ")"}'

echo ""
echo "======================================================================"

