#!/bin/bash
# Script rápido para ver el estado actual del entrenamiento

cd "$(dirname "$0")/.."

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          ESTADO ACTUAL DEL ENTRENAMIENTO                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar proceso
PROCESS=$(ps aux | grep "train_models.py" | grep -v grep | head -1)

if [ -z "$PROCESS" ]; then
    echo "⚠️  No hay proceso de entrenamiento activo"
    echo ""
    echo "📁 Verificando modelos completados..."
    if [ -f "models/sparse_classifier.pkl" ]; then
        echo "   ✅ Sparse: Completado"
    else
        echo "   ⏳ Sparse: No encontrado"
    fi
    
    if [ -f "models/hierarchical_classifier_metadata.pkl" ]; then
        echo "   ✅ Hierarchical: Completado"
    else
        echo "   ⏳ Hierarchical: No encontrado"
    fi
else
    PID=$(echo $PROCESS | awk '{print $2}')
    CPU=$(echo $PROCESS | awk '{print $3}')
    MEM=$(echo $PROCESS | awk '{print $4}')
    
    ETIME=$(ps -p $PID -o etime= | tr -d ' ')
    STATE=$(ps -p $PID -o state= | tr -d ' ')
    
    echo "🔄 Proceso activo (PID: $PID)"
    echo "   Estado: $STATE $([ "$STATE" = "R" ] && echo "✅ ACTIVO" || echo "⏸️  EN ESPERA")"
    echo "   Tiempo: $ETIME"
    echo "   CPU: $CPU%"
    echo "   Memoria: $MEM%"
    echo ""
    
    # Determinar modelo
    if echo "$PROCESS" | grep -q "train-sparse"; then
        echo "   🎯 Entrenando: Modelo Sparse (K-SVD + OMP)"
    elif echo "$PROCESS" | grep -q "train-hierarchical"; then
        echo "   🎯 Entrenando: Modelo Hierarchical (TCN + Fusion)"
    elif echo "$PROCESS" | grep -q "train-hybrid"; then
        echo "   🎯 Entrenando: Modelo Hybrid"
    elif echo "$PROCESS" | grep -q "train-all"; then
        echo "   🎯 Entrenando: Todos los modelos"
    fi
    
    echo ""
    echo "📝 Última actividad del log:"
    if [ -f "/tmp/training_sparse_fixed.log" ]; then
        tail -1 /tmp/training_sparse_fixed.log | head -c 80
        echo "..."
    else
        echo "   (Log no disponible)"
    fi
fi

echo ""
echo "💡 Para monitoreo continuo: python scripts/monitor_training.py [intervalo_seg]"
echo "   Ejemplo: python scripts/monitor_training.py 900  (cada 15 minutos)"

