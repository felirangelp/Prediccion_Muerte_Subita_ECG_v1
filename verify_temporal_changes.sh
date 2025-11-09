#!/bin/bash
echo "🔍 Verificando cambios en análisis temporal..."
echo ""

# Verificar que el código tiene el mapeo
if grep -q "Mapear claves \[1,2,3,4,5\]" results/dashboard_scd_prediction.html; then
    echo "✅ Mapeo de claves presente"
else
    echo "❌ Mapeo de claves NO encontrado"
fi

# Verificar rango dinámico
if grep -q "range: \[minY, maxY\]" results/dashboard_scd_prediction.html; then
    echo "✅ Rango Y dinámico presente"
else
    echo "❌ Rango Y dinámico NO encontrado"
fi

# Verificar que usa accuracy
if grep -q "accuracies.push(acc \* 100)" results/dashboard_scd_prediction.html; then
    echo "✅ Usa accuracy para el gráfico"
else
    echo "❌ No usa accuracy"
fi

echo ""
echo "📋 Para ver los cambios:"
echo "1. Abre el dashboard: open results/dashboard_scd_prediction.html"
echo "2. O en GitHub Pages: https://felirangelp.github.io/Prediccion_Muerte_Subita_ECG_v1/?v=$(date +%s)"
echo "3. Si no ves cambios, limpia el caché del navegador (Ctrl+Shift+Delete)"
