#!/bin/bash
# Script para actualizar el dashboard en GitHub Pages
# Uso: ./scripts/update_github_pages.sh

set -e

echo "🔄 Actualizando dashboard para GitHub Pages..."

# Verificar que el dashboard existe
if [ ! -f "results/dashboard_scd_prediction.html" ]; then
    echo "❌ Error: No se encontró results/dashboard_scd_prediction.html"
    echo "   Primero genera el dashboard con: python scripts/generate_dashboard.py"
    exit 1
fi

# Copiar dashboard a docs/
echo "📋 Copiando dashboard a docs/index.html..."
cp results/dashboard_scd_prediction.html docs/index.html

# Verificar que .nojekyll existe
if [ ! -f "docs/.nojekyll" ]; then
    echo "📝 Creando docs/.nojekyll..."
    touch docs/.nojekyll
fi

echo "✅ Dashboard actualizado para GitHub Pages"
echo ""
echo "📤 Para publicar los cambios:"
echo "   git add docs/index.html docs/.nojekyll"
echo "   git commit -m 'Actualizar dashboard'"
echo "   git push origin main"
echo ""
echo "⏱️  GitHub Pages se actualizará automáticamente en 1-5 minutos"

