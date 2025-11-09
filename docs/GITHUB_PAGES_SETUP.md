# Configuración de GitHub Pages

Este documento explica cómo publicar el dashboard del proyecto en GitHub Pages.

## ✅ Configuración Actual

El proyecto ya está configurado para GitHub Pages:

- ✅ Dashboard copiado a `docs/index.html`
- ✅ Archivo `.nojekyll` creado para evitar problemas con Jekyll
- ✅ Todas las dependencias (Plotly) cargadas desde CDN

## 🚀 Pasos para Activar GitHub Pages

### 1. Configurar en GitHub

1. Ve a tu repositorio en GitHub: `https://github.com/felirangelp/Prediccion_Muerte_Subita_ECG_v1`
2. Haz clic en **Settings** (Configuración)
3. En el menú lateral, busca **Pages**
4. En la sección **Source**:
   - Selecciona **Deploy from a branch**
   - Branch: **main**
   - Folder: **/docs**
5. Haz clic en **Save**

### 2. Esperar el Despliegue

- GitHub procesará el despliegue en 1-5 minutos
- Verás un mensaje verde indicando que el sitio está publicado
- La URL será: `https://felirangelp.github.io/Prediccion_Muerte_Subita_ECG_v1/`

### 3. Verificar

Abre la URL en tu navegador. Deberías ver el dashboard interactivo.

## 🔄 Actualizar el Dashboard

Cada vez que regeneres el dashboard, actualízalo en GitHub Pages:

```bash
# 1. Regenerar el dashboard
python scripts/generate_dashboard.py --output results/dashboard_scd_prediction.html

# 2. Copiar a docs/ para GitHub Pages
cp results/dashboard_scd_prediction.html docs/index.html

# 3. Hacer commit y push
git add docs/index.html
git commit -m "Actualizar dashboard"
git push origin main
```

GitHub Pages se actualizará automáticamente en unos minutos.

## 🛠️ Solución de Problemas

### Error 404

Si ves un error 404:

1. **Verifica que el archivo existe**: `docs/index.html` debe existir
2. **Verifica `.nojekyll`**: El archivo `docs/.nojekyll` debe existir (puede estar vacío)
3. **Espera unos minutos**: GitHub Pages puede tardar hasta 5 minutos en actualizar
4. **Verifica la configuración**: En Settings → Pages, debe estar configurado para `/docs`

### El dashboard no carga correctamente

1. **Abre la consola del navegador** (F12) y revisa errores
2. **Verifica que Plotly se carga**: El dashboard usa Plotly desde CDN
3. **Verifica la conexión a internet**: El dashboard necesita internet para cargar Plotly

### El dashboard se ve mal

1. **Limpia la caché del navegador** (Ctrl+Shift+R o Cmd+Shift+R)
2. **Verifica que el HTML está completo**: El archivo debe tener más de 1MB
3. **Revisa la consola del navegador** para errores de JavaScript

## 📝 Notas Importantes

- El archivo `.nojekyll` es **crítico** para evitar problemas de 404
- GitHub Pages solo sirve archivos estáticos (HTML, CSS, JS)
- El dashboard usa Plotly desde CDN, no requiere archivos locales adicionales
- Los cambios en `docs/` se reflejan automáticamente después de hacer push

## 🔗 Enlaces Útiles

- [Documentación de GitHub Pages](https://docs.github.com/en/pages)
- [Solución de problemas comunes](https://docs.github.com/en/pages/getting-started-with-github-pages/troubleshooting-github-pages)

