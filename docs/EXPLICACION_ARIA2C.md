# ⚡ ¿Por qué aria2c muestra 0 procesos?

## 🔍 Explicación

Es **normal** que aria2c muestre 0 procesos en este momento porque:

### 1. El script actual está usando `wget`
- El proceso que está corriendo (`ejecutar_y_monitorear.py` o `eliminar_y_descargar.py`) usa **wget** por defecto
- Estos scripts no tienen la lógica para usar aria2c automáticamente

### 2. aria2c requiere instalación y configuración
- aria2c necesita estar instalado: `brew install aria2`
- El script `descarga_maxima_velocidad.py` detecta aria2c y lo usa si está disponible
- Pero el proceso actual probablemente NO está usando ese script

### 3. Estado actual
- ✅ **84 procesos wget activos**: Funcionando bien
- ❌ **0 procesos aria2c**: No se está usando (normal)

## ⚠️ ¿Es un problema?

**NO es un problema** - wget está funcionando correctamente:
- 84 procesos paralelos descargando
- Progreso: 4.8% (0.79 GB / 16 GB)
- La descarga está avanzando

## 💡 ¿Quieres usar aria2c para más velocidad?

Si quieres usar aria2c (más rápido), necesitas:

1. **Instalar aria2c** (si no está instalado):
   ```bash
   brew install aria2
   ```

2. **Detener el proceso actual**:
   ```bash
   pkill -f "ejecutar_y_monitorear"
   pkill -f "wget.*physionet"
   ```

3. **Ejecutar el script con aria2c**:
   ```bash
   python3 scripts/descarga_maxima_velocidad.py
   ```

Este script:
- Detecta automáticamente si aria2c está disponible
- Usa aria2c si está instalado (más rápido)
- Usa wget como fallback si no está disponible

## 📊 Comparación

| Herramienta | Procesos | Velocidad | Estado |
|------------|---------|-----------|--------|
| **wget** (actual) | 84 procesos | Buena | ✅ Funcionando |
| **aria2c** (opcional) | 150 procesos + 16 conexiones/archivo | Muy buena | ⚠️ Requiere instalación |

## ✅ Conclusión

**Es normal que aria2c muestre 0** - el proceso actual está usando wget y funciona bien. Si quieres máxima velocidad, puedes instalar aria2c y usar `descarga_maxima_velocidad.py`.

