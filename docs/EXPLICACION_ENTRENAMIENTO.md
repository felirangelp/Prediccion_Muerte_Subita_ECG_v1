# 📚 ¿Por qué necesitas datos descargados para entrenar modelos?

## 🔍 Razón Principal

Los modelos de machine learning necesitan **datos reales** para aprender patrones. Sin datos, no hay nada que aprender.

## 📊 ¿Qué datos necesita el script `train_models.py`?

### 1. Archivos `.dat` (Datos de señales ECG)
- **Contenido**: Señales ECG digitales (voltajes medidos)
- **Tamaño**: Archivos grandes (cientos de MB cada uno)
- **Uso**: Extracción de características de la señal

### 2. Archivos `.hea` (Headers)
- **Contenido**: Metadatos de la señal (frecuencia de muestreo, canales, etc.)
- **Tamaño**: Archivos pequeños (KB)
- **Uso**: Configuración y validación de la señal

### 3. Archivos `.atr` (Anotaciones)
- **Contenido**: Etiquetas y anotaciones (latidos, eventos, etc.)
- **Tamaño**: Archivos pequeños (KB)
- **Uso**: Validación y etiquetado

## 🔄 Proceso de Entrenamiento

```python
# 1. Cargar archivos ECG (requiere archivos .dat, .hea, .atr)
signal, metadata = load_ecg_record(record_path, channels=[0])

# 2. Preprocesar señales (filtrado, normalización)
processed_signal = preprocess_ecg_signal(signal, fs=128.0)

# 3. Extraer características
features = extract_features(processed_signal)

# 4. Entrenar modelo con características
model.fit(X_train, y_train)
```

## ❌ ¿Qué pasa si intentas entrenar sin datos?

El script `train_models.py` verifica si los datasets existen:

```python
if not sddb_path.exists() or not nsrdb_path.exists():
    print("❌ Datasets no encontrados. Por favor descarga los datasets primero.")
    return
```

**Resultado**: El script se detiene y no puede continuar.

## ✅ Estado Actual de tu Descarga

Según el monitor:
- **Archivos .dat**: 75/76 (98.7%) ✅
- **Tamaño total**: 0.52 GB / 16 GB (3.2%) 🔄
- **Procesos activos**: 103 procesos wget descargando

**Los archivos grandes (.dat con datos) se están descargando ahora.**

## ⏱️ ¿Cuándo puedes entrenar?

1. **Espera a que termine la descarga** (100% de tamaño: ~16 GB)
2. **Verifica integridad**: `python scripts/validacion_completa.py`
3. **Entonces puedes entrenar**: `python scripts/train_models.py --train-all`

## 💡 Resumen

- **Sin datos** = No hay señales ECG = No hay características = No hay modelo
- **Con datos** = Señales ECG = Características extraídas = Modelo entrenado

**¡Espera a que termine la descarga actual (103 procesos activos) antes de entrenar!**

