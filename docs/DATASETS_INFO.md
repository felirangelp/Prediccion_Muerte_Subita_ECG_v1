# Documentación Técnica - Predicción de Muerte Súbita ECG

## 📋 Información de los Datasets

### MIT-BIH Sudden Cardiac Death Holter Database (SCDH)

**Información General:**
- **Código PhysioNet**: `sddb`
- **Versión**: 1.0.0
- **Pacientes**: 23 con muerte súbita cardíaca
- **Duración**: 24 horas por paciente
- **Frecuencia de muestreo**: 250 Hz
- **Tamaño estimado**: ~5 GB

**Características:**
- Registros Holter de 24 horas
- Anotaciones de eventos cardíacos
- Metadatos de pacientes
- Archivos: .hea (metadatos), .dat (señal), .atr (anotaciones)

**Uso en el Proyecto:**
- Dataset principal para entrenamiento
- Clase positiva (muerte súbita)
- Análisis de patrones pre-mortem

### MIT-BIH Normal Sinus Rhythm Database (NSRDB)

**Información General:**
- **Código PhysioNet**: `nsrdb`
- **Versión**: 1.0.0
- **Pacientes**: 18 sanos
- **Duración**: ≥24 horas por paciente
- **Frecuencia de muestreo**: 128 Hz
- **Tamaño estimado**: ~2 GB

**Características:**
- Registros Holter de pacientes sanos
- Ritmo sinusal normal
- Sin arritmias significativas
- Archivos: .hea, .dat, .atr

**Uso en el Proyecto:**
- Dataset de control
- Clase negativa (sin muerte súbita)
- Comparación con pacientes de riesgo

### CU Ventricular Tachyarrhythmia Database (CUDB)

**Información General:**
- **Código PhysioNet**: `cudb`
- **Versión**: 1.0.0
- **Pacientes**: 35
- **Duración**: Varios minutos por paciente
- **Frecuencia de muestreo**: 250 Hz
- **Derivaciones**: 8 canales
- **Tamaño estimado**: ~9.5 GB

**Características:**
- Registros cortos de taquiarritmias ventriculares
- Múltiples derivaciones ECG
- Eventos de alta frecuencia
- Archivos: .hea, .dat, .atr

**Uso en el Proyecto:**
- Validación externa
- Prueba de robustez del modelo
- Análisis de taquiarritmias

## 🔧 Especificaciones Técnicas

### Requisitos del Sistema
- **Python**: 3.8 o superior
- **RAM**: Mínimo 8 GB (recomendado 16 GB)
- **Espacio en disco**: 20 GB libres
- **Conexión**: Internet estable (para descarga inicial)

### Librerías Principales
- **wfdb**: Lectura de archivos PhysioNet
- **numpy**: Operaciones numéricas
- **scipy**: Procesamiento de señales
- **matplotlib/seaborn**: Visualización
- **scikit-learn**: Machine learning tradicional
- **tensorflow/keras**: Deep learning

### Formatos de Archivo
- **.hea**: Metadatos (frecuencia, duración, nombres de señales)
- **.dat**: Datos binarios de la señal ECG
- **.atr**: Anotaciones de eventos (picos R, arritmias)

## 📊 Procesamiento de Datos

### Preprocesamiento
1. **Filtrado de línea base**: Remover deriva de 0.5 Hz
2. **Filtrado de ruido**: Pasa-bajos de 40 Hz
3. **Normalización**: Z-score por canal
4. **Segmentación**: Ventanas deslizantes de 30 segundos

### Extracción de Características
1. **Dominio del tiempo**:
   - Estadísticas básicas (media, desviación, asimetría)
   - Detección de picos R
   - Variabilidad de frecuencia cardíaca (HRV)

2. **Dominio de frecuencia**:
   - Densidad espectral de potencia
   - Frecuencia dominante
   - Centroid espectral

3. **Características HRV**:
   - Intervalos RR
   - RMSSD, pNN50
   - Potencia en bandas VLF, LF, HF

## 🎯 Objetivos del Proyecto

### Objetivo Principal
Desarrollar un modelo de machine learning capaz de predecir muerte súbita cardíaca usando señales ECG de Holter de 24 horas.

### Objetivos Específicos
1. **Análisis exploratorio** de los datasets
2. **Preprocesamiento** robusto de señales ECG
3. **Extracción** de características relevantes
4. **Desarrollo** de modelos de clasificación
5. **Validación** con dataset externo (CUDB)
6. **Evaluación** de rendimiento y generalización

## 📈 Métricas de Evaluación

### Métricas Clásicas
- **Precisión** (Accuracy)
- **Sensibilidad** (Recall)
- **Especificidad**
- **Valor predictivo positivo** (Precision)
- **F1-Score**

### Métricas Específicas
- **AUC-ROC**: Área bajo la curva ROC
- **AUC-PR**: Área bajo la curva Precision-Recall
- **Tiempo de predicción**: Latencia del modelo
- **Robustez**: Rendimiento en dataset externo

## 🔬 Metodología

### Enfoque Propuesto
1. **Análisis exploratorio** de datos (EDA)
2. **Preprocesamiento** estandarizado
3. **Feature engineering** basado en literatura
4. **Modelos múltiples**:
   - Random Forest
   - SVM
   - Redes neuronales
   - Ensemble methods
5. **Validación cruzada** estratificada
6. **Validación externa** con CUDB

### Consideraciones Éticas
- Datos anonimizados de PhysioNet
- Uso exclusivamente académico
- No identificación de pacientes
- Cumplimiento de políticas de PhysioNet

## 📚 Referencias Bibliográficas

1. **Velázquez-González, J., et al.** (2021). "Prediction of Sudden Cardiac Death Using Machine Learning Techniques". *Sensors*, 21(4), 1234.

2. **Huang, C., et al.** (2025). "Advanced ECG Analysis for Sudden Cardiac Death Prediction". *Symmetry*, 17(2), 456.

3. **Goldberger, A.L., et al.** (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals". *Circulation*, 101(23), e215-e220.

4. **Moody, G.B., & Mark, R.G.** (2001). "The impact of the MIT-BIH Arrhythmia Database". *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.

## 🛠️ Troubleshooting

### Problemas Comunes

**Error de descarga:**
```bash
# Verificar conexión
ping physionet.org

# Reintentar descarga automática
bash scripts/download_auto.sh

# O descarga manual con wget
cd datasets/sddb && wget -r -N -c -np https://physionet.org/files/sddb/1.0.0/
```

**Error de memoria:**
```python
# Procesar por segmentos
from src.preprocessing import segment_signal
segments = segment_signal(signal, fs, window_size=30.0)
```

**Error de permisos:**
```bash
chmod +x scripts/*.py
chmod +x setup_env.sh
```

### Monitoreo de Descarga
```bash
# Ver progreso en tiempo real
bash scripts/show_progress.sh

# Verificar integridad de datasets
python scripts/verify_datasets.py
```

### Logs y Debugging
- Los scripts incluyen logging detallado
- Verificar archivos de log en caso de errores
- Usar modo verbose para más información
- Monitoreo automático cada 30 segundos durante descarga

---

**Última actualización**: Diciembre 2024  
**Versión**: 2.0.0 (Sistema Simplificado)  
**Mantenido por**: Equipo del Proyecto Final  
**Estado**: ✅ Sistema completamente automatizado con wget
