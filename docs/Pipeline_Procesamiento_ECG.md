# Pipeline de Procesamiento para Predicción de Muerte Súbita Cardíaca

## Descripción General

Este documento explica el flujo completo del pipeline de procesamiento de señales ECG para la predicción de muerte súbita cardíaca (MSC). El proceso consta de tres etapas principales: **Preprocesamiento**, **Extracción de Características** y **Clasificación**.

## Diagrama de Flujo

```
Señal ECG Raw
    ↓
Preprocesamiento
    ├──→ Filtrado Línea Base
    ├──→ Filtrado Pasa-Bajos
    └──→ Normalización
    ↓
Segmentación
    ↓
Extracción de Características
    ├──→ Método 1: Representaciones Dispersas (Sparse)
    ├──→ Método 2: Fusión Jerárquica (Hierarchical)
    └──→ Método 3: Modelo Híbrido (Hybrid)
    ↓
Clasificación SVM
    ↓
Resultados
```

## Etapas del Procesamiento

### 1. Preprocesamiento

El preprocesamiento es la primera etapa crítica que prepara la señal ECG cruda para el análisis. Esta etapa incluye:

#### 1.1 Filtrado de Línea Base (Baseline Filtering)
- **Frecuencia de corte**: 0.5 Hz
- **Propósito**: Eliminar el componente de línea base (drift) que puede introducir artefactos en la señal
- **Método**: Filtro pasa-altos que elimina componentes de muy baja frecuencia

#### 1.2 Filtrado Pasa-Bajos (Low-Pass Filtering)
- **Frecuencia de corte**: 40 Hz
- **Propósito**: Eliminar ruido de alta frecuencia y artefactos electromagnéticos
- **Método**: Filtro pasa-bajos que preserva las componentes espectrales relevantes del ECG

#### 1.3 Normalización Z-score
- **Propósito**: Estandarizar la amplitud de la señal para que todos los registros tengan la misma escala
- **Método**: 
  ```
  x_normalizado = (x - μ) / σ
  ```
  donde μ es la media y σ es la desviación estándar

#### 1.4 Segmentación
- **Tamaño de ventana**: 30 segundos
- **Propósito**: Dividir las señales largas en segmentos manejables para el análisis
- **Ventaja**: Permite procesar señales de diferentes duraciones de manera uniforme

### 2. Extracción de Características

Una vez preprocesada la señal, se aplican tres métodos diferentes de extracción de características en paralelo:

#### 2.1 Método 1: Representaciones Dispersas (Sparse Representations)
- **Enfoque**: Aprende diccionarios de átomos que representan morfologías características del ECG
- **Técnicas**:
  - **OMP (Orthogonal Matching Pursuit)**: Para la codificación dispersa
  - **k-SVD**: Para el aprendizaje del diccionario
- **Ventajas**:
  - Marco matemáticamente sólido
  - Aprende morfologías directamente de los datos
  - Robusto a variabilidad entre pacientes
- **Rendimiento**: 94.2% accuracy

#### 2.2 Método 2: Fusión Jerárquica (Hierarchical Fusion)
- **Enfoque**: Combina características a múltiples escalas y niveles de abstracción
- **Componentes**:
  - **Características lineales**: Estadísticas tradicionales (media, varianza, etc.)
  - **Características no lineales**: Wavelets, transformadas tiempo-frecuencia
  - **Características de Deep Learning**: TCN-Seq2vec para patrones temporales complejos
- **Ventajas**:
  - Captura información a múltiples escalas
  - Combina lo mejor de métodos tradicionales y modernos
  - Mayor robustez al capturar detalles finos y características globales
- **Rendimiento**: 87.86% accuracy

#### 2.3 Método 3: Modelo Híbrido (Hybrid Model)
- **Enfoque**: Combina elementos de ambos métodos anteriores
- **Propósito**: Explorar sinergias entre representaciones dispersas y fusión jerárquica
- **Rendimiento**: 74.8% accuracy
- **Nota**: Requiere optimización adicional para mejorar su rendimiento

### 3. Clasificación

La etapa final utiliza las características extraídas para realizar la clasificación binaria:

#### 3.1 Clasificador SVM (Support Vector Machine)
- **Kernel**: RBF (Radial Basis Function)
- **Propósito**: Separar las clases Normal vs. Muerte Súbita Cardíaca
- **Ventajas**:
  - Eficiente con características de alta dimensionalidad
  - Buen rendimiento con datos no lineales
  - Menor propensión al sobreajuste

#### 3.2 División de Datos
- **Método**: Train-Test Split estratificado
- **Proporción**: 80% entrenamiento, 20% prueba (o 70% entrenamiento, 30% prueba según el script)
- **Estratificación**: Mantiene la proporción de clases en ambos conjuntos
- **Random State**: 42 (para reproducibilidad)
- **Propósito**: Evaluar el rendimiento del modelo en datos no vistos durante el entrenamiento

#### 3.3 Evaluación de Métricas
Las métricas evaluadas incluyen:
- **Accuracy**: Precisión general del clasificador
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos detectados correctamente
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC, mide la capacidad de discriminación

## Flujo de Datos Completo

1. **Entrada**: Señal ECG cruda (raw) de pacientes
2. **Preprocesamiento**: 
   - Filtrado de línea base → Filtrado pasa-bajos → Normalización → Segmentación
3. **División de Datos**: 
   - División estratificada en conjunto de entrenamiento y prueba (train_test_split)
4. **Extracción Paralela**: 
   - Los tres métodos (Sparse, Hierarchical, Hybrid) procesan la señal segmentada simultáneamente
   - Cada método se entrena en el conjunto de entrenamiento
5. **Clasificación**: 
   - Cada método genera características que alimentan un clasificador SVM
   - Los modelos entrenados se evalúan en el conjunto de prueba
6. **Salida**: 
   - Resultados de clasificación con métricas de rendimiento (accuracy, precision, recall, F1-score, AUC-ROC)
   - Matrices de confusión
   - Curvas ROC
   - Predicción: Normal o Muerte Súbita Cardíaca

## Consideraciones Técnicas

### Ventajas del Pipeline
- **Modularidad**: Cada etapa puede optimizarse independientemente
- **Comparabilidad**: Los tres métodos procesan los mismos datos, permitiendo comparación directa
- **Robustez**: Múltiples enfoques aumentan la confiabilidad del sistema
- **División estratificada**: Garantiza que ambos conjuntos (train/test) mantengan la proporción de clases

### Limitaciones
- **Complejidad computacional**: El procesamiento en paralelo requiere recursos significativos
- **Tiempo de procesamiento**: El preprocesamiento y extracción pueden ser costosos computacionalmente
- **Dependencia de datos**: La calidad del preprocesamiento afecta directamente el rendimiento final
- **Evaluación única**: Se utiliza una sola división train/test, lo que puede no capturar toda la variabilidad de los datos

## Resultados Esperados

Al final del pipeline, se obtienen:
- **Matrices de confusión** para cada método
- **Curvas ROC** que muestran la capacidad de discriminación
- **Métricas de rendimiento** (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- **Comparación** entre los tres métodos para identificar el más efectivo

## Referencias

- Velázquez-González et al. (Sensors 2021): Representaciones Dispersas
- Huang et al. (Symmetry 2025): Fusión Jerárquica

