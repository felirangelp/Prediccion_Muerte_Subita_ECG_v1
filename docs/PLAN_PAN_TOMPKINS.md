# 📋 Plan de Implementación: Algoritmo Pan-Tompkins Completo

## 📊 Resumen Ejecutivo

Este documento describe el plan para implementar el algoritmo **Pan-Tompkins completo** con visualización paso a paso, detección de ondas P, Q, S, T, y análisis de tacograma (HRV) para el proyecto de predicción de muerte súbita cardíaca.

**Objetivo:** Mejorar la preparación de datos y la visualización mediante la implementación completa del algoritmo Pan-Tompkins, incluyendo detección de todas las ondas del ECG y análisis avanzado de variabilidad de frecuencia cardíaca.

---

## 🔍 Análisis del Estado Actual

### ✅ Lo que Ya Existe

#### 1. Implementación Básica de Pan-Tompkins
- **Ubicación:** `src/preprocessing.py` (función `detect_r_peaks`)
- **Ubicación:** `src/hierarchical_fusion.py` (función `detect_r_peaks_advanced`)
- **Estado:** Implementación simplificada que incluye:
  - Derivada de la señal
  - Cuadrado de la derivada
  - Suavizado con Savitzky-Golay
  - Detección básica de picos R

**Limitaciones:**
- ❌ No usa filtros FIR con `scipy.signal.filter()` (requisito del proyecto)
- ❌ No implementa diferenciación e integración con ventana b apropiada
- ❌ No realiza umbralización estadística sobre señal integrada
- ❌ No visualiza cada paso del algoritmo
- ❌ Implementación incompleta del algoritmo clásico

#### 2. Detección de Picos R
- **Estado:** Funcional pero básico
- **Método:** Usa `scipy.signal.find_peaks` con umbral fijo
- **Limitación:** No usa umbralización estadística adaptativa

#### 3. Cálculo de HRV
- **Ubicación:** `src/preprocessing.py` (función `calculate_hrv_features`)
- **Estado:** Implementación parcial que incluye:
  - Intervalos RR básicos
  - Métricas en dominio del tiempo (mean_rr, std_rr, RMSSD, pNN50)
  - Análisis espectral básico (VLF, LF, HF)

**Limitaciones:**
- ❌ No calcula tacograma completo (gráfica de intervalos RR vs tiempo)
- ❌ No calcula frecuencia cardíaca global en bpm de forma explícita
- ❌ Visualización limitada

#### 4. Visualización
- **Estado:** Dashboard interactivo con Plotly existente
- **Capacidad:** Visualización de señales ECG procesadas
- **Limitación:** No visualiza pasos intermedios de Pan-Tompkins

### ❌ Lo que Falta

#### 1. Pan-Tompkins Completo
- [ ] Implementación completa del algoritmo con todos los pasos
- [ ] Diferenciación usando filtro FIR (`scipy.signal.filter()` con b, a=1)
- [ ] Integración usando filtro FIR con ventana b apropiada
- [ ] Umbralización estadística sobre señal integrada
- [ ] Visualización de cada paso del algoritmo

#### 2. Detección de Ondas P, Q, S, T
- [ ] Estrategia para encontrar pico de onda P
- [ ] Estrategia para encontrar pico de onda Q
- [ ] Estrategia para encontrar pico de onda S
- [ ] Estrategia para encontrar pico de onda T
- [ ] Uso de ventanas alrededor de R (±300-400 ms)
- [ ] Búsqueda de mínimos/máximos en ventanas antes/después de R

#### 3. Tacograma y Frecuencia Cardíaca
- [ ] Cálculo completo de tacograma (intervalos RR vs tiempo)
- [ ] Cálculo de frecuencia cardíaca global en bpm
- [ ] Visualización del tacograma

#### 4. Visualización Completa
- [ ] Visualización paso a paso de Pan-Tompkins
- [ ] Visualización de ondas detectadas (P, Q, R, S, T)
- [ ] Visualización de tacograma
- [ ] Integración con dashboard existente

---

## 🎯 Objetivos de la Implementación

### Objetivo Principal
Implementar el algoritmo Pan-Tompkins completo siguiendo las especificaciones del proyecto:
- Usar `scipy.signal.filter()` con filtros FIR (solo b, a=1)
- Graficar cada uno de los pasos
- Realizar umbralización estadística sobre señal integrada
- Usar `findpeaks` para encontrar picos R
- Detectar ondas P, Q, S, T
- Calcular tacograma y frecuencia cardíaca global

### Objetivos Secundarios
1. **Mejora de Preprocesamiento:**
   - Detección más robusta de picos R
   - Extracción de características adicionales (amplitudes, anchos de ondas)
   - Mejor calidad de datos para entrenamiento

2. **Visualización Educativa:**
   - Comprensión visual del procesamiento de señales
   - Validación de detecciones
   - Material educativo para presentaciones

3. **Análisis Avanzado:**
   - Características adicionales para modelos ML
   - Análisis de HRV más completo
   - Validación de calidad de señal

---

## 📐 Especificaciones Técnicas

### Algoritmo Pan-Tompkins - Pasos Requeridos

#### Paso 1: Preprocesamiento Inicial
- Filtro paso-banda (0.5-40 Hz) - Ya existe en `preprocessing_unified.py`
- Normalización - Ya existe

#### Paso 2: Diferenciación
- **Método:** Filtro FIR usando `scipy.signal.filter()`
- **Coeficientes:** b = [-1, -2, 0, 2, 1] / 8 (o similar)
- **Nota:** Como es filtro FIR, a = 1
- **Propósito:** Enfatizar picos R y reducir ruido de baja frecuencia

#### Paso 3: Cuadrado
- Elevar señal diferenciada al cuadrado
- **Propósito:** Hacer todos los valores positivos y amplificar picos

#### Paso 4: Integración
- **Método:** Filtro FIR usando `scipy.signal.filter()`
- **Ventana b:** Ventana móvil de N muestras (típicamente N = fs * 0.15)
- **Coeficientes:** b = [1, 1, 1, ..., 1] / N (ventana rectangular)
- **Nota:** Como es filtro FIR, a = 1
- **Propósito:** Suavizar señal y reducir falsos positivos

#### Paso 5: Umbralización Estadística (MEJORADA)
- Calcular umbral adaptativo basado en estadísticas de señal integrada
- **Método mejorado:** Usa percentil 65% en lugar de media+std para mayor robustez ante outliers
- **Cálculo:** `umbral = percentil_65 + k * (max - percentil_65) * 0.3`, con k típicamente 0.5
- **Límites adaptativos:** El umbral se mantiene entre 20% y 60% del máximo de la señal integrada
- **Propósito:** Detección robusta de picos R con mayor resistencia a valores atípicos

#### Paso 6: Detección de Picos R
- Usar `scipy.signal.find_peaks` sobre señal umbralizada
- **Parámetros:**
  - `height`: Umbral estadístico calculado
  - `prominence`: Prominencia adaptativa (15% del rango de señal integrada)
  - `width`: Ancho mínimo de 20ms
  - `distance`: Distancia mínima de 200ms entre picos

#### Paso 7: Post-procesamiento y Refinamiento (MEJORA IMPLEMENTADA)
- **Búsqueda del máximo absoluto:** Para cada pico detectado en la señal integrada, buscar el máximo absoluto 
  en la señal original dentro de una ventana de 150ms alrededor del pico detectado
- **Validación de prominencia:** Verificar que el pico refinado tenga prominencia relativa ≥30% del rango de la señal
- **Validación de máximo local:** Verificar que el pico sea el máximo en su vecindad inmediata (20ms)
- **Propósito:** Corregir desplazamientos causados por la integración y asegurar que los picos R coincidan 
  exactamente con los máximos reales del complejo QRS, evitando seleccionar pequeñas deflexiones

### Detección de Ondas P, Q, S, T

#### Estrategia General
Para cada pico R detectado:
1. Definir ventana de búsqueda: R ± 300-400 ms (o ± 0.25-0.4 * RR_interval)
2. Buscar ondas dentro de la ventana

#### Onda Q
- **Ubicación:** Antes del pico R
- **Ventana:** [R - 0.1*RR, R]
- **Método:** Primer mínimo local antes de R
- **Validación:** Q debe estar dentro de complejo QRS

#### Onda S
- **Ubicación:** Después del pico R
- **Ventana:** [R, R + 0.1*RR]
- **Método:** Primer mínimo local después de R
- **Validación:** S debe estar dentro de complejo QRS

#### Onda T
- **Ubicación:** Después del complejo QRS
- **Ventana:** [R + 0.2*RR, R + 0.6*RR]
- **Método:** Máximo (o mínimo si invertida) en la ventana
- **Validación:** T no debe solaparse con siguiente QRS

#### Onda P
- **Ubicación:** Antes del complejo QRS
- **Ventana:** [R - 0.4*RR, R - 0.1*RR]
- **Método:** Máximo (o mínimo si invertida) en la ventana
- **Validación:** P no debe solaparse con QRS anterior

### Tacograma y Frecuencia Cardíaca

#### Tacograma
- **Definición:** Gráfica de intervalos RR (en ms) vs tiempo
- **Cálculo:**
  - Para cada par de picos R consecutivos: RR_i = (R_{i+1} - R_i) / fs * 1000
  - Tiempo asociado: t_i = R_i / fs
- **Visualización:** Scatter plot o línea de intervalos RR vs tiempo

#### Frecuencia Cardíaca Global
- **Cálculo:** HR_global = 60 / (mean_RR / 1000) bpm
  - Donde mean_RR es el promedio de intervalos RR en ms
- **Alternativa:** HR_global = 60000 / mean_RR bpm
- **Validación:** Filtrar intervalos RR anómalos (300-2000 ms)

---

## 🏗️ Estructura Propuesta

### Nuevos Módulos a Crear

```
src/
├── pan_tompkins_complete.py          # Implementación completa del algoritmo
├── ecg_wave_detection.py             # Detección de ondas P, Q, S, T
└── tachogram_analysis.py             # Análisis de tacograma y HRV completo

scripts/
├── visualize_pan_tompkins.py         # Script de visualización paso a paso
├── demo_pan_tompkins.py              # Demo interactivo del algoritmo
└── test_pan_tompkins.py              # Tests unitarios

docs/
└── PAN_TOMPKINS_IMPLEMENTATION.md    # Documentación técnica (este archivo)

results/
└── pan_tompkins_visualizations/      # Gráficas generadas (opcional)
```

### Detalles de Cada Módulo

#### 1. `src/pan_tompkins_complete.py`

**Funciones principales:**
```python
def pan_tompkins_complete(ecg_signal, fs, visualize=False):
    """
    Implementación completa del algoritmo Pan-Tompkins
    
    Args:
        ecg_signal: Señal ECG 1D
        fs: Frecuencia de muestreo
        visualize: Si retornar señales intermedias para visualización
    
    Returns:
        dict con:
            - r_peaks: Índices de picos R detectados
            - signals: Diccionario con señales intermedias (si visualize=True)
            - thresholds: Umbrales utilizados
    """
    pass

def differentiate_signal(signal, fs):
    """
    Diferenciación usando filtro FIR
    
    Returns:
        señal_diferenciada, coeficientes_b
    """
    pass

def integrate_signal(signal, fs, window_size=None):
    """
    Integración usando filtro FIR con ventana móvil
    
    Returns:
        señal_integrada, coeficientes_b
    """
    pass

def statistical_threshold(signal, method='adaptive'):
    """
    Umbralización estadística sobre señal integrada
    
    Returns:
        umbral, señal_umbralizada
    """
    pass
```

#### 2. `src/ecg_wave_detection.py`

**Funciones principales:**
```python
def detect_all_waves(ecg_signal, r_peaks, fs, rr_intervals=None):
    """
    Detectar ondas P, Q, S, T basado en picos R
    
    Args:
        ecg_signal: Señal ECG 1D
        r_peaks: Índices de picos R
        fs: Frecuencia de muestreo
        rr_intervals: Intervalos RR (opcional, se calculan si no se proporcionan)
    
    Returns:
        dict con:
            - p_waves: Índices de ondas P
            - q_waves: Índices de ondas Q
            - s_waves: Índices de ondas S
            - t_waves: Índices de ondas T
            - wave_features: Características de cada onda
    """
    pass

def detect_q_wave(ecg_signal, r_peak, fs, rr_interval):
    """Detectar onda Q antes de R"""
    pass

def detect_s_wave(ecg_signal, r_peak, fs, rr_interval):
    """Detectar onda S después de R"""
    pass

def detect_t_wave(ecg_signal, r_peak, fs, rr_interval):
    """Detectar onda T después de QRS"""
    pass

def detect_p_wave(ecg_signal, r_peak, fs, rr_interval):
    """Detectar onda P antes de QRS"""
    pass
```

#### 3. `src/tachogram_analysis.py`

**Funciones principales:**
```python
def calculate_tachogram(r_peaks, fs):
    """
    Calcular tacograma completo
    
    Returns:
        dict con:
            - rr_intervals: Array de intervalos RR (ms)
            - time_points: Array de tiempos asociados (s)
            - tachogram_data: DataFrame con datos del tacograma
    """
    pass

def calculate_global_heart_rate(rr_intervals):
    """
    Calcular frecuencia cardíaca global
    
    Returns:
        heart_rate_bpm: Frecuencia cardíaca en bpm
    """
    pass

def filter_rr_intervals(rr_intervals, min_rr=300, max_rr=2000):
    """
    Filtrar intervalos RR anómalos
    
    Returns:
        rr_filtered: Intervalos RR filtrados
        valid_indices: Índices de intervalos válidos
    """
    pass
```

#### 4. `scripts/visualize_pan_tompkins.py`

**Funciones principales:**
```python
def visualize_pan_tompkins_steps(signals_dict, fs, r_peaks=None, output_file=None):
    """
    Visualizar todos los pasos del algoritmo Pan-Tompkins
    
    Args:
        signals_dict: Diccionario con señales de cada paso
        fs: Frecuencia de muestreo
        r_peaks: Picos R detectados (opcional)
        output_file: Archivo de salida (opcional)
    
    Returns:
        fig: Figura de Plotly
    """
    pass

def visualize_detected_waves(ecg_signal, waves_dict, fs, duration=10, output_file=None):
    """
    Visualizar ondas detectadas (P, Q, R, S, T)
    
    Args:
        ecg_signal: Señal ECG original
        waves_dict: Diccionario con ondas detectadas
        fs: Frecuencia de muestreo
        duration: Duración a visualizar (segundos)
        output_file: Archivo de salida (opcional)
    
    Returns:
        fig: Figura de Plotly
    """
    pass

def visualize_tachogram(tachogram_data, output_file=None):
    """
    Visualizar tacograma
    
    Args:
        tachogram_data: Datos del tacograma
        output_file: Archivo de salida (opcional)
    
    Returns:
        fig: Figura de Plotly
    """
    pass
```

---

## 🔄 Integración con Proyecto Existente

### Modificaciones a Módulos Existentes

#### 1. `src/preprocessing.py`
- **Opción A:** Mantener función `detect_r_peaks` existente para compatibilidad
- **Opción B:** Actualizar para usar `pan_tompkins_complete` internamente
- **Recomendación:** Opción A (mantener compatibilidad, agregar nueva función)

#### 2. `src/preprocessing_unified.py`
- **No requiere cambios:** Ya tiene filtrado y normalización
- **Integración:** Usar señales preprocesadas como entrada a Pan-Tompkins

#### 3. `src/hierarchical_fusion.py`
- **Actualizar:** Función `detect_r_peaks_advanced` para usar nueva implementación
- **Mantener:** Compatibilidad con código existente

#### 4. `scripts/generate_dashboard.py`
- **Agregar:** Nueva sección "Análisis Pan-Tompkins" al dashboard
- **Incluir:**
  - Visualización paso a paso del algoritmo
  - Visualización de ondas detectadas
  - Visualización de tacograma
  - Métricas de HRV mejoradas

### Flujo de Integración

```
Señal ECG Original
    ↓
preprocessing_unified.py (filtrado, normalización)
    ↓
pan_tompkins_complete.py (detección R completa)
    ↓
ecg_wave_detection.py (detección P, Q, S, T)
    ↓
tachogram_analysis.py (tacograma y HRV)
    ↓
Extracción de características mejoradas
    ↓
Modelos ML (Sparse, Hierarchical, Hybrid)
```

---

## 📊 Visualización Propuesta

### Dashboard - Nueva Sección: "Análisis Pan-Tompkins"

#### Subsección 1: Pasos del Algoritmo
- **Gráfica 1:** Señal ECG original
- **Gráfica 2:** Señal diferenciada
- **Gráfica 3:** Señal al cuadrado
- **Gráfica 4:** Señal integrada
- **Gráfica 5:** Señal umbralizada
- **Gráfica 6:** Picos R detectados sobre señal original

**Layout:** 2 columnas x 3 filas (subplots)

#### Subsección 2: Ondas Detectadas
- **Gráfica:** Señal ECG con ondas P, Q, R, S, T marcadas
- **Colores:**
  - P: Azul
  - Q: Verde
  - R: Rojo
  - S: Naranja
  - T: Púrpura
- **Leyenda:** Interactiva con Plotly

#### Subsección 3: Tacograma y HRV
- **Gráfica 1:** Tacograma (RR intervals vs tiempo)
- **Gráfica 2:** Histograma de intervalos RR
- **Métricas:**
  - Frecuencia cardíaca global (bpm)
  - Media de intervalos RR (ms)
  - Desviación estándar de RR (ms)
  - RMSSD, pNN50, etc.

---

## 🧪 Plan de Pruebas

### Tests Unitarios

#### 1. Tests de Pan-Tompkins
- [ ] Test de diferenciación con señal conocida
- [ ] Test de integración con señal conocida
- [ ] Test de umbralización estadística
- [ ] Test de detección de picos R en señal sintética
- [ ] Validación con señales de PhysioNet con anotaciones

#### 2. Tests de Detección de Ondas
- [ ] Test de detección de Q antes de R
- [ ] Test de detección de S después de R
- [ ] Test de detección de T después de QRS
- [ ] Test de detección de P antes de QRS
- [ ] Validación con anotaciones de PhysioNet (si disponibles)

#### 3. Tests de Tacograma
- [ ] Test de cálculo de intervalos RR
- [ ] Test de filtrado de intervalos anómalos
- [ ] Test de cálculo de frecuencia cardíaca global
- [ ] Validación con valores esperados

### Tests de Integración
- [ ] Integración con `preprocessing_unified.py`
- [ ] Integración con modelos ML existentes
- [ ] Integración con dashboard
- [ ] Validación end-to-end con datos reales

---

## 📅 Plan de Implementación

### Fase 1: Implementación Core (Semana 1)

#### Día 1-2: Pan-Tompkins Completo
- [ ] Implementar `pan_tompkins_complete.py`
- [ ] Implementar diferenciación con filtro FIR
- [ ] Implementar integración con filtro FIR
- [ ] Implementar umbralización estadística
- [ ] Tests unitarios básicos

#### Día 3-4: Detección de Ondas
- [ ] Implementar `ecg_wave_detection.py`
- [ ] Implementar detección de Q, S
- [ ] Implementar detección de T
- [ ] Implementar detección de P
- [ ] Tests unitarios

#### Día 5: Tacograma
- [ ] Implementar `tachogram_analysis.py`
- [ ] Implementar cálculo de tacograma
- [ ] Implementar cálculo de frecuencia cardíaca global
- [ ] Tests unitarios

### Fase 2: Visualización (Semana 2)

#### Día 1-2: Scripts de Visualización
- [ ] Implementar `visualize_pan_tompkins.py`
- [ ] Crear visualización paso a paso
- [ ] Crear visualización de ondas detectadas
- [ ] Crear visualización de tacograma

#### Día 3-4: Integración con Dashboard
- [ ] Agregar sección al dashboard
- [ ] Integrar visualizaciones
- [ ] Agregar métricas de HRV mejoradas
- [ ] Tests de integración

#### Día 5: Demo y Documentación
- [ ] Crear `demo_pan_tompkins.py`
- [ ] Documentar uso de funciones
- [ ] Crear ejemplos de uso
- [ ] Actualizar README

### Fase 3: Validación y Refinamiento (Semana 3)

#### Día 1-2: Validación con Datos Reales
- [ ] Probar con señales de SDDB
- [ ] Probar con señales de NSRDB
- [ ] Comparar con anotaciones de PhysioNet (si disponibles)
- [ ] Ajustar parámetros según resultados

#### Día 3-4: Optimización
- [ ] Optimizar rendimiento
- [ ] Mejorar robustez ante ruido
- [ ] Refinar detección de ondas
- [ ] Mejorar visualizaciones

#### Día 5: Documentación Final
- [ ] Completar documentación técnica
- [ ] Crear guía de usuario
- [ ] Actualizar documentación del proyecto
- [ ] Preparar ejemplos para presentación

---

## 🎯 Criterios de Éxito

### Funcionalidad
- ✅ Implementación completa de Pan-Tompkins con filtros FIR
- ✅ Detección correcta de picos R (≥95% precisión en señales limpias)
- ✅ Detección de ondas P, Q, S, T (≥80% precisión)
- ✅ Cálculo correcto de tacograma y frecuencia cardíaca
- ✅ Visualización completa y clara

### Calidad
- ✅ Código bien documentado y comentado
- ✅ Tests unitarios con cobertura ≥80%
- ✅ Integración sin romper funcionalidad existente
- ✅ Rendimiento aceptable (procesamiento <1s para 10s de señal)

### Usabilidad
- ✅ Fácil de usar desde otros módulos
- ✅ Visualizaciones claras y educativas
- ✅ Documentación completa
- ✅ Ejemplos de uso disponibles

---

## 🔧 Consideraciones Técnicas

### Dependencias
- **Ya disponibles:**
  - `numpy`, `scipy` (filtros, find_peaks)
  - `plotly` (visualización)
  - `pandas` (tacograma como DataFrame)
- **No requiere nuevas dependencias**

### Rendimiento
- **Optimizaciones:**
  - Usar operaciones vectorizadas de NumPy
  - Evitar loops cuando sea posible
  - Cachear resultados intermedios si es necesario
- **Complejidad esperada:** O(n) para procesamiento de señal

### Compatibilidad
- **Python:** 3.8+ (compatible con proyecto actual)
- **Plataforma:** Compatible con MacBook M1 (ya configurado)
- **Integración:** Compatible con código existente

### Manejo de Errores
- Validación de entrada (señal no vacía, fs > 0)
- Manejo de casos edge (sin picos R detectados, señales muy ruidosas)
- Mensajes de error claros y útiles
- Logging para debugging

---

## 📚 Referencias Técnicas

### Algoritmo Pan-Tompkins
- **Paper original:** Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm"
- **Implementación de referencia:** Varias implementaciones en Python disponibles
- **Especificaciones del proyecto:** Uso de `filter()` con filtros FIR

### Detección de Ondas ECG
- **Métodos comunes:** Búsqueda de mínimos/máximos locales
- **Ventanas adaptativas:** Basadas en intervalos RR
- **Validación:** Evitar solapamiento entre ondas

### Análisis de HRV
- **Estándares:** Task Force of the European Society of Cardiology
- **Métricas:** Time-domain y frequency-domain
- **Tacograma:** Visualización estándar de variabilidad RR

---

## 🚀 Próximos Pasos Inmediatos

### Para Comenzar la Implementación:

1. **Revisar especificaciones del proyecto:**
   - Confirmar requisitos exactos de filtros FIR
   - Verificar formato de salida esperado

2. **Preparar entorno:**
   - Crear branch de desarrollo: `git checkout -b feature/pan-tompkins-complete`
   - Crear estructura de archivos propuesta

3. **Implementar en orden:**
   - Fase 1: Pan-Tompkins completo (base)
   - Fase 2: Detección de ondas (extensión)
   - Fase 3: Tacograma (análisis)
   - Fase 4: Visualización (presentación)

4. **Validar progresivamente:**
   - Tests después de cada función
   - Validación con datos reales
   - Integración incremental

---

## 📝 Notas Adicionales

### Ventajas de esta Implementación
1. **Mejora de calidad de datos:** Detección más robusta de características
2. **Características adicionales:** Más features para modelos ML
3. **Visualización educativa:** Mejor comprensión del procesamiento
4. **Validación:** Verificación visual de detecciones
5. **Extensibilidad:** Base para futuras mejoras

### Posibles Extensiones Futuras
- Detección de arritmias específicas
- Análisis de morfología de ondas
- Detección de segmentos ST
- Análisis de variabilidad de ondas T
- Integración con deep learning para detección

---

## 🔗 Archivos Relacionados

- `docs/PLAN_PROXIMOS_PASOS.md` - Plan general del proyecto
- `docs/ENTRENAMIENTO_MODELOS.md` - Documentación de modelos
- `src/preprocessing.py` - Preprocesamiento actual
- `src/hierarchical_fusion.py` - Implementación actual de detección R
- `scripts/generate_dashboard.py` - Dashboard existente

---

**Última actualización:** [Fecha a completar]  
**Estado:** 📋 Planificación completada - Listo para implementación  
**Prioridad:** Media-Alta (mejora significativa de preprocesamiento y visualización)

