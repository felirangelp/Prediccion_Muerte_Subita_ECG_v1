# Documento Consolidado - Revisión Comparativa de Técnicas de Extracción de Características en ECG para la Predicción Temprana de Muerte Súbita Cardíaca

**Autores:** Felipe Rangel Perez, Nicolas Torres Paez  
**Institución:** Pontificia Universidad Javeriana, Bogotá D.C., Colombia  
**Fecha:** 26/10/2025

---

## Resumen

La muerte súbita cardíaca (MSC) representa un desafío significativo en la cardiología moderna debido a su naturaleza abrupta e impredecible. La identificación temprana de pacientes en riesgo es crucial para la intervención preventiva. El electrocardiograma (ECG), al registrar la actividad eléctrica del corazón, contiene información vital que puede ser explotada mediante técnicas avanzadas de procesamiento de señales. 

Esta revisión de literatura realiza un análisis comparativo de dos enfoques recientes para la predicción de la MSC. El primer método, propuesto por Velázquez-González et al., se basa en el uso de representaciones dispersas para extraer características robustas de la señal de ECG. El segundo, desarrollado por Huang et al., implementa una estrategia de fusión jerárquica de características, combinando descriptores multi-escala obtenidos a través de la transformada wavelet.

Se evalúan y comparan ambas metodologías en términos de su pre-procesamiento, extracción de características, y rendimiento de clasificación, utilizando bases de datos públicas estándar. El análisis revela que, si bien ambos métodos alcanzan una alta precisión, el enfoque de fusión de características ofrece una mayor robustez al capturar información tanto local como global de la morfología del ECG. Finalmente, se discuten las implicaciones de estos hallazgos y se proponen futuras líneas de investigación.

---

## I. INTRODUCCIÓN

La muerte súbita cardíaca (MSC) es un evento fatal inesperado de origen cardíaco que ocurre en un corto período de tiempo, generalmente dentro de la primera hora desde el inicio de los síntomas. Constituye uno de los mayores problemas de salud pública a nivel mundial. La principal causa subyacente de la MSC son las arritmias ventriculares malignas, como la fibrilación ventricular. Ante este panorama, la estratificación del riesgo se ha convertido en una prioridad clínica, buscando identificar a aquellos individuos en la población general que tienen una mayor probabilidad de sufrir un evento de este tipo para poder aplicar terapias preventivas.

La herramienta de diagnóstico no invasiva más utilizada para evaluar la función cardíaca es el electrocardiograma (ECG). Esta señal biomédica ofrece una ventana a la fisiología eléctrica del corazón, permitiendo detectar anomalías asociadas con un mayor riesgo arrítmico. Sin embargo, los marcadores de riesgo en el ECG de superficie a menudo son sutiles y no son fácilmente detectables mediante la inspección visual. Por ello, el procesamiento avanzado de señales y el aprendizaje automático se han posicionado como herramientas fundamentales para extraer patrones complejos y desarrollar modelos predictivos robustos.

El estado del arte actual ha explorado diversas técnicas, desde el análisis de la variabilidad de la frecuencia cardíaca (HRV) hasta el estudio de la morfología de ondas específicas como la onda T. A continuación se detalla algunas técnicas que se están haciendo basado en las propuestas específicas de los autores:

### 1. Detección de SCD mediante Representaciones Dispersas (Sparse Representations)

- **Extracción Dinámica de Características**
- **Algoritmos Orthogonal Matching Pursuit (OMP) y k-Singular Value Decomposition (k-SVD)**
- **Clasificación Flexible**: En lugar de depender de clasificadores complejos, se utiliza el vector de coeficientes α de la representación dispersa como vector de características.
- **Esquema Multi-Clase**: Para abordar el sesgo de la clasificación binaria tradicional

### 2. Predicción de SCD mediante Fusión Jerárquica de Características

- **Extracción de Características Multi-Nivel**: 
  - Características Lineales
  - Características No Lineales
  - Características de Aprendizaje Profundo
- **Fusión Jerárquica**
- **Clasificación**
- **Extensión del Horizonte de Predicción**

### 3. Acciones Comunes y Preparación de Datos

- Segmentación de Señales
- Muestreo y Normalización
- Reducción de Ruido
- Evaluación y Validación

### Desafíos Actuales

A pesar de los progresos realizados, aún quedan retos importantes:

1. **Sesgo en la Clasificación Binaria Tradicional**: Un inconveniente principal del esquema de clasificación binaria común (Normal versus SCD) es que puede generar sesgos

2. **Horizonte de Predicción Limitado**: La mayoría de los métodos se han concentrado en ventanas de predicción a corto plazo, analizando las señales solo minutos antes del inicio de la Fibrilación Ventricular (VF)

3. **Silos Metodológicos**: Las metodologías actuales tienden a operar en compartimentos. Se basan exclusivamente en características diseñadas manualmente (basadas en conocimiento fisiológico) o dependen puramente de representaciones de deep learning (de extremo a extremo)

Este trabajo se centra en la revisión y comparación de dos metodologías recientes que abordan este problema desde perspectivas diferentes pero complementarias. El objetivo es analizar críticamente sus contribuciones, ventajas y limitaciones, en el contexto de la pipeline de procesamiento de señales biomédicas estudiada en nuestro curso.

---

## II. MÉTODOS

Ambos trabajos analizados utilizan un marco metodológico común que se alinea con los estándares del procesamiento de señales biomédicas: una etapa de preparación de los datos y pre-procesamiento, seguida de la extracción de características, y finalmente, la clasificación y validación.

### A. Bases de Datos

Las dos investigaciones validan sus algoritmos utilizando registros de ECG de larga duración (Holter) provenientes de la plataforma PhysioNet, un estándar de facto en la investigación cardiológica. Específicamente, emplean dos bases de datos:

- **MIT-BIH Sudden Cardiac Death Holter Database (sddb)**: Contiene registros de 23 sujetos que experimentaron MSC. Este conjunto de datos sirve como la "clase positiva" o grupo de riesgo.

- **MIT-BIH Normal Sinus Rhythm Database (nsrdb)**: Incluye registros de 18 individuos sanos sin arritmias significativas, funcionando como la "clase negativa" o grupo de control.

El uso de estas bases de datos estandarizadas es fundamental, ya que permite la reproducibilidad y la comparación objetiva entre diferentes metodologías.

#### Tabla 1: Características de Adquisición - Método 1 (Velázquez-González et al.)

| Aspecto de la Adquisición | Detalles Específicos |
|---------------------------|----------------------|
| **Bases de Datos Fuente** | MIT/BIH Sudden Cardiac Death Holter (SCDH) y MIT/BIH Normal Sinus Rhythm (NSR) |
| **Tipo de Señal Adquirida** | Señales de **Electrocardiograma (ECG)** |
| **Frecuencias de Muestreo** | Las señales de ECG de la base **NSR** fueron adquiridas a **128 Hz**. Las señales de ECG de la base **SCDH** fueron adquiridas a **250 Hz** |
| **Digitalización** | Las señales fueron digitalizadas con un **conversor analógico-digital de 12 bits** |
| **Contexto de Adquisición** | La base SCDH contiene registros de **24 horas** de sujetos que sufrieron SCD (Muerte Súbita Cardíaca) |

#### Tabla 2: Características de Adquisición - Método 2 (Huang et al.)

| Aspecto de la Adquisición | Detalles Específicos |
|---------------------------|----------------------|
| **Bases de Datos Fuente** | Sudden Cardiac Death Holter Database (SCDH) y Normal Sinus Rhythm (NSR) database (ambas de MIT-BIH) |
| **Tipo de Señal Adquirida** | Señales de **Electrocardiograma (ECG)** no invasivas |
| **Frecuencias de Muestreo** | Las grabaciones de la base **SCDH** fueron muestreadas a **250 Hz**. Los registros de la base **NSR** fueron muestreados a **128 Hz** |
| **Duración de la Adquisición** | La base SCDH contiene grabaciones de ECG de **largo plazo**, con duraciones que varían desde varias horas hasta **24 horas** por registro |
| **Configuración de Derivación** | Para el análisis, se utiliza la **primera derivación de las señales de doble derivación** provistas en la base SCDH |
| **Implicación Tecnológica** | Se menciona el avance de los **dispositivos *wearable* portátiles de ECG de una sola derivación** como ideales para el monitoreo continuo y ambulatorio |

### B. Enfoque 1: Identificación Basada en Representaciones Dispersas (Velázquez-González et al.)

Esta metodología se centra en la idea de que una señal puede ser representada eficientemente por una combinación lineal de unos pocos elementos (átomos) de un diccionario predefinido o aprendido.

#### 1. Pre-procesamiento

Los registros de ECG son primero filtrados para eliminar el ruido y las interferencias. Se aplica un filtro paso banda para atenuar la desviación de la línea base y el ruido de alta frecuencia, asegurando que el análisis se centre en el contenido frecuencial relevante de la señal cardíaca.

#### 2. Extracción de Características

El núcleo de este método es el uso de **representaciones dispersas**. En lugar de usar características tradicionales, los autores construyen "diccionarios" de formas de onda de ECG. Luego, para cada segmento de la señal de un paciente, buscan la combinación más "simple" (dispersa) de elementos del diccionario que pueda reconstruirla. Los coeficientes de esta combinación se convierten en el vector de características del paciente, capturando su morfología cardíaca de manera compacta y robusta.

#### 3. Clasificación

Con los vectores de características generados, se entrena un clasificador de Máquinas de Soporte Vectorial (SVM, por sus siglas en inglés) para distinguir entre los pacientes del grupo de MSC y los del grupo de control sano.

#### Metodología Detallada

La metodología propuesta inicia con un proceso automático de diezmado para segmentar la señal ECG en función del tiempo y su posterior normalización. Durante la fase de entrenamiento, se generan bases de señal (diccionarios) mediante los algoritmos OMP y k-SVD, con el fin de aprender las características principales de cada conjunto de señales y facilitar el reconocimiento de similitudes con señales de prueba a través de su descomposición. 

El esquema convencional de clasificación de señales pre-SCD compara las características de una señal de entrada con las de señales normales y pre-SCD, asignándola a la clase SCD si no se parecen a las normales, lo que puede introducir sesgos. Para mejorar este proceso, se propone un enfoque multicategoría que incluye la clase normal y varios intervalos temporales previos al SCD, de modo que la señal se clasifique en la clase con mayor similitud.

**Referencia completa:** Velázquez-González, J. R., et al. "ECG-Based Identification of Sudden Cardiac Death through Sparse Representations," *Sensors*, vol. 21, no. 22, p. 7666, Nov. 2021.

**Archivo asociado:** `sensors-21-07666-v2 (2).pdf`

**Detalles Adicionales del Paper:**

Este trabajo aborda el problema de la Muerte Súbita Cardíaca (SCD) utilizando representaciones dispersas para la identificación temprana. Los autores proponen evitar el uso de ranking fijo de características al considerar un diccionario como un conjunto flexible de características donde cada representación dispersa puede verse como un proceso de extracción de características dinámico.

**Resultados Principales:**
- **Precisión bajo esquema común:** 95.3%
- **Precisión bajo esquema multi-clase propuesto:** 80.5%
- **Ventana de predicción:** 30 minutos antes del evento SCD
- **Bases de datos:** MIT/BIH-SCDH y MIT/BIH-NSR

**Metodología Técnica:**
- **Algoritmos utilizados:** Orthogonal Matching Pursuit (OMP) y k-Singular Value Decomposition (k-SVD)
- **Pre-procesamiento:** Filtrado paso banda, normalización, diezmado temporal
- **Clasificador:** Support Vector Machine (SVM)
- **Innovación:** Esquema multi-clase que distingue entre señales normales y diferentes intervalos temporales pre-SCD

**Limitaciones Identificadas:**
El esquema de clasificación binaria tradicional (Normal vs SCD) puede generar sesgos, ya que una señal de ECG de un paciente podría contener características que difieren de una señal ECG normal debido a enfermedades cardíacas previas, pero no necesariamente por un episodio futuro de SCD.

---

### C. Enfoque 2: Fusión Jerárquica de Características (Huang et al.)

Este trabajo aborda el problema desde un ángulo diferente, argumentando que la combinación de información a diferentes escalas puede mejorar la capacidad predictiva.

#### 1. Pre-procesamiento

De manera similar al primer enfoque, se realiza un filtrado digital para acondicionar la señal, eliminando artefactos y ruido para aislar la actividad eléctrica cardíaca de interés.

#### 2. Extracción de Características

La innovación aquí radica en una **fusión jerárquica**. Primero, utilizan la **Transformada Wavelet Continua (CWT)**, una técnica vista en detalle en clase, para descomponer la señal de ECG en diferentes escalas de tiempo y frecuencia. Esto genera un escalograma que revela patrones transitorios. A partir de esta representación, extraen múltiples tipos de características: características estadísticas, de textura y de entropía. Luego, estas características se fusionan de manera jerárquica para crear un descriptor final altamente informativo.

#### 3. Clasificación

El vector de características fusionadas se utiliza para entrenar un modelo clasificador, que en este caso también es una Máquina de Soporte Vectorial, para realizar la predicción del riesgo de MSC.

#### Metodología Detallada

El marco propuesto consta de cuatro etapas secuenciales: preprocesamiento de la señal, extracción de características, fusión de características y clasificación. Primero, las señales ECG crudas se someten a eliminación de ruido mediante transformada wavelet discreta (DWT). Posteriormente, la extracción de características se realiza en tres ramas:

1. **Lineal**: cinco características como intervalos RR, complejos QRS y ondas T
2. **No lineal**: exponente de escala α1 obtenido de las secuencias de intervalos RR mediante análisis de fluctuación detrendida de segundo orden (DFA-2)
3. **Profunda**: representaciones multiescala derivadas por un modelo TCN-Seq2vec

Estas características heterogéneas se fusionan de manera jerárquica para obtener una representación unificada y más discriminativa. Finalmente, una capa totalmente conectada clasifica las características fusionadas para producir las predicciones probabilísticas de las categorías objetivo.

**Referencia completa:** Huang, X., et al. "Improving Early Prediction of Sudden Cardiac Death Risk via Hierarchical Feature Fusion," *Symmetry*, vol. 17, no. 10, p. 1738, Oct. 2025.

**Archivo asociado:** `symmetry-17-01738 (2).pdf`

**Detalles Adicionales del Paper:**

Este trabajo propone un marco novedoso de fusión jerárquica de características para la predicción temprana del riesgo de SCD, diseñado para capturar patrones no estacionarios y asimétricos en datos de ECG a través de seis intervalos de tiempo distintos que preceden al inicio de la fibrilación ventricular (VF).

**Resultados Principales:**
- **Precisión promedio para 60 min pre-SCD:** 97.48%
- **Precisión promedio para 30 min pre-SCD:** 98.8%
- **Paradigma de evaluación:** Inter-paciente (clínicamente relevante)
- **Validación externa:** Creighton University Ventricular Tachyarrhythmia Database (CUDB)
- **Extensión del horizonte:** Hasta 60 minutos antes del evento

**Arquitectura de Características:**
1. **Características Lineales:** Extraídas mediante métodos de detección de formas de onda (intervalos RR, complejos QRS, ondas T)
2. **Características No Lineales:** Exponente de escala α1 obtenido de secuencias de intervalos RR mediante análisis de fluctuación detrendida de segundo orden (DFA-2)
3. **Características de Aprendizaje Profundo:** Representaciones multiescala capturadas usando un modelo TCN-Seq2vec (Temporal Convolutional Network)

**Contribuciones Principales:**
1. Arquitectura multi-nivel que integra información complementaria de representaciones lineales, no lineales y basadas en deep learning
2. Módulo de fusión jerárquica dedicado para combinar contextos temporales multiescala
3. Evaluación rigurosa bajo paradigma inter-paciente, demostrando mejoras significativas en precisión de predicción

**Desafíos Abordados:**
- Características asimétricas y no estacionarias de señales ECG
- Predicción a largo plazo (extendiendo el horizonte más allá de minutos)
- Integración de características diseñadas manualmente con representaciones de deep learning

---

## III. RESULTADOS Y DISCUSIÓN

Al evaluar el rendimiento de ambos métodos, es importante no solo mirar las cifras de precisión, sino también entender qué significan en el contexto clínico.

### Método de Representaciones Dispersas

El método de **representaciones dispersas** reporta una alta capacidad de clasificación, alcanzando una precisión, sensibilidad y especificidad superiores al 90%. Su principal fortaleza radica en la capacidad de aprender las morfologías fundamentales del ECG directamente de los datos, lo que potencialmente lo hace robusto a la variabilidad entre pacientes.

### Método de Fusión Jerárquica

Por su parte, el método de **fusión jerárquica** también alcanza un rendimiento sobresaliente, con métricas de precisión ligeramente superiores en algunas de las pruebas. La ventaja de este enfoque, conectado con los conceptos de la transformada wavelet, es su capacidad para capturar simultáneamente detalles finos del complejo QRS y características más amplias relacionadas con la repolarización (onda T) en una sola representación unificada.

### Análisis Comparativo

¿Cuál enfoque es mejor?

El método de representaciones dispersas ofrece un marco elegante y matemáticamente sólido, pero su rendimiento puede depender fuertemente de la calidad del "diccionario" aprendido. Por otro lado, el enfoque de fusión de características es muy potente, pero puede ser computacionalmente más intensivo y la selección de las características a fusionar puede parecer algo *ad-hoc*.

Una observación interesante es que ambos métodos se alejan de los marcadores tradicionales del ECG y se adentran en representaciones más abstractas de la señal. Esto sugiere que los patrones que predicen la MSC pueden no ser evidentes a simple vista, sino que están codificados en la textura y estructura de la señal a múltiples escalas. El éxito del método basado en wavelets, en particular, refuerza la idea de que el análisis tiempo-frecuencia es una herramienta excepcionalmente adecuada para señales biomédicas no estacionarias como el ECG.

---

## IV. CONCLUSIONES

Esta revisión ha analizado dos metodologías de vanguardia para la predicción de la muerte súbita cardíaca. Se ha demostrado que tanto las representaciones dispersas como la fusión jerárquica de características basadas en wavelets son enfoques viables y potentes, superando a muchos métodos tradicionales.

El análisis comparativo sugiere que las técnicas que integran información de múltiples escalas, como el método de Huang et al., podrían tener una ligera ventaja al capturar la compleja dinámica del sistema eléctrico cardíaco. Sin embargo, la simplicidad relativa y la solidez teórica del enfoque de representaciones dispersas de Velázquez-González et al. lo convierten en una alternativa muy atractiva.

### Trabajo Futuro

Como trabajo futuro, sería de gran interés explorar modelos híbridos que combinen ambas filosofías: por ejemplo, utilizando la transformada wavelet para generar los "átomos" de un diccionario para representaciones dispersas. Además, la validación de estos algoritmos en bases de datos más grandes y diversas sigue siendo un paso necesario antes de que puedan considerarse para su aplicación clínica. En última instancia, el camino hacia la predicción fiable de la MSC dependerá de la sinergia entre el procesamiento de señales avanzado y la validación clínica rigurosa.

---

## V. REFERENCIAS

[1] J. R. Velázquez-González, H. Peregrina-Barreto, J. J. Rangel-Magdaleno, J. M. Ramirez-Cortes, and J. P. Amezquita-Sanchez, "ECG-Based Identification of Sudden Cardiac Death through Sparse Representations," *Sensors*, vol. 21, no. 22, p. 7666, Nov. 2021.

[2] X. Huang, G. Jia, M. Huang, X. He, Y. Li, and M. Jiang, "Improving Early Prediction of Sudden Cardiac Death Risk via Hierarchical Feature Fusion," *Symmetry*, vol. 17, no. 10, p. 1738, Oct. 2025.

---

## ANEXOS

### Anexo A: Documentos Originales

Este documento consolida la información de tres documentos fuente:

1. **IEEE Borrador 1.docx** - Documento principal de revisión comparativa
2. **sensors-21-07666-v2 (2).pdf** - Paper completo de Velázquez-González et al. (Sensors, 2021)
   - 15 páginas
   - Método: Representaciones Dispersas (Sparse Representations)
   - Precisión: 95.3% (esquema común), 80.5% (esquema multi-clase)
   - Ventana de predicción: 30 minutos

3. **symmetry-17-01738 (2).pdf** - Paper completo de Huang et al. (Symmetry, 2025)
   - 24 páginas
   - Método: Fusión Jerárquica de Características
   - Precisión: 97.48% (60 min), 98.8% (30 min)
   - Ventana de predicción: Hasta 60 minutos

**Nota:** Para consultar los detalles completos de los métodos, métricas específicas, figuras y tablas detalladas, se recomienda revisar los documentos originales en formato PDF mencionados arriba.

### Anexo B: Resumen Comparativo de Métricas

| Aspecto | Velázquez-González et al. | Huang et al. |
|---------|---------------------------|--------------|
| **Método Principal** | Representaciones Dispersas | Fusión Jerárquica |
| **Precisión (30 min)** | 95.3% (esquema común) | 98.8% |
| **Precisión (60 min)** | No reportado | 97.48% |
| **Ventana Máxima** | 30 minutos | 60 minutos |
| **Algoritmos Clave** | OMP, k-SVD | TCN-Seq2vec, DFA-2 |
| **Características** | Dinámicas (diccionario) | Multi-nivel (lineal, no lineal, deep) |
| **Clasificador** | SVM | Capas Fully Connected |
| **Paradigma** | Binario y Multi-clase | Inter-paciente |
| **Validación Externa** | No reportada | CUDB |

### Anexo C: Contexto Clínico

**Relevancia Clínica:**

Ambos métodos abordan el problema crítico de la predicción temprana de la muerte súbita cardíaca, que:
- Representa más del 50% de las muertes por enfermedades cardiovasculares
- Ocurre generalmente dentro de la primera hora desde el inicio de los síntomas
- Tiene una tasa de supervivencia que disminuye aproximadamente 10% por minuto después del inicio de la fibrilación ventricular
- Requiere intervención temprana para mejorar significativamente las tasas de supervivencia

**Implicaciones Tecnológicas:**

El avance de los dispositivos *wearable* portátiles de ECG de una sola derivación ha revolucionado el campo, permitiendo:
- Monitoreo continuo y ambulatorio
- Detección temprana fuera de entornos clínicos tradicionales
- Análisis en tiempo real de señales ECG
- Aplicación de sistemas de alerta temprana para SCD

---

**Documento Consolidado**  
**Versión:** 1.0  
**Fecha de Consolidación:** Noviembre 2024  
**Compilado por:** Sistema de Documentación del Proyecto

---

