# Guía de Descarga de Datasets ECG

Esta guía explica cómo descargar los datasets de ECG desde PhysioNet usando el script optimizado con aria2c.

## Resumen

El proyecto utiliza tres bases de datos de PhysioNet:
- **SDDB** (Sudden Cardiac Death Holter Database): ~5 GB, 23 registros
- **NSRDB** (Normal Sinus Rhythm Database): ~2 GB, 18 registros  
- **CUDB** (CU Ventricular Tachyarrhythmia Database): ~9.5 GB, 35 registros

**Total:** ~16.5 GB de datos

## Requisitos

- Python 3.8 o superior
- `aria2c` (se instala automáticamente si no está disponible)
- Conexión a internet estable
- ~20 GB de espacio libre en disco

## Sistema Robusto de Descarga Garantizada

Para garantizar que la descarga finalice al 100%, el proyecto incluye un sistema de tres capas:

### Opción 1: Descarga Simple (Recomendado para inicio rápido)

```bash
python scripts/descarga_maxima_velocidad.py
```

**Características:**
- Timeouts extendidos (1 hora por archivo)
- Espera de procesos aumentada (30 minutos)
- Reintentos automáticos (hasta 5 por archivo)
- Checkpointing automático
- Logging detallado

### Opción 2: Supervisor Automático (Recomendado para garantizar 100%)

```bash
python scripts/supervisor_descarga.py
```

**Características:**
- Monitorea el proceso cada 30 segundos
- Reinicia automáticamente si se detiene antes de completar
- Detecta progreso estancado y reinicia
- Continúa hasta completar al 100%
- Logging continuo del supervisor

### Opción 3: Descarga Persistente (Máxima garantía)

```bash
python scripts/descarga_persistente.py
```

**Características:**
- Ejecuta múltiples rondas de descarga (hasta 10 rondas)
- Verifica completitud después de cada ronda
- NO TERMINA hasta completar al 100%
- Espera inteligente entre rondas (5 minutos)
- Verificación robusta de tamaños y archivos

### ¿Cuál usar?

- **Primera vez**: Usa `descarga_persistente.py` para máxima garantía
- **Si el proceso se detiene**: Usa `supervisor_descarga.py` para monitoreo continuo
- **Descarga manual**: Usa `descarga_maxima_velocidad.py` directamente

## Uso Rápido

### 1. Descargar Datasets (Recomendado: Sistema Persistente)

```bash
# Activar ambiente virtual (si existe)
source venv/bin/activate

# Opción A: Descarga persistente (GARANTIZA 100% - RECOMENDADO)
python scripts/descarga_persistente.py

# Opción B: Con supervisor automático
python scripts/supervisor_descarga.py

# Opción C: Descarga simple (puede requerir reinicio manual)
python scripts/descarga_maxima_velocidad.py
```

**Recomendación:** Usa `descarga_persistente.py` para máxima garantía de completitud.

### 2. Monitorear Progreso

En otra terminal, ejecuta:

```bash
python scripts/monitor_aria2c.py
```

El monitor muestra:
- Progreso por dataset (archivos y tamaño)
- Velocidad de descarga (MB/s)
- Tiempo estimado restante (ETA)
- Barras de progreso visuales
- Desglose por tipo de archivo (.dat, .hea, .atr)

### 3. Verificar Integridad

Una vez completada la descarga:

```bash
python scripts/validacion_completa.py
```

Este script verifica:
- Tamaño de los datasets (debe ser ≥95% del esperado)
- Conteo de archivos (.dat, .hea, .atr)
- Checksums SHA256 (si están disponibles)
- Capacidad de carga con `wfdb` (verificación funcional)

## Cómo Funciona el Script

### `descarga_maxima_velocidad.py`

El script principal realiza las siguientes acciones:

1. **Verificación de Herramientas**
   - Comprueba si `aria2c` está disponible
   - Si no está, intenta instalarlo con `brew install aria2`
   - Si `aria2c` no está disponible, usa `wget` como respaldo

2. **Limpieza Inicial**
   - Detiene procesos de descarga anteriores
   - Elimina datasets incompletos
   - Limpia archivos temporales y logs antiguos

3. **Verificación de Espacio**
   - Comprueba que haya al menos 20 GB libres
   - Muestra el espacio disponible

4. **Descarga Paralela**
   - Usa `ThreadPoolExecutor` con 200 workers simultáneos
   - Descarga 225 archivos en total (69 .dat + 69 .hea + 87 .atr)
   - Cada archivo se descarga con `aria2c` usando:
     - `-x 16`: 16 conexiones simultáneas por archivo (máximo permitido)
     - `-s 16`: 16 fragmentos por archivo
     - `--timeout=60`: Timeout de 60 segundos
     - `--max-tries=5`: Hasta 5 reintentos
     - `--continue`: Permite reanudar descargas interrumpidas

5. **Verificación de Archivos**
   - Verifica que cada archivo descargado exista y tenga tamaño > 0
   - Muestra progreso cada 10 archivos completados

6. **Validación Final**
   - Ejecuta `validacion_completa.py` automáticamente al finalizar

### Optimizaciones Aplicadas

- **Paralelismo**: 200 procesos simultáneos
- **Conexiones múltiples**: 16 conexiones por archivo (aria2c)
- **Sin caché DNS**: Evita problemas de caché
- **Reintentos automáticos**: Hasta 5 intentos por archivo
- **Timeouts extendidos**: 60 segundos para archivos grandes
- **Verificación robusta**: Valida existencia y tamaño de cada archivo

## Problemas Encontrados y Soluciones

### Problema 1: Parámetros Inválidos de aria2c

**Error inicial:**
```
Exception: We encountered a problem while processing the option '--max-connection-per-server'.
max-connection-per-server must be between 1 and 16.
```

**Solución:**
- Eliminar el parámetro `--max-connection-per-server` (no es válido en esta versión)
- Usar solo `-x 16` que es equivalente y el máximo permitido
- El parámetro `-x` ya controla las conexiones máximas por servidor

### Problema 2: Parámetro --min-split-size Inválido

**Error:**
```
Exception: min-split-size must be between...
```

**Solución:**
- Eliminar el parámetro `--min-split-size=512K`
- Usar los valores por defecto de aria2c que son más compatibles

### Configuración Final

Los parámetros finales que funcionan correctamente:

```bash
aria2c -x 16 -s 16 \
  --timeout=60 \
  --max-tries=5 \
  --continue \
  --dir <directorio> \
  --out <archivo> \
  <URL>
```

- `-x 16`: 16 conexiones simultáneas (máximo permitido)
- `-s 16`: 16 fragmentos por archivo
- Sin `--max-connection-per-server` (no válido)
- Sin `--min-split-size` (no válido en esta versión)

## Estructura de Archivos Descargados

Los datasets se descargan en la siguiente estructura:

```
datasets/
├── sddb/
│   └── physionet.org/
│       └── files/
│           └── sddb/
│               └── 1.0.0/
│                   ├── 30.dat
│                   ├── 30.hea
│                   ├── 30.atr
│                   ├── 31.dat
│                   └── ...
├── nsrdb/
│   └── physionet.org/
│       └── files/
│           └── nsrdb/
│               └── 1.0.0/
│                   ├── 16265.dat
│                   ├── 16265.hea
│                   └── ...
└── cudb/
    └── physionet.org/
        └── files/
            └── cudb/
                └── 1.0.0/
                    ├── cu01.dat
                    ├── cu01.hea
                    └── ...
```

## Tiempo Estimado de Descarga

Dependiendo de tu conexión a internet:

- **Conexión rápida (>50 Mbps)**: 1-2 horas
- **Conexión media (10-50 Mbps)**: 2-4 horas
- **Conexión lenta (<10 Mbps)**: 4-8 horas

Con 200 procesos simultáneos y aria2c, la descarga aprovecha al máximo el ancho de banda disponible.

## Solución de Problemas

### La descarga se detiene

1. Verifica tu conexión a internet
2. Revisa el espacio en disco disponible
3. Ejecuta el script nuevamente (usa `--continue` para reanudar)

### Error de instalación de aria2c

Si `aria2c` no se instala automáticamente:

```bash
# macOS
brew install aria2

# Linux (Ubuntu/Debian)
sudo apt-get install aria2

# Linux (CentOS/RHEL)
sudo yum install aria2
```

### Verificación falla

Si la validación indica archivos faltantes:

1. Revisa los logs del script
2. Ejecuta `validacion_completa.py` para ver detalles
3. Re-descarga los archivos faltantes manualmente si es necesario

## Referencias

- [PhysioNet](https://physionet.org/) - Fuente de los datasets
- [SDDB Documentation](https://physionet.org/content/sddb/1.0.0/)
- [NSRDB Documentation](https://physionet.org/content/nsrdb/1.0.0/)
- [CUDB Documentation](https://physionet.org/content/cudb/1.0.0/)
- [aria2c Documentation](https://aria2.github.io/)

