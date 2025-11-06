## 🚀 Descarga Rápida de Datasets (PARALELA)

Para aprovechar mejor el ancho de banda, descarga los 3 datasets simultáneamente:

```bash
# Descarga PARALELA (más rápido con buen ancho de banda)
bash scripts/download_parallel.sh
```

**En otra terminal, puedes monitorear el progreso:**
```bash
bash scripts/monitor_parallel.sh
```

**Ventajas de descarga paralela:**
- ⚡ 2-3x más rápido que descarga secuencial
- 📊 Monitoreo en tiempo real de las 3 descargas
- 🔄 Aprovecha mejor el ancho de banda disponible

**Tiempo estimado con descarga paralela:**
- Con buen ancho de banda: 1-2 horas (vs 3-5 horas secuencial)

---

## 📋 Pasos Principales

# 1. Descargar datasets (PARALELO - recomendado)
bash scripts/download_parallel.sh

# 2. Entrenar modelos (requiere datos descargados - esperar a que termine la descarga)
# ¿Por qué necesita datos descargados?
# - Los modelos necesitan cargar archivos .dat (señales ECG), .hea (headers) y .atr (anotaciones)
# - Sin estos archivos, no puede extraer características ni entrenar los clasificadores
# - Verificar descarga completa primero: python scripts/validacion_completa.py
# 
python scripts/train_models.py --train-all --data-dir datasets/ --models-dir models/

# 3. Evaluar modelos entrenados
python scripts/evaluate_models.py --models-dir models/ --data-dir datasets/

# 4. Generar dashboard completo
python scripts/generate_dashboard.py --output dashboard_scd_prediction.html

# 5. Ejecutar pipeline completo
python scripts/run_complete_pipeline.py --data-dir datasets/ --models-dir models/