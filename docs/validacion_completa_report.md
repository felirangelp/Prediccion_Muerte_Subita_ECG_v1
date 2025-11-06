# Reporte de Validación Completa de Datasets

**Fecha de verificación**: 2025-11-06 07:28:45

## Resumen Ejecutivo

**Total de datasets verificados**: 3
**Datasets con integridad 100%**: 0/3

⚠️  **CONCLUSIÓN**: Algunos datasets tienen problemas de integridad.

---

## SDDB

### Tamaño
- Actual: 1.1253 GB
- Esperado: 5.00 GB
- Porcentaje: 22.51%
- Estado: ❌

### Archivos
- .dat: 23/23
- .hea: 23/23
- .atr: 12/23
- Estado: ✅

### Checksums SHA256
- SHA256SUMS.txt: No disponible

### Carga con wfdb
- wfdb: No disponible

### Estado General
- ⚠️  INTEGRIDAD PARCIAL

---

## NSRDB

### Tamaño
- Actual: 0.5367 GB
- Esperado: 2.00 GB
- Porcentaje: 26.84%
- Estado: ❌

### Archivos
- .dat: 18/18
- .hea: 17/18
- .atr: 17/18
- Estado: ❌

### Checksums SHA256
- SHA256SUMS.txt: No disponible

### Carga con wfdb
- wfdb: No disponible

### Estado General
- ⚠️  INTEGRIDAD PARCIAL

---

## CUDB

### Tamaño
- Actual: 0.0063 GB
- Esperado: 9.50 GB
- Porcentaje: 0.07%
- Estado: ❌

### Archivos
- .dat: 35/35
- .hea: 35/35
- .atr: 35/35
- Estado: ✅

### Checksums SHA256
- SHA256SUMS.txt: No disponible

### Carga con wfdb
- wfdb: No disponible

### Estado General
- ⚠️  INTEGRIDAD PARCIAL

---

## Recomendaciones

⚠️  Algunos datasets requieren atención:

- **SDDB**: tamaño
- **NSRDB**: tamaño, archivos
- **CUDB**: tamaño

**Recomendación**: Reiniciar descargas completas:
```bash
bash scripts/download_completo_verificado.sh
```
