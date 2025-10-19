"""
Módulo principal para Predicción de Muerte Súbita ECG

Basado en los papers:
- Velázquez-González et al., Sensors 2021
- Huang et al., Symmetry 2025

Este módulo contiene utilidades para trabajar con datasets de ECG
de PhysioNet para predicción de muerte súbita.
"""

__version__ = "1.0.0"
__author__ = "Proyecto Final - Maestría IA"
__description__ = "Predicción de Muerte Súbita usando señales ECG"

# Importar funciones principales
from .utils import load_ecg_record, plot_ecg_signal, get_record_info
from .preprocessing import preprocess_ecg_signal, extract_features

__all__ = [
    'load_ecg_record',
    'plot_ecg_signal', 
    'get_record_info',
    'preprocess_ecg_signal',
    'extract_features'
]
