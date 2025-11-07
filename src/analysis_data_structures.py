"""
Estructuras de datos para análisis avanzados de predicción SCD
Incluye clases para almacenar resultados por intervalo temporal,
resultados multi-clase, y validación inter-paciente
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path


@dataclass
class TemporalIntervalResult:
    """Resultados de un modelo para un intervalo temporal específico"""
    interval_minutes: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    n_samples: int


@dataclass
class TemporalAnalysisResults:
    """Resultados completos de análisis temporal por intervalos"""
    intervals: List[int]  # [5, 10, 15, 20, 25, 30] minutos antes de SCD
    results_by_model: Dict[str, Dict[int, TemporalIntervalResult]] = field(default_factory=dict)
    # Estructura: {'sparse': {5: TemporalIntervalResult, 10: ..., ...}, ...}
    
    def add_result(self, model_name: str, interval: int, result: TemporalIntervalResult):
        """Añadir resultado para un modelo e intervalo"""
        if model_name not in self.results_by_model:
            self.results_by_model[model_name] = {}
        self.results_by_model[model_name][interval] = result
    
    def save(self, filepath: str):
        """Guardar resultados en archivo pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar resultados desde archivo pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


@dataclass
class MulticlassResult:
    """Resultados de clasificación multi-clase"""
    classes: List[str]  # ['Normal', '5min', '10min', ...]
    accuracy: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    n_samples_per_class: Dict[str, int]


@dataclass
class MulticlassAnalysisResults:
    """Resultados completos de análisis multi-clase"""
    binary_results: Dict[str, float] = field(default_factory=dict)  # {'sparse': 0.942, ...}
    multiclass_results: Dict[str, MulticlassResult] = field(default_factory=dict)
    # {'sparse': MulticlassResult, 'hierarchical': MulticlassResult, ...}
    
    def save(self, filepath: str):
        """Guardar resultados en archivo pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar resultados desde archivo pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


@dataclass
class InterPatientSplit:
    """División de datos para validación inter-paciente"""
    fold_id: int
    train_records: List[int]  # Números de registro para entrenamiento
    test_records: List[int]    # Números de registro para prueba
    n_train: int
    n_test: int


@dataclass
class InterPatientValidationResults:
    """Resultados de validación inter-paciente"""
    splits: List[InterPatientSplit]
    results_by_fold: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Estructura: {fold_id: {'sparse': {'accuracy': 0.87, ...}, ...}, ...}
    average_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Promedio de resultados por modelo
    
    def add_fold_result(self, fold_id: int, model_name: str, metrics: Dict[str, float]):
        """Añadir resultados de un fold y modelo"""
        if fold_id not in self.results_by_fold:
            self.results_by_fold[fold_id] = {}
        self.results_by_fold[fold_id][model_name] = metrics
    
    def calculate_averages(self):
        """Calcular promedios de resultados por modelo"""
        model_names = set()
        for fold_results in self.results_by_fold.values():
            model_names.update(fold_results.keys())
        
        metric_names = set()
        for fold_results in self.results_by_fold.values():
            for model_results in fold_results.values():
                metric_names.update(model_results.keys())
        
        for model_name in model_names:
            self.average_results[model_name] = {}
            for metric_name in metric_names:
                values = [
                    self.results_by_fold[fold_id][model_name][metric_name]
                    for fold_id in self.results_by_fold.keys()
                    if model_name in self.results_by_fold[fold_id]
                    and metric_name in self.results_by_fold[fold_id][model_name]
                ]
                if values:
                    self.average_results[model_name][metric_name] = np.mean(values)
    
    def save(self, filepath: str):
        """Guardar resultados en archivo pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar resultados desde archivo pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


@dataclass
class PaperComparisonData:
    """Datos de comparación con papers científicos"""
    paper_name: str
    year: int
    accuracy_by_interval: Dict[int, float]  # {5: 0.944, 10: 0.935, ...}
    prediction_horizon: int  # minutos antes de SCD
    methodology: Dict[str, str]  # {'preprocessing': ..., 'features': ..., 'classifier': ...}
    database: str


@dataclass
class PapersComparisonResults:
    """Resultados de comparación con papers"""
    our_results: Dict[str, Dict[int, float]] = field(default_factory=dict)
    # {'sparse': {5: 0.92, 10: 0.91, ...}, ...}
    papers_data: List[PaperComparisonData] = field(default_factory=list)
    methodology_comparison: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Comparación de metodologías
    
    def add_paper(self, paper_data: PaperComparisonData):
        """Añadir datos de un paper"""
        self.papers_data.append(paper_data)
    
    def save(self, filepath: str):
        """Guardar resultados en archivo pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar resultados desde archivo pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def check_data_availability(results_dir: str = "results") -> Dict[str, bool]:
    """
    Verificar qué datos están disponibles para el dashboard
    
    Returns:
        Dict con flags de disponibilidad de datos
    """
    results_path = Path(results_dir)
    availability = {
        'temporal_results': (results_path / 'temporal_results.pkl').exists(),
        'multiclass_results': (results_path / 'multiclass_results.pkl').exists(),
        'inter_patient_results': (results_path / 'inter_patient_results.pkl').exists(),
        'papers_comparison': (results_path / 'papers_comparison.pkl').exists(),
        'evaluation_results': (results_path / 'evaluation_results.pkl').exists(),
    }
    return availability

