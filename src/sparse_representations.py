"""
Implementación del Método 1: Representaciones Dispersas para Predicción de Muerte Súbita Cardíaca
Basado en: Velázquez-González et al., Sensors 2021

Incluye:
- Algoritmo OMP (Orthogonal Matching Pursuit)
- k-SVD para aprendizaje de diccionarios
- Construcción de diccionarios por clase
- Extracción de características dispersas
- Clasificación SVM multi-clase
"""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csc_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Tuple, List, Dict, Optional
import warnings
import pickle
from pathlib import Path

class OrthogonalMatchingPursuit:
    """
    Implementación del algoritmo Orthogonal Matching Pursuit (OMP)
    para representaciones dispersas
    """
    
    def __init__(self, n_nonzero_coefs: Optional[int] = None, 
                 tolerance: float = 1e-6, max_iter: int = 100):
        """
        Args:
            n_nonzero_coefs: Número máximo de coeficientes no cero (None = automático)
            tolerance: Tolerancia para convergencia
            max_iter: Número máximo de iteraciones
        """
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tolerance = tolerance
        self.max_iter = max_iter
    
    def fit(self, signal: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
        """
        Encontrar representación dispersa de una señal usando el diccionario
        
        Args:
            signal: Señal a representar (1D array)
            dictionary: Diccionario de átomos (columnas son átomos)
        
        Returns:
            Coeficientes dispersos (vector)
        """
        signal = signal.flatten()
        n_atoms = dictionary.shape[1]
        
        # Inicializar
        residual = signal.copy()
        coef = np.zeros(n_atoms)
        support = []  # Índices de átomos seleccionados
        
        # Normalizar diccionario
        dict_norm = dictionary / np.linalg.norm(dictionary, axis=0, keepdims=True)
        
        # Determinar número máximo de coeficientes
        max_coefs = self.n_nonzero_coefs
        if max_coefs is None:
            max_coefs = min(n_atoms, len(signal) // 2)
        
        for iteration in range(min(max_coefs, self.max_iter)):
            # Calcular correlación entre residual y átomos
            correlations = np.abs(dict_norm.T @ residual)
            
            # Seleccionar átomo con mayor correlación (que no esté ya en support)
            correlations[support] = -np.inf
            atom_idx = np.argmax(correlations)
            
            # Agregar a support
            support.append(atom_idx)
            
            # Resolver problema de mínimos cuadrados sobre support
            selected_atoms = dict_norm[:, support]
            coef_support = solve(selected_atoms.T @ selected_atoms, 
                               selected_atoms.T @ signal, 
                               assume_a='pos')
            
            # Actualizar coeficientes
            coef[support] = coef_support
            
            # Actualizar residual
            residual = signal - dict_norm @ coef
            
            # Verificar convergencia
            if np.linalg.norm(residual) < self.tolerance:
                break
        
        return coef

class KSVD:
    """
    Implementación del algoritmo k-SVD para aprendizaje de diccionarios
    """
    
    def __init__(self, n_atoms: int = 100, n_nonzero_coefs: int = 10,
                 n_iterations: int = 50, tolerance: float = 1e-6):
        """
        Args:
            n_atoms: Número de átomos en el diccionario
            n_nonzero_coefs: Número de coeficientes no cero por señal
            n_iterations: Número de iteraciones de entrenamiento
            tolerance: Tolerancia para convergencia
        """
        self.n_atoms = n_atoms
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.dictionary = None
        self.omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    
    def fit(self, training_signals: np.ndarray, 
            init_dictionary: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aprender diccionario a partir de señales de entrenamiento
        
        Args:
            training_signals: Array de señales (n_signals x signal_length)
            init_dictionary: Diccionario inicial (opcional)
        
        Returns:
            Diccionario aprendido
        """
        n_signals, signal_length = training_signals.shape
        
        # Limpiar NaN e Inf de las señales de entrenamiento antes de procesar
        training_signals = np.nan_to_num(training_signals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Inicializar diccionario
        if init_dictionary is not None:
            self.dictionary = init_dictionary.copy()
            # Limpiar diccionario inicial
            self.dictionary = np.nan_to_num(self.dictionary, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Inicialización aleatoria con normalización
            self.dictionary = np.random.randn(signal_length, self.n_atoms)
            self.dictionary = self.dictionary / np.linalg.norm(self.dictionary, axis=0, keepdims=True)
        
        # Limpiar diccionario después de inicialización
        self.dictionary = np.nan_to_num(self.dictionary, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizar señales
        norms = np.linalg.norm(training_signals, axis=1, keepdims=True) + 1e-10
        training_signals_norm = training_signals / norms
        
        # Limpiar NaN e Inf después de normalizar
        training_signals_norm = np.nan_to_num(training_signals_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        prev_error = np.inf
        
        for iteration in range(self.n_iterations):
            # Mostrar progreso cada 5 iteraciones
            if iteration % 5 == 0 or iteration == 0:
                print(f"   🔄 Iteración k-SVD: {iteration + 1}/{self.n_iterations}")
            
            # Paso 1: Sparse coding - encontrar coeficientes para todas las señales
            coefficients = np.zeros((n_signals, self.n_atoms))
            
            # Procesar en lotes para mostrar progreso si hay muchas señales
            batch_size = max(1000, n_signals // 10) if n_signals > 10000 else n_signals
            for batch_start in range(0, n_signals, batch_size):
                batch_end = min(batch_start + batch_size, n_signals)
                for i in range(batch_start, batch_end):
                    coef = self.omp.fit(training_signals_norm[i], self.dictionary)
                    coefficients[i] = coef
            
            # Paso 2: Dictionary update - actualizar cada átomo usando k-SVD
            for atom_idx in range(self.n_atoms):
                # Encontrar señales que usan este átomo
                atom_usage = coefficients[:, atom_idx] != 0
                
                if not np.any(atom_usage):
                    # Si el átomo no se usa, reemplazarlo con señal de error residual
                    errors = training_signals_norm - self.dictionary @ coefficients.T
                    error_norms = np.linalg.norm(errors, axis=1)
                    max_error_idx = np.argmax(error_norms)
                    self.dictionary[:, atom_idx] = errors[max_error_idx]
                    self.dictionary[:, atom_idx] /= (np.linalg.norm(self.dictionary[:, atom_idx]) + 1e-10)
                    continue
                
                # Calcular error residual sin este átomo
                # training_signals_norm[atom_usage] tiene shape (n_signals, signal_length)
                # Necesitamos calcular: signals - D @ coef.T + D[:, atom_idx] @ coef[:, atom_idx]
                signals_using_atom = training_signals_norm[atom_usage]  # (n_signals, signal_length)
                coefs_using_atom = coefficients[atom_usage]  # (n_signals, n_atoms)
                
                # Reconstrucción completa sin el átomo actual
                # D @ coef.T da (signal_length, n_signals), necesitamos (n_signals, signal_length)
                reconstruction = (self.dictionary @ coefs_using_atom.T).T  # (n_signals, signal_length)
                
                # Agregar la contribución del átomo actual que queremos actualizar
                # np.outer da (signal_length, n_signals), necesitamos (n_signals, signal_length)
                atom_contribution = np.outer(coefs_using_atom[:, atom_idx], self.dictionary[:, atom_idx])  # (n_signals, signal_length)
                
                # Error residual: señal - reconstrucción + contribución del átomo
                residual = signals_using_atom - reconstruction + atom_contribution
                
                # Limpiar NaN e Inf del residual antes de SVD
                residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
                
                # SVD del residual
                try:
                    U, S, Vt = np.linalg.svd(residual.T, full_matrices=False)
                    
                    # Actualizar átomo y coeficientes
                    self.dictionary[:, atom_idx] = U[:, 0]
                    coefficients[atom_usage, atom_idx] = S[0] * Vt[0, :]
                    
                    # Limpiar NaN e Inf después de actualizar
                    self.dictionary[:, atom_idx] = np.nan_to_num(self.dictionary[:, atom_idx], nan=0.0, posinf=0.0, neginf=0.0)
                    coefficients[atom_usage, atom_idx] = np.nan_to_num(coefficients[atom_usage, atom_idx], nan=0.0, posinf=0.0, neginf=0.0)
                except:
                    # Si SVD falla, mantener el átomo actual pero limpiarlo
                    self.dictionary[:, atom_idx] = np.nan_to_num(self.dictionary[:, atom_idx], nan=0.0, posinf=0.0, neginf=0.0)
                    pass
            
            # Normalizar diccionario
            self.dictionary = self.dictionary / (
                np.linalg.norm(self.dictionary, axis=0, keepdims=True) + 1e-10
            )
            
            # Limpiar NaN e Inf después de normalizar
            self.dictionary = np.nan_to_num(self.dictionary, nan=0.0, posinf=0.0, neginf=0.0)
            coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Verificar convergencia
            error = np.mean([
                np.linalg.norm(signal - self.dictionary @ coef) ** 2
                for signal, coef in zip(training_signals_norm, coefficients)
            ])
            
            # Limpiar NaN e Inf del error
            if np.isnan(error) or np.isinf(error):
                error = prev_error
            
            # Mostrar progreso cada 5 iteraciones
            if iteration % 5 == 0 or iteration == 0:
                print(f"      Error: {error:.6f} (diferencia: {abs(prev_error - error):.6f})")
            
            if abs(prev_error - error) < self.tolerance:
                print(f"   ✅ Convergencia alcanzada en iteración {iteration + 1}")
                break
            
            prev_error = error
        
        # Limpiar diccionario final antes de retornar
        self.dictionary = np.nan_to_num(self.dictionary, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self.dictionary
    
    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Obtener representaciones dispersas de señales usando el diccionario
        
        Args:
            signals: Array de señales (n_signals x signal_length)
        
        Returns:
            Coeficientes dispersos (n_signals x n_atoms)
        """
        if self.dictionary is None:
            raise ValueError("Diccionario no entrenado. Llamar fit() primero.")
        
        coefficients = []
        signals_norm = signals / (
            np.linalg.norm(signals, axis=1, keepdims=True) + 1e-10
        )
        
        for signal in signals_norm:
            coef = self.omp.fit(signal, self.dictionary)
            coefficients.append(coef)
        
        return np.array(coefficients)

class SparseRepresentationClassifier:
    """
    Clasificador completo usando representaciones dispersas
    """
    
    def __init__(self, n_atoms: int = 100, n_nonzero_coefs: int = 10,
                 svm_kernel: str = 'rbf', svm_c: float = 1.0,
                 multi_class: bool = True):
        """
        Args:
            n_atoms: Número de átomos por diccionario
            n_nonzero_coefs: Número de coeficientes no cero
            svm_kernel: Kernel para SVM ('rbf', 'linear', 'poly')
            svm_c: Parámetro C para SVM
            multi_class: Si usar esquema multi-clase o binario
        """
        self.n_atoms = n_atoms
        self.n_nonzero_coefs = n_nonzero_coefs
        self.svm_kernel = svm_kernel
        self.svm_c = svm_c
        self.multi_class = multi_class
        
        self.dictionaries = {}  # Diccionarios por clase
        self.ksvd_models = {}  # Modelos k-SVD por clase
        self.scaler = StandardScaler()
        self.svm_classifier = None
        self.classes_ = None
    
    def _prepare_training_signals(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        Preparar señales para entrenamiento (normalizar longitud)
        
        Args:
            signals: Lista de señales (pueden tener diferentes longitudes)
        
        Returns:
            Array de señales normalizadas
        """
        # Encontrar longitud mínima
        min_length = min(len(s) for s in signals)
        
        # Truncar o interpolar a longitud común
        prepared = []
        for signal in signals:
            # Limpiar NaN e Inf antes de procesar
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(signal) > min_length:
                # Truncar
                prepared.append(signal[:min_length])
            elif len(signal) < min_length:
                # Interpolar
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(signal))
                x_new = np.linspace(0, 1, min_length)
                f = interp1d(x_old, signal, kind='linear', fill_value='extrapolate')
                interpolated = f(x_new)
                # Limpiar NaN e Inf después de interpolar
                interpolated = np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0)
                prepared.append(interpolated)
            else:
                prepared.append(signal)
        
        result = np.array(prepared)
        # Limpiar NaN e Inf del resultado final
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        return result
    
    def fit(self, X: List[np.ndarray], y: np.ndarray,
            time_intervals: Optional[List[float]] = None):
        """
        Entrenar clasificador
        
        Args:
            X: Lista de señales ECG
            y: Etiquetas de clase
            time_intervals: Intervalos temporales para multi-clase (minutos antes de SCD)
        """
        self.classes_ = np.unique(y)
        
        # Preparar señales
        print(f"🔧 Preparando {len(X)} señales para entrenamiento...")
        X_prepared = [self._prepare_training_signals([s])[0] for s in X]
        signal_length = len(X_prepared[0])
        
        # Entrenar diccionarios por clase
        print(f"📚 Entrenando diccionarios para {len(self.classes_)} clases...")
        
        for class_label in self.classes_:
            class_signals = [X_prepared[i] for i in range(len(X_prepared)) if y[i] == class_label]
            
            if len(class_signals) == 0:
                continue
            
            # Convertir a array
            class_signals_array = np.array(class_signals)
            
            # Entrenar k-SVD con menos iteraciones para velocidad
            ksvd = KSVD(n_atoms=self.n_atoms, 
                       n_nonzero_coefs=self.n_nonzero_coefs,
                       n_iterations=20)  # Reducido de 50 a 20 para velocidad
            
            print(f"   🔄 Entrenando diccionario para clase {class_label}...")
            print(f"      Señales: {len(class_signals)}")
            print(f"      Átomos: {self.n_atoms}, Coeficientes no cero: {self.n_nonzero_coefs}")
            dictionary = ksvd.fit(class_signals_array)
            
            self.dictionaries[class_label] = dictionary
            self.ksvd_models[class_label] = ksvd
            
            print(f"   ✅ Diccionario clase {class_label}: {dictionary.shape}")
        
        # Extraer características dispersas
        print(f"🔍 Extrayendo características dispersas...")
        features = []
        
        for signal in X_prepared:
            signal_features = []
            
            # Obtener representación dispersa para cada diccionario
            for class_label in self.classes_:
                dictionary = self.dictionaries[class_label]
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                coef = omp.fit(signal, dictionary)
                signal_features.extend(coef)
            
            features.append(signal_features)
        
        features = np.array(features)
        
        # Normalizar características
        features_scaled = self.scaler.fit_transform(features)
        
        # Entrenar SVM
        print(f"🤖 Entrenando clasificador SVM...")
        self.svm_classifier = SVC(kernel=self.svm_kernel, C=self.svm_c, 
                                  probability=True, random_state=42)
        self.svm_classifier.fit(features_scaled, y)
        
        print(f"✅ Clasificador entrenado")
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predecir clases para nuevas señales
        
        Args:
            X: Lista de señales ECG
        
        Returns:
            Predicciones de clase
        """
        # Preparar señales
        X_prepared = [self._prepare_training_signals([s])[0] for s in X]
        
        # Extraer características
        features = []
        
        for signal in X_prepared:
            signal_features = []
            
            for class_label in self.classes_:
                dictionary = self.dictionaries[class_label]
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                coef = omp.fit(signal, dictionary)
                signal_features.extend(coef)
            
            features.append(signal_features)
        
        features = np.array(features)
        features_scaled = self.scaler.transform(features)
        
        # Predecir
        predictions = self.svm_classifier.predict(features_scaled)
        
        return predictions
    
    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Obtener probabilidades de clase
        
        Args:
            X: Lista de señales ECG
        
        Returns:
            Probabilidades por clase
        """
        X_prepared = [self._prepare_training_signals([s])[0] for s in X]
        
        features = []
        for signal in X_prepared:
            signal_features = []
            for class_label in self.classes_:
                dictionary = self.dictionaries[class_label]
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                coef = omp.fit(signal, dictionary)
                signal_features.extend(coef)
            features.append(signal_features)
        
        features = np.array(features)
        features_scaled = self.scaler.transform(features)
        
        return self.svm_classifier.predict_proba(features_scaled)
    
    def save(self, filepath: str):
        """Guardar modelo entrenado"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar modelo entrenado"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Método 1: Representaciones Dispersas")
    print("=" * 50)
    
    # Crear señales de ejemplo
    np.random.seed(42)
    n_signals = 50
    signal_length = 7680  # 60 segundos a 128 Hz
    
    # Señales clase 0 (Normal)
    normal_signals = [
        np.sin(2 * np.pi * 1.2 * np.linspace(0, 60, signal_length)) + 
        0.1 * np.random.randn(signal_length)
        for _ in range(n_signals // 2)
    ]
    
    # Señales clase 1 (Pre-SCD)
    prescd_signals = [
        np.sin(2 * np.pi * 1.5 * np.linspace(0, 60, signal_length)) + 
        0.2 * np.random.randn(signal_length)
        for _ in range(n_signals // 2)
    ]
    
    X = normal_signals + prescd_signals
    y = np.array([0] * (n_signals // 2) + [1] * (n_signals // 2))
    
    # Entrenar clasificador
    classifier = SparseRepresentationClassifier(
        n_atoms=50,
        n_nonzero_coefs=5,
        svm_kernel='rbf',
        multi_class=False
    )
    
    classifier.fit(X, y)
    
    # Predecir
    predictions = classifier.predict(X[:10])
    probabilities = classifier.predict_proba(X[:10])
    
    print(f"\n📊 Resultados:")
    print(f"   Predicciones (primeras 10): {predictions}")
    print(f"   Probabilidades clase 1: {probabilities[:, 1][:5]}")
    
    print(f"\n💡 Para usar en el proyecto:")
    print(f"   from src.sparse_representations import SparseRepresentationClassifier")
    print(f"   classifier = SparseRepresentationClassifier()")
    print(f"   classifier.fit(X_train, y_train)")

