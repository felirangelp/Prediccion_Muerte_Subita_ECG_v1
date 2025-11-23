"""
Modelo Híbrido: Combinación de Representaciones Dispersas y Fusión Jerárquica
Basado en recomendaciones futuras del documento consolidado

Combina:
- Transformada Wavelet para generar átomos del diccionario
- Representaciones dispersas sobre escalogramas wavelet
- Fusión dual de características (sparse + hierarchical)
- Ensemble de clasificadores
"""

import numpy as np
import pywt
from scipy.signal import find_peaks
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import pickle

from src.sparse_representations import (
    OrthogonalMatchingPursuit, KSVD, SparseRepresentationClassifier
)
from src.hierarchical_fusion import (
    extract_linear_features, extract_nonlinear_features,
    create_tcn_seq2vec, HierarchicalFusionClassifier
)

def wavelet_decomposition(ecg_signal: np.ndarray, wavelet: str = 'db4',
                         levels: int = 5) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Descomposición wavelet multinivel
    
    Args:
        ecg_signal: Señal ECG (1D)
        wavelet: Tipo de wavelet ('db4', 'haar', 'coif2', etc.)
        levels: Número de niveles de descomposición
    
    Returns:
        Tuple con (coeficientes, escalograma)
    """
    # Descomposición discreta
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=levels)
    
    # Crear escalograma (representación tiempo-frecuencia)
    # Reconstruir cada nivel por separado
    scales = []
    for i in range(levels + 1):
        detail_coeffs = [np.zeros_like(c) for c in coeffs]
        if i == 0:
            detail_coeffs[0] = coeffs[0]  # Aproximación
        else:
            detail_coeffs[i] = coeffs[i]  # Detalle
        
        reconstructed = pywt.waverec(detail_coeffs, wavelet)
        # Limpiar NaN e Inf inmediatamente después de la reconstrucción
        reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        # Asegurar misma longitud
        if len(reconstructed) > len(ecg_signal):
            reconstructed = reconstructed[:len(ecg_signal)]
        elif len(reconstructed) < len(ecg_signal):
            reconstructed = np.pad(reconstructed, (0, len(ecg_signal) - len(reconstructed)))
        # Limpiar nuevamente después del padding
        reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        scales.append(reconstructed)
    
    escalograma = np.array(scales).T  # (samples x levels)
    # Limpiar NaN e Inf del escalograma completo
    escalograma = np.nan_to_num(escalograma, nan=0.0, posinf=0.0, neginf=0.0)
    
    return coeffs, escalograma

def create_wavelet_dictionary(training_signals: List[np.ndarray],
                             wavelet: str = 'db4', levels: int = 5,
                             n_atoms_per_level: int = 20) -> np.ndarray:
    """
    Crear diccionario usando átomos wavelet
    
    Args:
        training_signals: Lista de señales de entrenamiento
        wavelet: Tipo de wavelet
        levels: Número de niveles
        n_atoms_per_level: Número de átomos por nivel
    
    Returns:
        Diccionario de átomos wavelet
    """
    all_atoms = []
    
    for signal in training_signals:
        # Limpiar señal de entrada antes de procesar
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        # Descomposición wavelet
        coeffs, escalograma = wavelet_decomposition(signal, wavelet, levels)
        # Asegurar que el escalograma no tenga NaN
        escalograma = np.nan_to_num(escalograma, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extraer átomos de diferentes niveles del escalograma
        for level in range(levels + 1):
            level_signal = escalograma[:, level]
            
            # Segmentar en ventanas para crear átomos
            window_size = len(level_signal) // n_atoms_per_level
            if window_size < 10:
                window_size = 10
            
            for start in range(0, len(level_signal) - window_size, 
                             max(1, window_size // 2)):
                atom = level_signal[start:start + window_size]
                # Normalizar
                # Limpiar NaN e Inf antes de normalizar
                atom = np.nan_to_num(atom, nan=0.0, posinf=0.0, neginf=0.0)
                norm = np.linalg.norm(atom)
                if norm > 1e-10:
                    atom = atom / norm
                else:
                    atom = np.zeros_like(atom)  # Si la norma es muy pequeña, usar vector cero
                
                # Interpolar a longitud estándar si es necesario
                target_length = 100
                if len(atom) != target_length:
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, len(atom))
                    x_new = np.linspace(0, 1, target_length)
                    # Manejar NaN e Inf antes de interpolar
                    atom = np.nan_to_num(atom, nan=0.0, posinf=0.0, neginf=0.0)
                    f = interp1d(x_old, atom, kind='linear', fill_value='extrapolate')
                    atom = f(x_new)
                    # Limpiar NaN después de interpolar
                    atom = np.nan_to_num(atom, nan=0.0, posinf=0.0, neginf=0.0)
                
                all_atoms.append(atom)
    
    # Seleccionar átomos únicos y representativos
    if len(all_atoms) > n_atoms_per_level * (levels + 1):
        # Usar k-means para seleccionar átomos representativos
        from sklearn.cluster import KMeans
        n_selected = n_atoms_per_level * (levels + 1)
        all_atoms_array = np.array(all_atoms)
        
        # Limpiar NaN e Inf antes de KMeans
        # Reemplazar NaN e Inf con 0
        all_atoms_array = np.nan_to_num(all_atoms_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verificar que no haya filas completamente cero o con problemas
        valid_rows = ~np.all(all_atoms_array == 0, axis=1)
        if np.sum(valid_rows) < n_selected:
            # Si no hay suficientes filas válidas, usar todas
            valid_rows = np.ones(len(all_atoms_array), dtype=bool)
        
        all_atoms_array = all_atoms_array[valid_rows]
        
        # Verificación final: asegurar que no haya NaN o Inf antes de KMeans
        if np.any(np.isnan(all_atoms_array)) or np.any(np.isinf(all_atoms_array)):
            # Si aún hay NaN/Inf, limpiar nuevamente
            all_atoms_array = np.nan_to_num(all_atoms_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verificar que el array sea finito antes de KMeans
        if not np.all(np.isfinite(all_atoms_array)):
            # Si hay valores no finitos, reemplazar con ceros
            all_atoms_array = np.where(np.isfinite(all_atoms_array), all_atoms_array, 0.0)
        
        if len(all_atoms_array) < n_selected:
            # Si no hay suficientes átomos, usar todos los disponibles
            selected_atoms = all_atoms_array
        else:
            kmeans = KMeans(n_clusters=n_selected, random_state=42, n_init=10)
            kmeans.fit(all_atoms_array)
            selected_atoms = kmeans.cluster_centers_
    else:
        selected_atoms = np.array(all_atoms[:n_atoms_per_level * (levels + 1)])
        # Limpiar NaN e Inf
        selected_atoms = np.nan_to_num(selected_atoms, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalizar cada átomo
    for i in range(len(selected_atoms)):
        atom = selected_atoms[i]
        # Limpiar NaN e Inf
        atom = np.nan_to_num(atom, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(atom)
        if norm > 1e-10:
            selected_atoms[i] = atom / norm
        else:
            selected_atoms[i] = np.zeros_like(atom)
    
    return selected_atoms.T  # (signal_length x n_atoms)

class HybridSCDClassifier:
    """
    Clasificador híbrido que combina ambos métodos
    """
    
    def __init__(self, n_atoms: int = 100, n_nonzero_coefs: int = 10,
                 wavelet: str = 'db4', wavelet_levels: int = 5,
                 tcn_filters: int = 64, fusion_dim: int = 128,
                 ensemble_method: str = 'voting'):
        """
        Args:
            n_atoms: Número de átomos en diccionario wavelet
            n_nonzero_coefs: Coeficientes no cero para OMP
            wavelet: Tipo de wavelet
            wavelet_levels: Niveles de descomposición
            tcn_filters: Filtros en TCN
            fusion_dim: Dimensión de fusión
            ensemble_method: Método de ensemble ('voting', 'stacking', 'weighted')
        """
        self.n_atoms = n_atoms
        self.n_nonzero_coefs = n_nonzero_coefs
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.tcn_filters = tcn_filters
        self.fusion_dim = fusion_dim
        self.ensemble_method = ensemble_method
        
        # Modelos componentes
        self.sparse_classifier = None
        self.hierarchical_classifier = None
        self.ensemble_classifier = None
        
        # Diccionarios wavelet
        self.wavelet_dictionaries = {}
        
        # Scaler para características fusionadas
        self.scaler_fusion = StandardScaler()
    
    def _prepare_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Preparar señal a longitud estándar"""
        if len(signal) >= target_length:
            return signal[:target_length]
        else:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, signal, kind='linear', fill_value='extrapolate')
            return f(x_new)
    
    def fit(self, X: List[np.ndarray], y: np.ndarray, fs: float = 128.0,
            epochs_hierarchical: int = 50, batch_size: int = 16):
        """
        Entrenar modelo híbrido
        
        Args:
            X: Lista de señales ECG
            y: Etiquetas de clase
            fs: Frecuencia de muestreo
            epochs_hierarchical: Épocas para entrenamiento jerárquico
            batch_size: Tamaño de batch
        """
        print(f"🔧 Entrenando modelo híbrido con {len(X)} señales...")
        
        # 1. Preparar señales
        target_length = int(60 * fs)  # 60 segundos
        X_prepared = [self._prepare_signal(s, target_length) for s in X]
        
        # 2. Entrenar componente de representaciones dispersas con diccionarios wavelet
        print(f"📚 Entrenando componente de representaciones dispersas (wavelet)...")
        
        classes = np.unique(y)
        
        for class_label in classes:
            class_signals = [X_prepared[i] for i in range(len(X_prepared)) if y[i] == class_label]
            
            if len(class_signals) == 0:
                continue
            
            # Crear diccionario wavelet para esta clase
            dictionary = create_wavelet_dictionary(
                class_signals,
                wavelet=self.wavelet,
                levels=self.wavelet_levels,
                n_atoms_per_level=self.n_atoms // (self.wavelet_levels + 1)
            )
            
            self.wavelet_dictionaries[class_label] = dictionary
            print(f"   ✅ Diccionario wavelet clase {class_label}: {dictionary.shape}")
        
        # 3. Entrenar clasificador disperso
        print(f"🤖 Entrenando clasificador de representaciones dispersas...")
        
        # Extraer características dispersas usando diccionarios wavelet
        sparse_features = []
        for signal in X_prepared:
            signal_features = []
            
            # Obtener representación dispersa para cada diccionario
            for class_label in classes:
                dictionary = self.wavelet_dictionaries[class_label]
                
                # Aplicar wavelet a la señal
                _, escalograma = wavelet_decomposition(signal, self.wavelet, self.wavelet_levels)
                
                # Usar escalograma promedio como entrada
                escalograma_avg = escalograma.mean(axis=1)
                
                # Limpiar NaN e Inf del escalograma antes de procesar
                escalograma_avg = np.nan_to_num(escalograma_avg, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Asegurar longitud compatible
                if len(escalograma_avg) > dictionary.shape[0]:
                    escalograma_avg = escalograma_avg[:dictionary.shape[0]]
                elif len(escalograma_avg) < dictionary.shape[0]:
                    # Interpolar
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, len(escalograma_avg))
                    x_new = np.linspace(0, 1, dictionary.shape[0])
                    f = interp1d(x_old, escalograma_avg, kind='linear', fill_value='extrapolate')
                    escalograma_avg = f(x_new)
                    # Limpiar nuevamente después de interpolar
                    escalograma_avg = np.nan_to_num(escalograma_avg, nan=0.0, posinf=0.0, neginf=0.0)
                
                # OMP sobre escalograma
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                coef = omp.fit(escalograma_avg, dictionary)
                signal_features.extend(coef)
            
            sparse_features.append(signal_features)
        
        sparse_features = np.array(sparse_features)
        
        # Limpiar NaN e Inf de las características dispersas
        sparse_features = np.nan_to_num(sparse_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Crear clasificador disperso simple
        from sklearn.svm import SVC
        self.sparse_classifier = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        sparse_features_scaled = self.scaler_fusion.fit_transform(sparse_features)
        # Verificar que no haya NaN después del escalado
        sparse_features_scaled = np.nan_to_num(sparse_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.sparse_classifier.fit(sparse_features_scaled, y)
        
        print(f"   ✅ Clasificador disperso entrenado")
        
        # 4. Entrenar componente de fusión jerárquica
        print(f"🔗 Entrenando componente de fusión jerárquica...")
        self.hierarchical_classifier = HierarchicalFusionClassifier(
            tcn_filters=self.tcn_filters,
            fusion_dim=self.fusion_dim,
            n_classes=len(classes)
        )
        
        self.hierarchical_classifier.fit(
            X, y, fs=fs,
            epochs=epochs_hierarchical,
            batch_size=batch_size
        )
        
        print(f"   ✅ Clasificador jerárquico entrenado")
        
        # 5. Crear ensemble
        print(f"🎯 Creando ensemble de clasificadores...")
        
        if self.ensemble_method == 'voting':
            # Voting classifier
            from sklearn.ensemble import VotingClassifier
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class SparseWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, classifier, scaler):
                    self.classifier = classifier
                    self.scaler = scaler
                
                def fit(self, X, y):
                    return self
                
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.classifier.predict(X_scaled)
                
                def predict_proba(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.classifier.predict_proba(X_scaled)
            
            # Wrapper para hierarchical (ya tiene predict_proba)
            class HierarchicalWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, classifier, X_train, fs):
                    self.classifier = classifier
                    self.X_train = X_train
                    self.fs = fs
                
                def fit(self, X, y):
                    return self
                
                def predict(self, X):
                    return self.classifier.predict(X, self.fs)
                
                def predict_proba(self, X):
                    return self.classifier.predict_proba(X, self.fs)
            
            # Para voting necesitamos entrenar con datos preparados
            # Usar probabilidades de ambos modelos
            sparse_proba = self.sparse_classifier.predict_proba(sparse_features_scaled)
            
            # Predecir con hierarchical para obtener probabilidades
            hierarchical_proba = self.hierarchical_classifier.predict_proba(X, fs=fs)
            
            # Combinar probabilidades (promedio ponderado)
            # Peso mayor para hierarchical (más robusto según papers)
            combined_proba = 0.4 * sparse_proba + 0.6 * hierarchical_proba
            
            # Entrenar meta-clasificador simple
            from sklearn.linear_model import LogisticRegression
            self.ensemble_classifier = LogisticRegression(random_state=42, max_iter=1000)
            
            # Limpiar NaN e Inf de las probabilidades antes de combinar
            sparse_proba = np.nan_to_num(sparse_proba, nan=0.0, posinf=0.0, neginf=0.0)
            hierarchical_proba = np.nan_to_num(hierarchical_proba, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Combinar características
            combined_features = np.hstack([sparse_proba, hierarchical_proba])
            
            # Limpiar NaN final antes de entrenar
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.ensemble_classifier.fit(combined_features, y)
            
        print(f"✅ Modelo híbrido entrenado completamente")
    
    def predict(self, X: List[np.ndarray], fs: float = 128.0) -> np.ndarray:
        """Predecir usando ensemble"""
        # Preparar señales
        target_length = int(60 * fs)
        X_prepared = [self._prepare_signal(s, target_length) for s in X]
        
        # Predicciones del componente disperso
        sparse_features = []
        classes = sorted(self.wavelet_dictionaries.keys())
        
        for signal in X_prepared:
            signal_features = []
            for class_label in classes:
                dictionary = self.wavelet_dictionaries[class_label]
                _, escalograma = wavelet_decomposition(signal, self.wavelet, self.wavelet_levels)
                escalograma_avg = escalograma.mean(axis=1)
                
                # Limpiar NaN e Inf del escalograma
                escalograma_avg = np.nan_to_num(escalograma_avg, nan=0.0, posinf=0.0, neginf=0.0)
                
                if len(escalograma_avg) > dictionary.shape[0]:
                    escalograma_avg = escalograma_avg[:dictionary.shape[0]]
                elif len(escalograma_avg) < dictionary.shape[0]:
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, len(escalograma_avg))
                    x_new = np.linspace(0, 1, dictionary.shape[0])
                    f = interp1d(x_old, escalograma_avg, kind='linear', fill_value='extrapolate')
                    escalograma_avg = f(x_new)
                    # Limpiar nuevamente después de interpolar
                    escalograma_avg = np.nan_to_num(escalograma_avg, nan=0.0, posinf=0.0, neginf=0.0)
                
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                coef = omp.fit(escalograma_avg, dictionary)
                signal_features.extend(coef)
            sparse_features.append(signal_features)
        
        sparse_features = np.array(sparse_features)
        sparse_features_scaled = self.scaler_fusion.transform(sparse_features)
        sparse_proba = self.sparse_classifier.predict_proba(sparse_features_scaled)
        
        # Predicciones del componente jerárquico
        hierarchical_proba = self.hierarchical_classifier.predict_proba(X, fs=fs)
        
        # Limpiar NaN e Inf antes de combinar
        sparse_proba = np.nan_to_num(sparse_proba, nan=0.0, posinf=0.0, neginf=0.0)
        hierarchical_proba = np.nan_to_num(hierarchical_proba, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combinar probabilidades
        combined_features = np.hstack([sparse_proba, hierarchical_proba])
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = self.ensemble_classifier.predict(combined_features)
        
        return predictions
    
    def predict_proba(self, X: List[np.ndarray], fs: float = 128.0) -> np.ndarray:
        """Obtener probabilidades"""
        target_length = int(60 * fs)
        X_prepared = [self._prepare_signal(s, target_length) for s in X]
        
        # Probabilidades dispersas
        sparse_features = []
        classes = sorted(self.wavelet_dictionaries.keys())
        
        for signal in X_prepared:
            signal_features = []
            for class_label in classes:
                dictionary = self.wavelet_dictionaries[class_label]
                _, escalograma = wavelet_decomposition(signal, self.wavelet, self.wavelet_levels)
                escalograma_avg = escalograma.mean(axis=1)
                
                # Limpiar NaN e Inf del escalograma
                escalograma_avg = np.nan_to_num(escalograma_avg, nan=0.0, posinf=0.0, neginf=0.0)
                
                if len(escalograma_avg) > dictionary.shape[0]:
                    escalograma_avg = escalograma_avg[:dictionary.shape[0]]
                elif len(escalograma_avg) < dictionary.shape[0]:
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, len(escalograma_avg))
                    x_new = np.linspace(0, 1, dictionary.shape[0])
                    f = interp1d(x_old, escalograma_avg, kind='linear', fill_value='extrapolate')
                    escalograma_avg = f(x_new)
                    # Limpiar nuevamente después de interpolar
                    escalograma_avg = np.nan_to_num(escalograma_avg, nan=0.0, posinf=0.0, neginf=0.0)
                
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                coef = omp.fit(escalograma_avg, dictionary)
                signal_features.extend(coef)
            sparse_features.append(signal_features)
        
        sparse_features = np.array(sparse_features)
        sparse_features_scaled = self.scaler_fusion.transform(sparse_features)
        sparse_proba = self.sparse_classifier.predict_proba(sparse_features_scaled)
        
        # Probabilidades jerárquicas
        hierarchical_proba = self.hierarchical_classifier.predict_proba(X, fs=fs)
        
        # Limpiar NaN e Inf antes de combinar
        sparse_proba = np.nan_to_num(sparse_proba, nan=0.0, posinf=0.0, neginf=0.0)
        hierarchical_proba = np.nan_to_num(hierarchical_proba, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combinar
        combined_features = np.hstack([sparse_proba, hierarchical_proba])
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        probabilities = self.ensemble_classifier.predict_proba(combined_features)
        
        return probabilities
    
    def save(self, filepath: str):
        """Guardar modelo"""
        # Guardar diccionarios wavelet
        with open(f"{filepath}_wavelet_dicts.pkl", 'wb') as f:
            pickle.dump(self.wavelet_dictionaries, f)
        
        # Guardar clasificador disperso
        with open(f"{filepath}_sparse.pkl", 'wb') as f:
            pickle.dump(self.sparse_classifier, f)
        
        # Guardar scaler
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler_fusion, f)
        
        # Guardar ensemble
        with open(f"{filepath}_ensemble.pkl", 'wb') as f:
            pickle.dump(self.ensemble_classifier, f)
        
        # Guardar clasificador jerárquico
        self.hierarchical_classifier.save(f"{filepath}_hierarchical")
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar modelo"""
        instance = cls()
        
        # Cargar diccionarios
        with open(f"{filepath}_wavelet_dicts.pkl", 'rb') as f:
            instance.wavelet_dictionaries = pickle.load(f)
        
        # Cargar clasificador disperso
        with open(f"{filepath}_sparse.pkl", 'rb') as f:
            instance.sparse_classifier = pickle.load(f)
        
        # Cargar scaler
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            instance.scaler_fusion = pickle.load(f)
        
        # Cargar ensemble
        with open(f"{filepath}_ensemble.pkl", 'rb') as f:
            instance.ensemble_classifier = pickle.load(f)
        
        # Cargar jerárquico
        instance.hierarchical_classifier = HierarchicalFusionClassifier.load(
            f"{filepath}_hierarchical"
        )
        
        return instance

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Modelo Híbrido: Wavelet + Sparse + Hierarchical")
    print("=" * 60)
    
    # Crear señales de ejemplo
    np.random.seed(42)
    fs = 128.0
    signal_length = int(60 * fs)
    
    normal_signals = [
        np.sin(2 * np.pi * 1.2 * np.linspace(0, 60, signal_length)) + 
        0.1 * np.random.randn(signal_length)
        for _ in range(20)
    ]
    
    prescd_signals = [
        np.sin(2 * np.pi * 1.5 * np.linspace(0, 60, signal_length)) + 
        0.2 * np.random.randn(signal_length)
        for _ in range(20)
    ]
    
    X = normal_signals + prescd_signals
    y = np.array([0] * 20 + [1] * 20)
    
    # Entrenar modelo híbrido
    classifier = HybridSCDClassifier(
        n_atoms=50,
        n_nonzero_coefs=5,
        wavelet='db4',
        tcn_filters=32,
        fusion_dim=64
    )
    
    classifier.fit(X, y, fs=fs, epochs_hierarchical=5, batch_size=8)
    
    # Predecir
    predictions = classifier.predict(X[:10], fs=fs)
    probabilities = classifier.predict_proba(X[:10], fs=fs)
    
    print(f"\n📊 Resultados:")
    print(f"   Predicciones (primeras 10): {predictions}")
    print(f"   Probabilidades clase 1: {probabilities[:, 1][:5]}")
    
    print(f"\n💡 Para usar en el proyecto:")
    print(f"   from src.hybrid_model import HybridSCDClassifier")
    print(f"   classifier = HybridSCDClassifier()")
    print(f"   classifier.fit(X_train, y_train, fs=128.0)")

