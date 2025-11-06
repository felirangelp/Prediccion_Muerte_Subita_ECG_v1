"""
Implementación del Método 2: Fusión Jerárquica de Características para Predicción de Muerte Súbita Cardíaca
Basado en: Huang et al., Symmetry 2025

Incluye:
- Extracción de características lineales (RR, QRS, T)
- Extracción de características no lineales (DFA-2, entropía)
- Modelo TCN-Seq2vec para características de deep learning
- Fusión jerárquica de características
- Clasificación Fully Connected
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import scipy.signal as signal
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import warnings
import pickle
from pathlib import Path

# Configurar TensorFlow para usar GPU M1 si está disponible
try:
    from src.config_m1 import configure_tensorflow_m1
    tf_config = configure_tensorflow_m1()
    if tf_config['gpu_available']:
        print(f"✅ GPU Metal detectada: {tf_config['gpu_device']}")
except:
    pass

def detect_r_peaks_advanced(ecg_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Detección avanzada de picos R usando Pan-Tompkins mejorado
    
    Args:
        ecg_signal: Señal ECG (1D)
        fs: Frecuencia de muestreo
    
    Returns:
        Índices de picos R
    """
    # Derivada para enfatizar picos R
    derivative = np.diff(ecg_signal)
    
    # Cuadrado
    squared = derivative ** 2
    
    # Integración con ventana móvil
    window_size = int(0.15 * fs)
    if window_size % 2 == 0:
        window_size += 1
    
    integrated = signal.savgol_filter(squared, window_size, 3)
    
    # Detectar picos
    peaks, _ = signal.find_peaks(
        integrated,
        height=np.max(integrated) * 0.5,
        distance=int(0.2 * fs)  # Mínimo 200ms entre picos
    )
    
    return peaks + 1

def extract_linear_features(ecg_signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extraer características lineales: intervalos RR, complejos QRS, ondas T
    
    Args:
        ecg_signal: Señal ECG (1D)
        fs: Frecuencia de muestreo
    
    Returns:
        Diccionario con características lineales
    """
    features = {}
    
    # Detectar picos R
    r_peaks = detect_r_peaks_advanced(ecg_signal, fs)
    
    if len(r_peaks) < 2:
        # Si no se detectan suficientes picos, retornar características por defecto
        return {
            'mean_rr': 0.0,
            'std_rr': 0.0,
            'qrs_width': 0.0,
            't_amplitude': 0.0,
            't_width': 0.0
        }
    
    # 1. Características de intervalos RR (en ms)
    rr_intervals = np.diff(r_peaks) / fs * 1000
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    
    if len(rr_intervals) > 0:
        features['mean_rr'] = np.mean(rr_intervals)
        features['std_rr'] = np.std(rr_intervals)
    else:
        features['mean_rr'] = 0.0
        features['std_rr'] = 0.0
    
    # 2. Características del complejo QRS
    # Ancho promedio del QRS (aprox. 80-120ms)
    qrs_width_samples = int(0.1 * fs)  # 100ms
    qrs_widths = []
    
    for r_peak in r_peaks[:min(10, len(r_peaks))]:  # Analizar primeros 10
        start = max(0, r_peak - qrs_width_samples // 2)
        end = min(len(ecg_signal), r_peak + qrs_width_samples // 2)
        qrs_segment = ecg_signal[start:end]
        
        if len(qrs_segment) > 0:
            # Ancho a mitad de altura
            peak_val = ecg_signal[r_peak]
            half_height = peak_val / 2
            above_threshold = np.abs(qrs_segment) > np.abs(half_height)
            if np.any(above_threshold):
                width = np.sum(above_threshold) / fs * 1000  # en ms
                qrs_widths.append(width)
    
    features['qrs_width'] = np.mean(qrs_widths) if qrs_widths else 0.0
    
    # 3. Características de onda T
    # Buscar ondas T después de cada QRS (típicamente 200-400ms después)
    t_amplitudes = []
    t_widths = []
    
    for r_peak in r_peaks[:min(10, len(r_peaks))]:
        t_start = r_peak + int(0.2 * fs)  # 200ms después del R
        t_end = min(len(ecg_signal), r_peak + int(0.6 * fs))  # hasta 600ms
        
        if t_end > t_start:
            t_segment = ecg_signal[t_start:t_end]
            
            if len(t_segment) > 0:
                # Amplitud de la onda T
                t_amplitude = np.max(np.abs(t_segment))
                t_amplitudes.append(t_amplitude)
                
                # Ancho aproximado
                threshold = t_amplitude * 0.5
                above_threshold = np.abs(t_segment) > threshold
                if np.any(above_threshold):
                    width = np.sum(above_threshold) / fs * 1000
                    t_widths.append(width)
    
    features['t_amplitude'] = np.mean(t_amplitudes) if t_amplitudes else 0.0
    features['t_width'] = np.mean(t_widths) if t_widths else 0.0
    
    return features

def detrended_fluctuation_analysis(rr_intervals: np.ndarray, order: int = 2) -> float:
    """
    Análisis de fluctuación detrended de segundo orden (DFA-2)
    
    Args:
        rr_intervals: Secuencia de intervalos RR
        order: Orden del polinomio para detrending (2 para DFA-2)
    
    Returns:
        Exponente de escala α1
    """
    if len(rr_intervals) < 10:
        return 0.5
    
    # Integrar la serie
    y = np.cumsum(rr_intervals - np.mean(rr_intervals))
    
    # Rango de tamaños de ventana
    scales = np.logspace(np.log10(4), np.log10(len(y) // 4), 10).astype(int)
    scales = scales[scales >= order + 1]
    
    fluctuations = []
    
    for scale in scales:
        # Dividir en ventanas
        n_windows = len(y) // scale
        
        if n_windows < 2:
            continue
        
        fluctuation_scale = []
        
        for i in range(n_windows):
            window = y[i * scale:(i + 1) * scale]
            
            # Detrending polinomial
            x = np.arange(len(window))
            coeffs = np.polyfit(x, window, order)
            trend = np.polyval(coeffs, x)
            detrended = window - trend
            
            # Fluctuación
            fluctuation_scale.append(np.sqrt(np.mean(detrended ** 2)))
        
        if fluctuation_scale:
            fluctuations.append(np.mean(fluctuation_scale))
    
    if len(fluctuations) < 2:
        return 0.5
    
    # Calcular exponente de escala
    log_scales = np.log10(scales[:len(fluctuations)])
    log_fluctuations = np.log10(fluctuations)
    
    # Ajuste lineal
    if len(log_scales) > 1:
        alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]
        return alpha
    else:
        return 0.5

def extract_nonlinear_features(ecg_signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extraer características no lineales: DFA-2, entropías
    
    Args:
        ecg_signal: Señal ECG (1D)
        fs: Frecuencia de muestreo
    
    Returns:
        Diccionario con características no lineales
    """
    features = {}
    
    # Detectar picos R
    r_peaks = detect_r_peaks_advanced(ecg_signal, fs)
    
    if len(r_peaks) < 10:
        return {
            'dfa_alpha1': 0.5,
            'sample_entropy': 0.0,
            'approximate_entropy': 0.0
        }
    
    # Calcular intervalos RR
    rr_intervals = np.diff(r_peaks) / fs * 1000
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    
    if len(rr_intervals) < 10:
        return {
            'dfa_alpha1': 0.5,
            'sample_entropy': 0.0,
            'approximate_entropy': 0.0
        }
    
    # 1. DFA-2
    features['dfa_alpha1'] = detrended_fluctuation_analysis(rr_intervals, order=2)
    
    # 2. Sample Entropy (simplificado)
    try:
        m = 2  # Longitud de patrón
        r = 0.2 * np.std(rr_intervals)  # Tolerancia
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([rr_intervals[i:i + m] for i in range(len(rr_intervals) - m + 1)])
            C = np.zeros(len(patterns))
            
            for i in range(len(patterns)):
                template = patterns[i]
                matches = sum([1 for p in patterns if _maxdist(template, p, m) <= r])
                if matches > 0:
                    C[i] = matches / len(patterns)
            
            phi = np.mean([np.log(c) for c in C if c > 0]) if np.any(C > 0) else 0
            return phi
        
        features['sample_entropy'] = _phi(m) - _phi(m + 1) if len(rr_intervals) > m + 1 else 0.0
    except:
        features['sample_entropy'] = 0.0
    
    # 3. Approximate Entropy (simplificado)
    try:
        m = 2
        r = 0.2 * np.std(rr_intervals)
        
        def _phi_ap(m):
            patterns = np.array([rr_intervals[i:i + m] for i in range(len(rr_intervals) - m + 1)])
            C = np.zeros(len(patterns))
            
            for i in range(len(patterns)):
                template = patterns[i]
                matches = sum([1 for p in patterns if _maxdist(template, p, m) <= r])
                C[i] = matches / len(patterns)
            
            phi = np.mean([np.log(c) for c in C if c > 0]) if np.any(C > 0) else 0
            return phi
        
        features['approximate_entropy'] = _phi_ap(m) - _phi_ap(m + 1) if len(rr_intervals) > m + 1 else 0.0
    except:
        features['approximate_entropy'] = 0.0
    
    return features

def create_tcn_seq2vec(input_length: int, n_filters: int = 64, 
                       kernel_size: int = 3, n_blocks: int = 3,
                       dropout: float = 0.2) -> Model:
    """
    Crear modelo TCN-Seq2vec (Temporal Convolutional Network)
    
    Args:
        input_length: Longitud de la secuencia de entrada
        n_filters: Número de filtros
        kernel_size: Tamaño del kernel
        n_blocks: Número de bloques TCN
        dropout: Tasa de dropout
    
    Returns:
        Modelo Keras
    """
    inputs = layers.Input(shape=(input_length, 1))
    
    x = inputs
    
    # Bloques TCN
    for i in range(n_blocks):
        # Dilated causal convolution
        dilation_rate = 2 ** i
        
        # Causal padding
        padding = (kernel_size - 1) * dilation_rate
        
        # Conv1D con dilation
        conv = layers.Conv1D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        
        # Batch normalization
        conv = layers.BatchNormalization()(conv)
        
        # Dropout
        conv = layers.Dropout(dropout)(conv)
        
        # Residual connection
        if i == 0:
            x = conv
        else:
            # Proyectar si las dimensiones no coinciden
            if x.shape[-1] != conv.shape[-1]:
                x = layers.Conv1D(n_filters, 1)(x)
            x = layers.Add()([x, conv])
    
    # Global pooling (Seq2Vec)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Capa de salida
    outputs = layers.Dense(n_filters, activation='relu')(x)
    outputs = layers.Dropout(dropout)(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

class HierarchicalFusionClassifier:
    """
    Clasificador completo usando fusión jerárquica de características
    """
    
    def __init__(self, tcn_filters: int = 64, tcn_kernel_size: int = 3,
                 tcn_blocks: int = 3, fusion_dim: int = 128,
                 n_classes: int = 2, dropout: float = 0.3):
        """
        Args:
            tcn_filters: Número de filtros en TCN
            tcn_kernel_size: Tamaño del kernel en TCN
            tcn_blocks: Número de bloques TCN
            fusion_dim: Dimensión de la capa de fusión
            n_classes: Número de clases
            dropout: Tasa de dropout
        """
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_blocks = tcn_blocks
        self.fusion_dim = fusion_dim
        self.n_classes = n_classes
        self.dropout = dropout
        
        self.tcn_model = None
        self.scaler_linear = StandardScaler()
        self.scaler_nonlinear = StandardScaler()
        self.fusion_model = None
        self.classifier_model = None
        self.input_length = None
    
    def _prepare_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Preparar señal para TCN (60 segundos = 7680 muestras a 128Hz)"""
        if len(signal) >= target_length:
            # Truncar
            return signal[:target_length]
        else:
            # Interpolar
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, signal, kind='linear', fill_value='extrapolate')
            return f(x_new)
    
    def fit(self, X: List[np.ndarray], y: np.ndarray, fs: float = 128.0,
            epochs: int = 50, batch_size: int = 16, validation_split: float = 0.2):
        """
        Entrenar clasificador
        
        Args:
            X: Lista de señales ECG
            y: Etiquetas de clase
            fs: Frecuencia de muestreo
            epochs: Número de épocas
            batch_size: Tamaño de batch
            validation_split: Proporción de validación
        """
        print(f"🔧 Preparando {len(X)} señales para entrenamiento...")
        
        # Determinar longitud de entrada (60 segundos)
        self.input_length = int(60 * fs)
        
        # Extraer características lineales
        print(f"📊 Extrayendo características lineales...")
        linear_features = []
        for signal in X:
            features = extract_linear_features(signal, fs)
            linear_features.append([
                features['mean_rr'],
                features['std_rr'],
                features['qrs_width'],
                features['t_amplitude'],
                features['t_width']
            ])
        linear_features = np.array(linear_features)
        linear_features_scaled = self.scaler_linear.fit_transform(linear_features)
        
        # Extraer características no lineales
        print(f"📈 Extrayendo características no lineales...")
        nonlinear_features = []
        for signal in X:
            features = extract_nonlinear_features(signal, fs)
            nonlinear_features.append([
                features['dfa_alpha1'],
                features['sample_entropy'],
                features['approximate_entropy']
            ])
        nonlinear_features = np.array(nonlinear_features)
        nonlinear_features_scaled = self.scaler_nonlinear.fit_transform(nonlinear_features)
        
        # Preparar señales para TCN
        print(f"🧠 Preparando señales para TCN-Seq2vec...")
        tcn_inputs = []
        for signal in X:
            prepared = self._prepare_signal(signal, self.input_length)
            tcn_inputs.append(prepared)
        tcn_inputs = np.array(tcn_inputs)
        tcn_inputs = tcn_inputs.reshape(-1, self.input_length, 1)
        
        # Crear y entrenar modelo TCN
        print(f"🤖 Creando modelo TCN-Seq2vec...")
        self.tcn_model = create_tcn_seq2vec(
            self.input_length,
            n_filters=self.tcn_filters,
            kernel_size=self.tcn_kernel_size,
            n_blocks=self.tcn_blocks,
            dropout=self.dropout
        )
        
        # Entrenar TCN (pre-entrenamiento)
        print(f"🎓 Entrenando TCN-Seq2vec...")
        self.tcn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        # Crear target dummy para pre-entrenamiento (autoencoder-like)
        tcn_targets = tcn_inputs.mean(axis=1, keepdims=True)
        self.tcn_model.fit(
            tcn_inputs, tcn_targets,
            epochs=10,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        # Extraer características profundas
        print(f"🔍 Extrayendo características profundas...")
        deep_features = self.tcn_model.predict(tcn_inputs, verbose=0)
        
        # Crear modelo de fusión jerárquica
        print(f"🔗 Creando modelo de fusión jerárquica...")
        
        # Inputs
        linear_input = layers.Input(shape=(5,), name='linear_input')
        nonlinear_input = layers.Input(shape=(3,), name='nonlinear_input')
        deep_input = layers.Input(shape=(self.tcn_filters,), name='deep_input')
        
        # Normalización
        linear_norm = layers.BatchNormalization()(linear_input)
        nonlinear_norm = layers.BatchNormalization()(nonlinear_input)
        deep_norm = layers.BatchNormalization()(deep_input)
        
        # Proyección a dimensión común
        linear_proj = layers.Dense(self.fusion_dim, activation='relu')(linear_norm)
        nonlinear_proj = layers.Dense(self.fusion_dim, activation='relu')(nonlinear_norm)
        deep_proj = layers.Dense(self.fusion_dim, activation='relu')(deep_norm)
        
        # Fusión jerárquica (concatenación + capa densa)
        fused = layers.Concatenate()([linear_proj, nonlinear_proj, deep_proj])
        fused = layers.Dense(self.fusion_dim * 2, activation='relu')(fused)
        fused = layers.Dropout(self.dropout)(fused)
        fused = layers.Dense(self.fusion_dim, activation='relu')(fused)
        
        # Clasificación
        output = layers.Dense(self.n_classes, activation='softmax')(fused)
        
        self.fusion_model = Model(
            inputs=[linear_input, nonlinear_input, deep_input],
            outputs=output
        )
        
        self.fusion_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar modelo de fusión
        print(f"🎯 Entrenando modelo de fusión jerárquica...")
        
        # Convertir y a formato one-hot si es necesario
        y_categorical = keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        self.fusion_model.fit(
            [linear_features_scaled, nonlinear_features_scaled, deep_features],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print(f"✅ Clasificador entrenado")
    
    def predict(self, X: List[np.ndarray], fs: float = 128.0) -> np.ndarray:
        """Predecir clases"""
        # Extraer características
        linear_features = []
        nonlinear_features = []
        tcn_inputs = []
        
        for signal in X:
            # Lineales
            features = extract_linear_features(signal, fs)
            linear_features.append([
                features['mean_rr'],
                features['std_rr'],
                features['qrs_width'],
                features['t_amplitude'],
                features['t_width']
            ])
            
            # No lineales
            features = extract_nonlinear_features(signal, fs)
            nonlinear_features.append([
                features['dfa_alpha1'],
                features['sample_entropy'],
                features['approximate_entropy']
            ])
            
            # TCN
            prepared = self._prepare_signal(signal, self.input_length)
            tcn_inputs.append(prepared)
        
        linear_features = np.array(linear_features)
        nonlinear_features = np.array(nonlinear_features)
        tcn_inputs = np.array(tcn_inputs).reshape(-1, self.input_length, 1)
        
        # Escalar
        linear_features_scaled = self.scaler_linear.transform(linear_features)
        nonlinear_features_scaled = self.scaler_nonlinear.transform(nonlinear_features)
        
        # Extraer características profundas
        deep_features = self.tcn_model.predict(tcn_inputs, verbose=0)
        
        # Predecir
        predictions = self.fusion_model.predict(
            [linear_features_scaled, nonlinear_features_scaled, deep_features],
            verbose=0
        )
        
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: List[np.ndarray], fs: float = 128.0) -> np.ndarray:
        """Obtener probabilidades"""
        # Similar a predict pero retorna probabilidades
        linear_features = []
        nonlinear_features = []
        tcn_inputs = []
        
        for signal in X:
            features = extract_linear_features(signal, fs)
            linear_features.append([
                features['mean_rr'],
                features['std_rr'],
                features['qrs_width'],
                features['t_amplitude'],
                features['t_width']
            ])
            
            features = extract_nonlinear_features(signal, fs)
            nonlinear_features.append([
                features['dfa_alpha1'],
                features['sample_entropy'],
                features['approximate_entropy']
            ])
            
            prepared = self._prepare_signal(signal, self.input_length)
            tcn_inputs.append(prepared)
        
        linear_features = np.array(linear_features)
        nonlinear_features = np.array(nonlinear_features)
        tcn_inputs = np.array(tcn_inputs).reshape(-1, self.input_length, 1)
        
        linear_features_scaled = self.scaler_linear.transform(linear_features)
        nonlinear_features_scaled = self.scaler_nonlinear.transform(nonlinear_features)
        
        deep_features = self.tcn_model.predict(tcn_inputs, verbose=0)
        
        probabilities = self.fusion_model.predict(
            [linear_features_scaled, nonlinear_features_scaled, deep_features],
            verbose=0
        )
        
        return probabilities
    
    def save(self, filepath: str):
        """Guardar modelo"""
        self.fusion_model.save(f"{filepath}_fusion.h5")
        self.tcn_model.save(f"{filepath}_tcn.h5")
        
        # Guardar scalers
        with open(f"{filepath}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'scaler_linear': self.scaler_linear,
                'scaler_nonlinear': self.scaler_nonlinear
            }, f)
        
        # Guardar metadatos
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'input_length': self.input_length,
                'tcn_filters': self.tcn_filters,
                'fusion_dim': self.fusion_dim,
                'n_classes': self.n_classes
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar modelo"""
        # Cargar metadatos
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Crear instancia
        instance = cls(
            tcn_filters=metadata['tcn_filters'],
            fusion_dim=metadata['fusion_dim'],
            n_classes=metadata['n_classes']
        )
        instance.input_length = metadata['input_length']
        
        # Cargar modelos
        instance.tcn_model = keras.models.load_model(f"{filepath}_tcn.h5")
        instance.fusion_model = keras.models.load_model(f"{filepath}_fusion.h5")
        
        # Cargar scalers
        with open(f"{filepath}_scalers.pkl", 'rb') as f:
            scalers = pickle.load(f)
        instance.scaler_linear = scalers['scaler_linear']
        instance.scaler_nonlinear = scalers['scaler_nonlinear']
        
        return instance

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Método 2: Fusión Jerárquica")
    print("=" * 50)
    
    # Crear señales de ejemplo
    np.random.seed(42)
    fs = 128.0
    signal_length = int(60 * fs)  # 60 segundos
    
    normal_signals = [
        np.sin(2 * np.pi * 1.2 * np.linspace(0, 60, signal_length)) + 
        0.1 * np.random.randn(signal_length)
        for _ in range(25)
    ]
    
    prescd_signals = [
        np.sin(2 * np.pi * 1.5 * np.linspace(0, 60, signal_length)) + 
        0.2 * np.random.randn(signal_length)
        for _ in range(25)
    ]
    
    X = normal_signals + prescd_signals
    y = np.array([0] * 25 + [1] * 25)
    
    # Entrenar clasificador
    classifier = HierarchicalFusionClassifier(
        tcn_filters=32,
        fusion_dim=64,
        n_classes=2
    )
    
    classifier.fit(X, y, fs=fs, epochs=5, batch_size=8)
    
    # Predecir
    predictions = classifier.predict(X[:10], fs=fs)
    probabilities = classifier.predict_proba(X[:10], fs=fs)
    
    print(f"\n📊 Resultados:")
    print(f"   Predicciones (primeras 10): {predictions}")
    print(f"   Probabilidades clase 1: {probabilities[:, 1][:5]}")
    
    print(f"\n💡 Para usar en el proyecto:")
    print(f"   from src.hierarchical_fusion import HierarchicalFusionClassifier")
    print(f"   classifier = HierarchicalFusionClassifier()")
    print(f"   classifier.fit(X_train, y_train, fs=128.0)")

