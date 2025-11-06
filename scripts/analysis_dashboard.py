import argparse
import os
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Añadir src al path para importar módulos locales
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import list_available_records, load_ecg_record
from src.preprocessing import preprocess_ecg_signal, extract_features


def process_record(record_path: str, label: int) -> dict:
    """Carga, preprocesa y extrae características de un único registro."""
    try:
        signal, metadata = load_ecg_record(record_path, channels=[0])
        
        # Limitar la señal a los primeros 15 minutos para simplificar el análisis
        max_samples = 15 * 60 * metadata['fs']
        if signal.shape[0] > max_samples:
            signal = signal[:max_samples, :]
            
        processed_signal = preprocess_ecg_signal(signal, metadata['fs'])
        features = extract_features(processed_signal, metadata['fs'])
        
        if features:
            features['record_name'] = metadata['record_name']
            features['label'] = label
            return features
            
    except Exception as e:
        print(f"⚠️  Error procesando {record_path}: {e}")
    
    return None

def main(args):
    """Función principal para el análisis y generación de dashboard."""
    print("🚀 Iniciando análisis de varianza y PCA para señales ECG")

    # Rutas a los datasets (ajustadas a la estructura real de descarga)
    sddb_path = Path(args.data_dir) / 'sddb' / 'physionet.org' / 'files' / 'sddb' / '1.0.0'
    nsrdb_path = Path(args.data_dir) / 'nsrdb' / 'physionet.org' / 'files' / 'nsrdb' / '1.0.0'

    print(f"📁 Directorio de datos: {args.data_dir}")
    print(f"   - Buscando en: {sddb_path}")
    print(f"   - Buscando en: {nsrdb_path}")

    # Paso 1: Cargar y procesar datos
    all_features = []

    # Cargar datos de muerte súbita (sddb)
    sddb_records = list_available_records(str(sddb_path))
    print(f"\n🔍 Encontrados {len(sddb_records)} registros en sddb.")
    
    for record_name in tqdm(sddb_records, desc="Procesando sddb"):
        record_path = str(sddb_path / record_name)
        features = process_record(record_path, label=1)
        if features:
            all_features.append(features)

    # Cargar datos de ritmo sinusal normal (nsrdb)
    nsrdb_records = list_available_records(str(nsrdb_path))
    print(f"🔍 Encontrados {len(nsrdb_records)} registros en nsrdb.")
    
    for record_name in tqdm(nsrdb_records, desc="Procesando nsrdb"):
        record_path = str(nsrdb_path / record_name)
        features = process_record(record_path, label=0)
        if features:
            all_features.append(features)
        
    print("\n✅ Carga y procesamiento de datos completado.")
    print("📊 Total de registros procesados:", len(all_features))

    if not all_features:
        print("❌ No se pudieron procesar registros. Abortando.")
        return

    # Paso 2: Crear y limpiar DataFrame
    features_df = pd.DataFrame(all_features).set_index('record_name')
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.dropna(axis=1, how='any') # Eliminar columnas con NaN

    if features_df.empty:
        print("❌ El DataFrame de características está vacío después de la limpieza. Abortando.")
        return

    print(f"📊 DataFrame de características creado con forma: {features_df.shape}")

    # Paso 3: Análisis de Varianza
    X = features_df.drop('label', axis=1)
    y = features_df['label']
    
    # Estandarizar datos antes de PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcular varianza de características estandarizadas
    variances = pd.Series(np.var(X_scaled, axis=0), index=X.columns).sort_values(ascending=False)
    print("\n📈 Varianza de las 5 características principales:")
    print(variances.head())

    # Paso 4: Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\n🧠 PCA aplicado. Número de componentes principales: {pca.n_components_}")
    
    # Paso 5: Generar dashboard
    print(f"\n⏳ Generando dashboard en {args.output_file}...")
    generate_dashboard(
        variances, 
        X, 
        y, 
        X_pca, 
        pca, 
        args.output_file
    )
    
    print(f"✅ Dashboard generado exitosamente en: file://{os.path.abspath(args.output_file)}")


def generate_dashboard(variances, X, y, X_pca, pca, output_file):
    """Genera un dashboard HTML interactivo con los resultados del análisis."""
    
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {"type": "scene", "rowspan": 2}],
               [{}, None],
               [{"colspan": 2}, None]],
        subplot_titles=(
            "Varianza de las Características (Top 20)", 
            "Espacio de Características 3D (Mayor Varianza)",
            "Varianza Explicada por Componente Principal (Scree Plot)",
            "Análisis de Componentes Principales (PCA)"
        )
    )

    # 1. Gráfico de Varianza
    top_variances = variances.head(20)
    fig.add_trace(
        go.Bar(x=top_variances.index, y=top_variances.values, name='Varianza'),
        row=1, col=1
    )

    # 2. Gráfico 3D
    top_3_features = variances.index[:3]
    colors = y.map({0: 'blue', 1: 'red'})
    symbols = y.map({0: 'circle', 1: 'diamond'})
    
    fig.add_trace(
        go.Scatter3d(
            x=X[top_3_features[0]], y=X[top_3_features[1]], z=X[top_3_features[2]],
            mode='markers',
            marker=dict(color=colors, symbol=symbols, size=5, opacity=0.7),
            name='Pacientes'
        ),
        row=1, col=2
    )

    # 3. Scree Plot
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(explained_variance_ratio) + 1)),
            y=explained_variance_ratio,
            name='Varianza Individual'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            mode='lines+markers',
            name='Varianza Acumulada'
        ),
        row=2, col=1
    )

    # Definir el número de componentes para alcanzar ~95% de varianza
    n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
    
    # 4. Texto explicativo
    num_original_features = X.shape[1]
    reduction_percentage = (1 - (n_components_95 / num_original_features)) * 100
    
    explanation_html = f"""
    <h2>Análisis y Conclusiones</h2>
    <p>
        Este dashboard presenta un análisis exploratorio de dos bases de datos de ECG: 
        <b>sddb</b> (pacientes con muerte súbita) y <b>nsrdb</b> (pacientes sanos).
    </p>
    
    <h3>¿Qué criterio se utilizó para definir el número de componentes?</h3>
    <p>
        Se utilizó el criterio de la <b>varianza acumulada explicada</b>. El objetivo es retener 
        suficientes componentes principales para explicar un alto porcentaje de la variabilidad 
        total de los datos, comúnmente entre el 95% y el 99%.
    </p>
    <p>
        Para este conjunto de datos, se necesitan <b>{n_components_95} componentes</b> para 
        capturar aproximadamente el <b>95% de la varianza total</b>. Esto se puede observar en el
        "Scree Plot", donde la línea de varianza acumulada cruza el umbral del 0.95.
    </p>
    
    <h3>¿En cuánto se reduce la dimensionalidad del problema?</h3>
    <p>
        La dimensionalidad original del problema, dada por el número total de características extraídas, 
        es de <b>{num_original_features} características</b>.
    </p>
    <p>
        Al seleccionar los primeros {n_components_95} componentes principales, se logra una 
        reducción de dimensionalidad del <b>{reduction_percentage:.2f}%</b>.
    </p>
    """
    
    # Configuración final y guardado
    fig.update_layout(
        height=1200, 
        title_text="Dashboard de Análisis de Varianza y PCA para ECG",
        scene=dict(
            xaxis_title=top_3_features[0],
            yaxis_title=top_3_features[1],
            zaxis_title=top_3_features[2]
        )
    )
    
    with open(output_file, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(explanation_html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Genera un dashboard de análisis de varianza y PCA para datasets de ECG."
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='datasets',
        help="Directorio raíz donde se encuentran los subdirectorios de los datasets (sddb, nsrdb)."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='dashboard_analisis_varianza.html',
        help="Archivo HTML de salida para el dashboard."
    )
    
    args = parser.parse_args()
    main(args)
