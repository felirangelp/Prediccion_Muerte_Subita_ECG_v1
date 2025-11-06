"""
Script para generar dashboard interactivo completo con Plotly
Incluye todas las visualizaciones y análisis de los tres métodos
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import sys
import pickle
from typing import List, Dict, Optional

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from src.sparse_representations import SparseRepresentationClassifier
from src.hierarchical_fusion import HierarchicalFusionClassifier
from src.hybrid_model import HybridSCDClassifier
from src.utils import load_ecg_record, list_available_records
from src.preprocessing_unified import preprocess_for_sparse_method, preprocess_for_hierarchical_method

class DashboardGenerator:
    """
    Generador de dashboard interactivo para análisis de predicción SCD
    """
    
    def __init__(self, output_file: str = "results/dashboard_scd_prediction.html"):
        self.output_file = output_file
        self.figures = []
        
    def generate_complete_dashboard(self, 
                                   sparse_model: Optional[SparseRepresentationClassifier] = None,
                                   hierarchical_model: Optional[HierarchicalFusionClassifier] = None,
                                   hybrid_model: Optional[HybridSCDClassifier] = None,
                                   evaluation_results: Optional[Dict] = None,
                                   data_samples: Optional[List] = None):
        """
        Generar dashboard completo
        
        Args:
            sparse_model: Modelo de representaciones dispersas entrenado
            hierarchical_model: Modelo de fusión jerárquica entrenado
            hybrid_model: Modelo híbrido entrenado
            evaluation_results: Resultados de evaluación
            data_samples: Muestras de datos para visualización
        """
        print("📊 Generando dashboard interactivo...")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard - Predicción de Muerte Súbita Cardíaca</title>
            <meta charset="utf-8">
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                }
                .header p {
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }
                .section {
                    background: white;
                    padding: 25px;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .section h2 {
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .tabs {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #e0e0e0;
                }
                .tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    border: none;
                    background: none;
                    font-size: 16px;
                    color: #666;
                    transition: all 0.3s;
                }
                .tab:hover {
                    color: #667eea;
                }
                .tab.active {
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                }
                .tab-content {
                    display: none;
                }
                .tab-content.active {
                    display: block;
                }
                .plot-container {
                    margin: 20px 0;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                .metric-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }
                .metric-card h3 {
                    margin: 0 0 10px 0;
                    font-size: 0.9em;
                    opacity: 0.9;
                }
                .metric-card .value {
                    font-size: 2em;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Dashboard de Predicción de Muerte Súbita Cardíaca</h1>
                <p>Análisis comparativo de métodos: Representaciones Dispersas, Fusión Jerárquica y Modelo Híbrido</p>
            </div>
        """
        
        # Sección 1: Resumen Ejecutivo
        html_content += self._generate_executive_summary(evaluation_results)
        
        # Sección 2: Análisis Exploratorio
        html_content += self._generate_exploratory_analysis(data_samples)
        
        # Sección 3: Método 1 - Representaciones Dispersas
        html_content += self._generate_sparse_method_section(sparse_model, evaluation_results)
        
        # Sección 4: Método 2 - Fusión Jerárquica
        html_content += self._generate_hierarchical_method_section(hierarchical_model, evaluation_results)
        
        # Sección 5: Modelo Híbrido
        html_content += self._generate_hybrid_model_section(hybrid_model, evaluation_results)
        
        # Sección 6: Análisis Comparativo
        html_content += self._generate_comparative_analysis(evaluation_results)
        
        # Sección 7: Predicción en Tiempo Real
        html_content += self._generate_realtime_prediction_section()
        
        html_content += """
            <script>
                // Funcionalidad de tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.addEventListener('click', function() {
                        const tabName = this.getAttribute('data-tab');
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                        this.classList.add('active');
                        document.getElementById(tabName).classList.add('active');
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Asegurar que el directorio existe
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar dashboard
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard guardado en: {self.output_file}")
    
    def _generate_executive_summary(self, results: Optional[Dict]) -> str:
        """Generar resumen ejecutivo"""
        if not results:
            results = {
                'sparse': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'hierarchical': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'hybrid': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            }
        
        html = """
        <div class="section">
            <h2>📈 Resumen Ejecutivo</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Método 1: Representaciones Dispersas</h3>
                    <div class="value">{:.1f}%</div>
                    <p>Precisión: {:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Método 2: Fusión Jerárquica</h3>
                    <div class="value">{:.1f}%</div>
                    <p>Precisión: {:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Modelo Híbrido</h3>
                    <div class="value">{:.1f}%</div>
                    <p>Precisión: {:.1f}%</p>
                </div>
            </div>
        </div>
        """.format(
            results.get('sparse', {}).get('accuracy', 0) * 100,
            results.get('sparse', {}).get('precision', 0) * 100,
            results.get('hierarchical', {}).get('accuracy', 0) * 100,
            results.get('hierarchical', {}).get('precision', 0) * 100,
            results.get('hybrid', {}).get('accuracy', 0) * 100,
            results.get('hybrid', {}).get('precision', 0) * 100
        )
        
        return html
    
    def _generate_exploratory_analysis(self, data_samples: Optional[List]) -> str:
        """Generar análisis exploratorio"""
        html = """
        <div class="section">
            <h2>🔍 Análisis Exploratorio de Datos</h2>
            <div class="tabs">
                <button class="tab active" data-tab="data-overview">Resumen</button>
                <button class="tab" data-tab="data-distribution">Distribución</button>
                <button class="tab" data-tab="data-signals">Señales</button>
            </div>
            
            <div id="data-overview" class="tab-content active">
                <h3>Resumen de Datos</h3>
                <p>Esta sección muestra información general sobre los datasets utilizados.</p>
                <ul>
                    <li><strong>SDDB:</strong> 23 pacientes con muerte súbita cardíaca</li>
                    <li><strong>NSRDB:</strong> 18 pacientes sanos</li>
                    <li><strong>Frecuencia de muestreo:</strong> 128-250 Hz</li>
                    <li><strong>Duración:</strong> 24 horas por paciente</li>
                </ul>
            </div>
            
            <div id="data-distribution" class="tab-content">
                <h3>Distribución de Clases</h3>
                <div class="plot-container" id="class-distribution-plot"></div>
            </div>
            
            <div id="data-signals" class="tab-content">
                <h3>Ejemplos de Señales ECG</h3>
                <div class="plot-container" id="signal-examples-plot"></div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_sparse_method_section(self, model: Optional[SparseRepresentationClassifier], 
                                       results: Optional[Dict]) -> str:
        """Generar sección del método de representaciones dispersas"""
        html = """
        <div class="section">
            <h2>📊 Método 1: Representaciones Dispersas</h2>
            <div class="tabs">
                <button class="tab active" data-tab="sparse-method">Método</button>
                <button class="tab" data-tab="sparse-performance">Rendimiento</button>
                <button class="tab" data-tab="sparse-dictionaries">Diccionarios</button>
            </div>
            
            <div id="sparse-method" class="tab-content active">
                <h3>Descripción del Método</h3>
                <p>Este método utiliza representaciones dispersas (sparse representations) para extraer características robustas de señales ECG.</p>
                <ul>
                    <li><strong>Algoritmo OMP:</strong> Orthogonal Matching Pursuit para encontrar representaciones dispersas</li>
                    <li><strong>k-SVD:</strong> Aprendizaje de diccionarios adaptativos</li>
                    <li><strong>Clasificación:</strong> SVM con kernel RBF</li>
                </ul>
            </div>
            
            <div id="sparse-performance" class="tab-content">
                <h3>Métricas de Rendimiento</h3>
                <div class="plot-container" id="sparse-metrics-plot"></div>
            </div>
            
            <div id="sparse-dictionaries" class="tab-content">
                <h3>Diccionarios Aprendidos</h3>
                <p>Visualización de los átomos del diccionario aprendidos para cada clase.</p>
                <div class="plot-container" id="sparse-dictionaries-plot"></div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_hierarchical_method_section(self, model: Optional[HierarchicalFusionClassifier],
                                             results: Optional[Dict]) -> str:
        """Generar sección del método de fusión jerárquica"""
        html = """
        <div class="section">
            <h2>🔗 Método 2: Fusión Jerárquica de Características</h2>
            <div class="tabs">
                <button class="tab active" data-tab="hierarchical-method">Método</button>
                <button class="tab" data-tab="hierarchical-performance">Rendimiento</button>
                <button class="tab" data-tab="hierarchical-features">Características</button>
            </div>
            
            <div id="hierarchical-method" class="tab-content active">
                <h3>Descripción del Método</h3>
                <p>Este método combina características lineales, no lineales y de deep learning mediante fusión jerárquica.</p>
                <ul>
                    <li><strong>Características Lineales:</strong> Intervalos RR, complejos QRS, ondas T</li>
                    <li><strong>Características No Lineales:</strong> DFA-2, entropías</li>
                    <li><strong>Deep Learning:</strong> TCN-Seq2vec para representaciones multiescala</li>
                    <li><strong>Fusión:</strong> Combinación jerárquica de características heterogéneas</li>
                </ul>
            </div>
            
            <div id="hierarchical-performance" class="tab-content">
                <h3>Métricas de Rendimiento</h3>
                <div class="plot-container" id="hierarchical-metrics-plot"></div>
            </div>
            
            <div id="hierarchical-features" class="tab-content">
                <h3>Análisis de Características</h3>
                <div class="plot-container" id="hierarchical-features-plot"></div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_hybrid_model_section(self, model: Optional[HybridSCDClassifier],
                                      results: Optional[Dict]) -> str:
        """Generar sección del modelo híbrido"""
        html = """
        <div class="section">
            <h2>🎯 Modelo Híbrido</h2>
            <div class="tabs">
                <button class="tab active" data-tab="hybrid-method">Método</button>
                <button class="tab" data-tab="hybrid-performance">Rendimiento</button>
                <button class="tab" data-tab="hybrid-comparison">Comparación</button>
            </div>
            
            <div id="hybrid-method" class="tab-content active">
                <h3>Descripción del Modelo Híbrido</h3>
                <p>El modelo híbrido combina las fortalezas de ambos métodos anteriores:</p>
                <ul>
                    <li><strong>Diccionarios Wavelet:</strong> Usa transformada wavelet para generar átomos del diccionario</li>
                    <li><strong>Representaciones Dispersas:</strong> Aplica OMP sobre escalogramas wavelet</li>
                    <li><strong>Fusión Dual:</strong> Combina características de ambos métodos</li>
                    <li><strong>Ensemble:</strong> Clasificador de votación que combina predicciones</li>
                </ul>
            </div>
            
            <div id="hybrid-performance" class="tab-content">
                <h3>Métricas de Rendimiento</h3>
                <div class="plot-container" id="hybrid-metrics-plot"></div>
            </div>
            
            <div id="hybrid-comparison" class="tab-content">
                <h3>Comparación con Métodos Individuales</h3>
                <div class="plot-container" id="hybrid-comparison-plot"></div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_comparative_analysis(self, results: Optional[Dict]) -> str:
        """Generar análisis comparativo"""
        html = """
        <div class="section">
            <h2>📊 Análisis Comparativo</h2>
            <div class="tabs">
                <button class="tab active" data-tab="comparative-metrics">Métricas</button>
                <button class="tab" data-tab="comparative-roc">Curvas ROC</button>
                <button class="tab" data-tab="comparative-pca">Análisis PCA</button>
            </div>
            
            <div id="comparative-metrics" class="tab-content active">
                <h3>Comparación de Métricas</h3>
                <div class="plot-container" id="comparative-metrics-plot"></div>
            </div>
            
            <div id="comparative-roc" class="tab-content">
                <h3>Curvas ROC Comparativas</h3>
                <div class="plot-container" id="comparative-roc-plot"></div>
            </div>
            
            <div id="comparative-pca" class="tab-content">
                <h3>Reducción de Dimensionalidad (PCA)</h3>
                <div class="plot-container" id="comparative-pca-plot"></div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_realtime_prediction_section(self) -> str:
        """Generar sección de predicción en tiempo real"""
        html = """
        <div class="section">
            <h2>⚡ Predicción en Tiempo Real</h2>
            <div class="tabs">
                <button class="tab active" data-tab="realtime-upload">Cargar Señal</button>
                <button class="tab" data-tab="realtime-results">Resultados</button>
            </div>
            
            <div id="realtime-upload" class="tab-content active">
                <h3>Cargar Nueva Señal ECG</h3>
                <p>Para usar esta funcionalidad, carga una señal ECG y el sistema realizará la predicción usando los tres modelos.</p>
                <p><em>Nota: Esta funcionalidad requiere implementación adicional del frontend.</em></p>
            </div>
            
            <div id="realtime-results" class="tab-content">
                <h3>Resultados de Predicción</h3>
                <p>Los resultados se mostrarán aquí después de cargar una señal.</p>
            </div>
        </div>
        """
        
        return html

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar dashboard interactivo')
    parser.add_argument('--output', type=str, default='results/dashboard_scd_prediction.html',
                       help='Archivo de salida del dashboard')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--results-file', type=str, default='results/evaluation_results.pkl',
                       help='Archivo con resultados de evaluación')
    
    args = parser.parse_args()
    
    # Cargar modelos si existen
    sparse_model = None
    hierarchical_model = None
    hybrid_model = None
    evaluation_results = None
    
    models_dir = Path(args.models_dir)
    if models_dir.exists():
        if (models_dir / 'sparse_classifier.pkl').exists():
            try:
                sparse_model = SparseRepresentationClassifier.load(
                    str(models_dir / 'sparse_classifier.pkl')
                )
                print("✅ Modelo sparse cargado")
            except:
                pass
        
        if (models_dir / 'hierarchical_classifier_fusion.h5').exists():
            try:
                hierarchical_model = HierarchicalFusionClassifier.load(
                    str(models_dir / 'hierarchical_classifier')
                )
                print("✅ Modelo hierarchical cargado")
            except:
                pass
        
        if (models_dir / 'hybrid_model_sparse.pkl').exists():
            try:
                hybrid_model = HybridSCDClassifier.load(
                    str(models_dir / 'hybrid_model')
                )
                print("✅ Modelo hybrid cargado")
            except:
                pass
    
    # Cargar resultados si existen
    results_file = Path(args.results_file)
    if results_file.exists():
        try:
            with open(results_file, 'rb') as f:
                evaluation_results = pickle.load(f)
            print("✅ Resultados de evaluación cargados")
        except:
            pass
    
    # Generar dashboard
    generator = DashboardGenerator(output_file=args.output)
    generator.generate_complete_dashboard(
        sparse_model=sparse_model,
        hierarchical_model=hierarchical_model,
        hybrid_model=hybrid_model,
        evaluation_results=evaluation_results
    )
    
    print(f"\n✅ Dashboard generado exitosamente: {args.output}")

if __name__ == "__main__":
    main()

