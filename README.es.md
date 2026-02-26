# 📊 Pipeline de Credit Scoring con Machine Learning

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸_English-gray?style=for-the-badge" alt="English"></a>
  <a href="#-descripción-del-proyecto"><img src="https://img.shields.io/badge/🇪🇸_Español-selected-blue?style=for-the-badge" alt="Español"></a>
</p>

## 📌 Descripción del Proyecto

Este proyecto implementa un **pipeline completo de credit scoring**, diseñado para predecir la probabilidad de incumplimiento (default) de clientes bancarios utilizando técnicas de machine learning.

El enfoque no se limita al modelo, sino que prioriza:

- **Reproducibilidad**
- **Modularidad**
- **Interpretabilidad**
- **Preparación para producción**

> [!NOTE]
> El dataset utilizado es el [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) del repositorio UCI. Se utiliza la columna `default` (si el cliente tiene crédito en mora) como variable objetivo para el modelado de riesgo crediticio.

---

## 🎯 Problema a Resolver

Las instituciones financieras necesitan evaluar el riesgo crediticio antes de otorgar un préstamo.

**Objetivos:**

- Estimar la probabilidad de default
- Manejar datos altamente desbalanceados
- Construir un pipeline robusto y reutilizable
- Garantizar consistencia entre entrenamiento e inferencia

---

## 🧠 Características del Dataset

| Propiedad | Descripción |
|---|---|
| **Variable objetivo** | `default_binary` (binaria: 0/1) |
| **Distribución de clases** | ~98.3% sin default / ~1.7% con default |
| **Total de muestras** | 4,521 |
| **Tipos de features** | Predominantemente categóricas (empleo, educación, vivienda, préstamo, etc.) |
| **Features numéricas** | Edad, balance (binneadas en preprocesamiento) |

> [!WARNING]
> Debido al bajo porcentaje de la clase minoritaria (~1.7%), las métricas tradicionales como accuracy **no son adecuadas**. Se priorizan métricas basadas en ranking y probabilidad.

---

## 🔍 Análisis Exploratorio de Datos (EDA)

El EDA permitió:

- Confirmar la severidad del desbalanceo de clases
- Validar decisiones de binning
- Justificar el uso de **WOE (Weight of Evidence)**
- Entender relaciones entre variables categóricas y riesgo de default

**Hallazgos clave:**

- El fuerte desbalanceo justificó el uso de **métricas basadas en Precision-Recall**
- Algunas variables categóricas mostraron relaciones monótonas con el riesgo de default
- Se confirmó la idoneidad del **encoding WOE**

---

## 🧩 Arquitectura del Pipeline

```
data/dataset/bank.csv
       │
       ▼
┌─────────────────┐
│  1. Ingesta      │  Validación de esquema, tipos, fallo temprano
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Creación     │  Variable binaria desde columna 'default'
│     de Target    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Split        │  70/30 estratificado preservando proporciones
│  Estratificado   │
└────────┬────────┘
    ┌────┴────┐
    ▼         ▼
  Train      Test
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Bin +  │ │ Bin +  │
│ WOE    │ │ Aplicar│  ← Usa mappings WOE guardados (sin leakage)
│ Encode │ │ WOE   │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│  4. Entrena-     │  Logistic Regression + Random Forest
│     miento       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Evaluación   │  AP, ROC AUC, Gini, Brier, F1
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Selección    │  Filtro por ROC AUC mínimo → maximizar AP
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Artefactos   │  Guardar modelo + WOE + features + metadata
└─────────────────┘
```

### 1. Ingesta y Validación de Datos

- Validación de esquema contra un contrato de datos
- Verificación de tipos (numérico vs string)
- Fallo temprano ante datos inválidos

### 2. Feature Engineering

- **Binning personalizado**: Edad → `[young, adult, middle_age, senior]`, Balance → `[very_low, low, medium, high]`
- **Encoding WOE (Weight of Evidence)** con suavizado de Laplace
- Mappings WOE almacenados para inferencia consistente

### 3. Entrenamiento de Modelos

Modelos evaluados:

| Modelo | Configuración | Propósito |
|---|---|---|
| Logistic Regression | `class_weight="balanced"`, `solver="liblinear"` | Baseline interpretable |
| Random Forest | `class_weight="balanced"`, `n_estimators=200` | Baseline no lineal |

Ambos modelos usan `class_weight="balanced"` para manejar el desbalanceo ajustando internamente los pesos de las muestras.

---

## 📈 Estrategia de Evaluación

Dado el desbalanceo severo, la evaluación se centró en métricas **independientes del umbral**:

| Métrica | Propósito |
|---|---|
| **Average Precision (AP)** | Rendimiento Precision-Recall |
| **ROC AUC** | Capacidad de discriminación/ranking |
| **Coeficiente de Gini** | Estándar de la industria de crédito (`2 × ROC AUC − 1`) |
| **Brier Score** | Calidad de calibración de probabilidades |
| **F1 óptimo + umbral** | Análisis de trade-off (solo diagnóstico) |

> [!IMPORTANT]
> La optimización del umbral (Max F1) fue analizada pero **no se usó para selección de modelo**, ya que el umbral de decisión debe ser definido por requerimientos de negocio, no por maximizar una métrica.

---

## 🏆 Selección del Modelo

Un módulo personalizado `select_best_model`:

1. **Filtra** modelos por un umbral mínimo de ROC AUC (≥ 0.75)
2. **Ordena** los modelos válidos por la métrica primaria (Average Precision)
3. **Desempata** usando una métrica secundaria (Brier Score, menor es mejor)

**Selección final: Logistic Regression** superó a Random Forest:

- Mayor capacidad de ranking (ROC AUC y Gini)
- Mayor estabilidad frente al desbalanceo severo
- Mayor interpretabilidad — fundamental en credit scoring por regulación

---

## 📦 Gestión de Artefactos

El modelo final se guarda como un **artefacto versionado** usando `joblib`. Los artefactos incluyen:

- Modelo entrenado
- Mappings de encoding WOE
- Lista de features
- Metadatos y métricas de evaluación
- Identificador de versión

```
artifacts/
└── model_v1/
    └── model.joblib
```

Esto garantiza **reproducibilidad**, **trazabilidad**, y permite actualizaciones seguras (`model_v2`, `model_v3`, etc.).

---

## 🔮 Simulación de Inferencia

Un script independiente (`run_inference.py`) demuestra:

- Carga del artefacto guardado
- Aplicación del **mismo preprocesamiento** (binning + WOE con mappings guardados)
- Generación de probabilidades de default (`credit_score`)

**Output:** Un score de probabilidad continuo por cliente, apto para despliegue batch o vía API.

```bash
python run_inference.py
```

---

## 📁 Estructura del Proyecto

```
credit_scoring/
├── main.py                          # Orquestador del pipeline
├── run_inference.py                 # Script de simulación de inferencia
├── README.md                        # Documentación (English)
├── README.es.md                     # Documentación (Español)
├── data/
│   └── dataset/
│       └── bank.csv                 # Dataset crudo
├── artifacts/
│   └── model_v1/
│       └── model.joblib             # Artefacto del modelo serializado
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
└── source/
    ├── __init__.py
    ├── ingestion/
    │   └── load_data.py             # Carga y validación de datos
    ├── preprocessing/
    │   └── feature_engineering.py   # Binning, encoding WOE
    ├── training/
    │   └── model_training.py        # Split, selección de features, entrenamiento
    ├── evaluation/
    │   ├── model_evaluation.py      # Cómputo de métricas
    │   └── model_selection.py       # Selección del mejor modelo
    └── artifacts/
        └── artifact_manager.py      # Guardar/cargar artefactos
```

---

## ⚠️ Desafíos Encontrados

- **Desbalanceo severo de clases** (~1.7% tasa de default) limitó métricas basadas en recall y generó curvas Precision-Recall ruidosas
- **Problemas de compatibilidad** entre versiones de NumPy y bibliotecas compiladas
- **Resolución de módulos** al estructurar scripts de inferencia como puntos de entrada separados

Todos los problemas se resolvieron con gestión apropiada del entorno, versionado explícito de artefactos, y diseño modular del pipeline.

---

## 📌 Conclusiones

1. El desbalanceo de clases **limita significativamente el F1 y AP alcanzables** — esto es esperable, no un fallo
2. **ROC AUC y Gini son indicadores más estables** y confiables para este escenario
3. **Logistic Regression sigue siendo un baseline fuerte** para credit scoring por su interpretabilidad
4. El **versionado de artefactos es crítico** en sistemas ML reales
5. Un **pipeline correcto importa más** que perseguir ganancias métricas marginales

---

## 🚀 Trabajo Futuro

- [ ] Agregar **XGBoost** con restricciones monótonas
- [ ] Implementar **validación cruzada** para robustez
- [ ] Construir **despliegue vía API** (FastAPI)
- [ ] Agregar interface de **scoring por lotes**
- [ ] Implementar **monitoreo y detección de drift**
