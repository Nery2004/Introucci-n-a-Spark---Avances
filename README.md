# Laboratorio 7 – Spark MLlib

Universidad del Valle de Guatemala  
CC3066 – Data Science  
Semestre II – 2026  

**Estudiante:** Nery Molina – 23218

## Objetivo

Este repositorio contiene el entregable reproducible del Laboratorio 7. Se
trabaja exclusivamente con las bases de **Personas** de la Encuesta Nacional de
Empleo e Ingresos Continua (ENEIC): se armonizan los cuatro trimestres de 2025,
se revisa calidad, se realizan análisis descriptivos y de correlación, se crean
perfiles con KMeans y se comparan pipelines de regresión lineal y Random Forest.

Los cuatro períodos de 2025 se usan para preparación, exploración, clustering,
train/validation y selección de hiperparámetros. Personas ENEIC I-2026 se
reserva exclusivamente para la evaluación final. El objetivo es `P05D01`,
renombrado como `salario_mensual`; no se utiliza `YLAB_PUBLI`, otros ingresos,
clusters ni identificadores como predictores.

## Estructura

```text
.
├── notebooks/
│   └── Laboratorio_7_SparkML_Nery_Molina_23218.ipynb
├── data/
│   ├── raw/                     # Excel locales de Personas, fuera de Git
│   └── processed/               # Parquet locales, fuera de Git
├── models/                      # Pipelines Spark locales, fuera de Git
├── outputs/
│   ├── figures/                 # Figuras generadas por el notebook
│   └── tables/                  # Tablas CSV generadas por el notebook
├── scripts/
│   ├── build_notebook.py        # Genera el notebook desde su fuente revisable
│   └── setup_windows_hadoop.ps1 # Bootstrap local de winutils en Windows
├── src/
│   └── utils.py                 # Funciones Spark-first reutilizables
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10 o superior.
- Java 17 recomendado para Spark 3.5.x.
- PySpark 3.5.x y `pyspark.ml`.
- JupyterLab o Jupyter Notebook.
- `pandas` y `openpyxl` solo para la lectura inicial de Excel y agregados
  pequeños necesarios para tablas o gráficos.
- `matplotlib` y `seaborn` para visualización.
- En Python 3.12, `setuptools` proporciona la compatibilidad `distutils` que
  requiere PySpark 3.5.x.

En Windows, Spark/Hadoop requiere `winutils.exe` y `hadoop.dll` para escribir
Parquet. El script de bootstrap instala de forma local y no versionada las
utilidades compatibles con Hadoop 3.3.6. El notebook configura `HADOOP_HOME`
solo para su propia sesión.

## Datos ENEIC

Coloque únicamente los Excel de **Personas** en `data/raw/`:

- `Base-de-datos-Personas-ENEIC-I-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-II-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-III-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-IV-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-I-2026.xlsx`

Para I-2026 se utiliza la hoja `Personas_ENEIC_T1-2026`; para 2025 se lee la
primera hoja de cada archivo. Los diccionarios de Personas pueden guardarse en
la misma carpeta para consulta, pero no se cargan como bases analíticas. No
coloque ni use archivos de Hogares, Vivienda u otros módulos.

Los datasets, los Parquet, la carpeta de modelos y las herramientas locales no
se versionan. Las tablas y figuras creadas con resultados reproducibles sí se
guardan en `outputs/`.

## Ejecución

Desde PowerShell, en la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/setup_windows_hadoop.ps1
python scripts/build_notebook.py
jupyter lab
```

Abra `notebooks/Laboratorio_7_SparkML_Nery_Molina_23218.ipynb` y ejecute
**Kernel → Restart Kernel and Run All Cells**.

Para ejecutar todo sin interfaz, use:

```powershell
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/Laboratorio_7_SparkML_Nery_Molina_23218.ipynb
python scripts/validate_lab.py
```

El notebook valida los cinco archivos y sus columnas antes de continuar. Si
falta un archivo, muestra el período y la ubicación esperada en lugar de
calcular resultados parciales o inventar métricas.

Por defecto el notebook usa `local[4]` para mantener el uso de memoria
controlado en Windows. Puede reemplazarse para otro entorno antes de ejecutar,
por ejemplo: `$env:SPARK_MASTER = "local[8]"`.

## Reproducibilidad y alcance

- `SEED = 42` se usa para KMeans, train/validation, Random Forest y muestras
  gráficas.
- La unión de 2025 utiliza `unionByName`, no unión por posición.
- Los filtros conservan personas de 15 años o más, ocupadas, con categoría
  ocupacional válida, salario positivo finito, antigüedad válida y horas entre
  1 y 168. No se imputan salarios ni se recortan outliers.
- Las métricas principales (MAE, RMSE y R²) se calculan con todos los registros
  de validación o test, nunca con una muestra.
- `FACTOR` se conserva. Un análisis poblacional formal requeriría incorporarlo
  junto con el diseño de encuesta; este laboratorio es no ponderado.

## Resultados producidos

El notebook escribe, cuando la ejecución finaliza correctamente:

- Parquet preparados en `data/processed/`.
- Tablas en `outputs/tables/`, incluyendo calidad, descriptivos, correlaciones,
  KMeans, validación, evaluación 2026 y errores por grupo/banda salarial.
- Figuras en `outputs/figures/`.
- Pipelines seleccionados y finales en `models/linear_regression_best`,
  `models/random_forest_best`, `models/linear_regression_final` y
  `models/random_forest_final`.
