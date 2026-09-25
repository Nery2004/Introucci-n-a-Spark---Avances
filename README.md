# Laboratorio 7 – Spark MLlib

Universidad del Valle de Guatemala  
CC3066 – Data Science  
Semestre II – 2026  

**Estudiante:** Nery Molina – 23218

## Propósito

Este repositorio contiene el avance del Laboratorio 7 sobre datos de **Personas**
de ENEIC. El avance se limita a:

1. Carga, armonización y calidad de datos.
2. Estadística descriptiva y exploración.
3. Relaciones entre variables numéricas.
4. Segmentación de perfiles con KMeans.

No incluye regresión lineal, Random Forest, entrenamiento supervisado,
evaluación final ni uso de I-2026 como conjunto de evaluación.

Los análisis empíricos de las actividades 2–4 usan exclusivamente los cuatro
períodos de 2025. El archivo Personas I-2026 se carga, armoniza y guarda por
separado para reservarlo como prueba final de actividades posteriores.

## Estructura

~~~
.
├── notebooks/
│   └── Laboratorio_7_Avance_Nery_Molina_23218.ipynb
├── data/
│   ├── raw/                 # Excel locales, excluidos de Git
│   └── processed/           # Parquet locales, excluidos de Git
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── utils.py
├── requirements.txt
└── README.md
~~~

## Requisitos

- Python 3.10 o 3.11 recomendado.
- Java 17 y Spark 3.5.x.
- JupyterLab o Jupyter Notebook.
- `pandas` y `openpyxl` para la lectura inicial de Excel.
- PySpark y `pyspark.ml` para el procesamiento, correlaciones y KMeans.
- `matplotlib` y `seaborn` para visualizaciones de agregados o muestras
  pequeñas.

Instale las dependencias desde la raíz del proyecto:

~~~
python -m pip install -r requirements.txt
~~~

## Datos ENEIC

Coloque únicamente los archivos de **Personas** en `data/raw/`. Se detectan
nombres parecidos a los siguientes:

- `Base-de-datos-Personas-ENEIC-I-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-II-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-III-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-IV-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-I-2026.xlsx`

No coloque ni use archivos de Hogares, Vivienda u otros módulos. El notebook
deriva el período del nombre de cada archivo, conserva la columna `TRIMESTRE`
original para auditoría y no la interpreta como trimestre calendario.

Los Excel y Parquet no se incluyen en Git por su tamaño y porque contienen
datos que deben permanecer en almacenamiento local autorizado. Las carpetas
vacías se conservan con `.gitkeep`.

## Ejecución

Desde la raíz del repositorio, con Java 17 y las dependencias instaladas:

~~~
jupyter lab
~~~

Abra `notebooks/Laboratorio_7_Avance_Nery_Molina_23218.ipynb` y ejecute
**Kernel → Restart Kernel and Run All Cells**. Si no están los cuatro archivos
de Personas 2025, el notebook prepara I-2026 por separado y muestra mensajes
claros en lugar de inventar estadísticas, correlaciones o clusters.

## Alcance y ponderación

Se conserva `FACTOR`, el factor de expansión de la encuesta. Podría utilizarse
para estimaciones poblacionales que consideren el diseño muestral; sin embargo,
este laboratorio solicita análisis no ponderados. Por ello, los resultados
describen los registros analizados y no deben presentarse como estimaciones
oficiales de toda Guatemala.

