# PROYECTO-OPCIONAL-MACHINE-LEARNING
# Proyecto de Machine Learning: Predicción del Rendimiento Académico

## Descripción del proyecto

Este proyecto tiene como objetivo aplicar técnicas de **Machine Learning** sobre un conjunto de datos de **1.000 estudiantes** para estudiar los factores relacionados con su rendimiento académico.

Se desarrollan dos problemas predictivos diferentes:

* **Regresión:** predicción de la `nota_final` del estudiante.
* **Clasificación:** predicción de si el estudiante **aprobará o suspenderá**.

El proyecto incluye las fases de **análisis exploratorio de datos (EDA), limpieza, tratamiento de valores nulos, análisis de outliers, codificación de variables categóricas, escalado, entrenamiento, validación y comparación de diferentes modelos de Machine Learning**.

---

## Objetivos

Los principales objetivos del proyecto son:

1. Analizar las características del conjunto de datos.
2. Detectar y tratar valores nulos y posibles valores atípicos.
3. Preparar las variables para su utilización en modelos de Machine Learning.
4. Evitar problemas de **data leakage** entre las variables objetivo.
5. Desarrollar un modelo de regresión para estimar la nota final.
6. Desarrollar un modelo de clasificación para predecir si un alumno aprobará.
7. Comparar diferentes técnicas de regularización.
8. Analizar el efecto del **desbalanceo de clases** en el problema de clasificación.
9. Interpretar la importancia de las variables en las predicciones.

---

## Dataset

El dataset utilizado contiene **1.000 observaciones y 11 variables**.

| Variable                    | Descripción                                   |
| --------------------------- | --------------------------------------------- |
| `horas_estudio_semanal`     | Horas de estudio realizadas semanalmente      |
| `nota_anterior`             | Calificación obtenida anteriormente           |
| `tasa_asistencia`           | Porcentaje de asistencia a clase              |
| `horas_sueno`               | Promedio diario de horas de sueño             |
| `edad`                      | Edad del estudiante                           |
| `nivel_dificultad`          | Nivel de dificultad percibido                 |
| `tiene_tutor`               | Indica si el estudiante dispone de tutor      |
| `horario_estudio_preferido` | Momento del día preferido para estudiar       |
| `estilo_aprendizaje`        | Estilo de aprendizaje del estudiante          |
| `nota_final`                | Nota final obtenida, entre 0 y 100            |
| `aprobado`                  | Variable binaria: 1 = aprobado y 0 = suspenso |

Para el problema de regresión, la variable objetivo es:

```text
nota_final
```

Para el problema de clasificación:

```text
aprobado
```

---

## Análisis exploratorio de datos

Durante el EDA se analizaron:

* Dimensiones y estructura del dataset.
* Tipos de variables.
* Estadísticos descriptivos.
* Distribución de las variables numéricas.
* Distribución de las variables categóricas.
* Valores duplicados.
* Valores nulos.
* Correlaciones.
* Relaciones entre las variables explicativas y las variables objetivo.
* Posibles inconsistencias.
* Valores atípicos mediante **IQR y Z-score**.

No se encontraron registros completamente duplicados ni valores numéricos claramente inconsistentes.

### Valores nulos

Se detectaron valores ausentes en tres variables:

| Variable                    | Valores nulos |
| --------------------------- | ------------: |
| `horas_sueno`               |           150 |
| `horario_estudio_preferido` |           100 |
| `estilo_aprendizaje`        |            50 |

Los valores categóricos ausentes se sustituyeron por:

```text
Unknown
```

Los valores ausentes de `horas_sueno` fueron imputados utilizando la **mediana**.

Tras la imputación, el dataset no contiene valores nulos.

---

## Principales resultados del EDA

La nota final media de los estudiantes es aproximadamente:

```text
71,44 puntos
```

La nota anterior media se sitúa en aproximadamente:

```text
69,89 puntos
```

Los alumnos estudian de media unas **10 horas semanales** y presentan una tasa de asistencia próxima al **74 %**.

El análisis muestra que las variables con mayor relación con el rendimiento académico son principalmente:

* Horas semanales de estudio.
* Nota anterior.
* Tasa de asistencia.

También se detectó un importante **desbalanceo en la variable `aprobado`**:

* Aprobados: aproximadamente **89,8 %**.
* Suspensos: aproximadamente **10,2 %**.

Este desequilibrio resulta especialmente importante para interpretar correctamente los modelos de clasificación.

---

## Preprocesamiento

El preprocesamiento se realizó de forma diferente para los problemas de regresión y clasificación.

### Regresión

El objetivo es predecir:

```text
nota_final
```

La variable `aprobado` fue eliminada para evitar **data leakage**, ya que se obtiene directamente a partir de la nota final.

Las variables categóricas fueron transformadas mediante **One-Hot Encoding**.

Posteriormente, los datos se dividieron en:

* 80 % entrenamiento.
* 20 % test.

Las variables numéricas se escalaron mediante:

```python
MinMaxScaler()
```

El escalador se ajustó únicamente sobre el conjunto de entrenamiento.

---

### Clasificación

El objetivo es predecir:

```text
aprobado
```

La variable `nota_final` fue eliminada antes del entrenamiento para evitar **data leakage**.

La transformación de variables categóricas se realizó mediante:

* Codificación binaria para `tiene_tutor`.
* Codificación ordinal para `nivel_dificultad`.
* One-Hot Encoding para `horario_estudio_preferido`.
* One-Hot Encoding para `estilo_aprendizaje`.

La división entrenamiento-test también se realizó en proporción **80/20**, utilizando:

```python
stratify=y_clas
```

para mantener la proporción de aprobados y suspensos.

---

# Modelos de regresión

Se analizaron cuatro modelos:

* Linear Regression.
* Ridge Regression.
* Lasso Regression.
* ElasticNet.

## Resultados

| Modelo     |  R² Test | MAE Test | RMSE Test |
| ---------- | -------: | -------: | --------: |
| Linear     |     0,36 |     5,82 |      7,25 |
| Ridge      |     0,36 | **5,81** |      7,24 |
| Lasso      | **0,37** |     5,85 |  **7,19** |
| ElasticNet |     0,32 |     6,04 |      7,48 |

Los resultados de **Linear, Ridge y Lasso son muy similares**.

Lasso obtiene ligeramente el mayor R² y el menor RMSE, mientras que Ridge presenta el menor MAE.

La regularización no produce, por tanto, una mejora sustancial respecto a la regresión lineal inicial.

---

## Interpretación de la regresión

La regresión lineal obtiene:

```text
R² Test = 0,36
MAE Test = 5,82
RMSE Test = 7,25
```

Esto significa que el modelo explica aproximadamente un **36 % de la variabilidad de las notas finales**.

El MAE indica que las predicciones se desvían, en promedio, aproximadamente **5,8 puntos de la nota real**.

Además, las métricas de entrenamiento y test son muy similares, por lo que no se observa un problema importante de **overfitting**.

---

## Variables más importantes en regresión

Las variables con mayores coeficientes son:

| Variable                 | Coeficiente aproximado |
| ------------------------ | ---------------------: |
| Horas de estudio semanal |                 +16,79 |
| Nota anterior            |                 +11,93 |
| Tasa de asistencia       |                  +7,50 |

Estos resultados muestran que las **horas de estudio, el rendimiento anterior y la asistencia** constituyen los principales factores predictivos de la nota final dentro del conjunto de variables disponibles.

---

# Modelo de clasificación

Para predecir si un estudiante aprobará se utilizaron diferentes variantes de **regresión logística**:

* Logistic Regression L2 - Ridge.
* Logistic Regression L1 - Lasso.
* Logistic Regression ElasticNet.

## Comparación de modelos

| Modelo              | Accuracy Test | Precision weighted | Recall weighted | F1 weighted |
| ------------------- | ------------: | -----------------: | --------------: | ----------: |
| Logistic L2 - Ridge |      **0,91** |          **0,918** |        **0,91** |   **0,875** |
| Logistic L1 - Lasso |          0,90 |              0,810 |            0,90 |       0,853 |
| Logistic ElasticNet |          0,90 |              0,810 |            0,90 |       0,853 |

La regresión logística con penalización **L2** obtiene los mejores resultados globales.

Sin embargo, debido al fuerte desbalanceo de las clases, la accuracy no es suficiente para evaluar correctamente el comportamiento del modelo.

---

## Problema del desbalanceo

El modelo L2 obtiene aproximadamente:

```text
Accuracy = 0,91
```

Pero el análisis por clases muestra:

| Clase    | Precision |   Recall |   F1 |
| -------- | --------: | -------: | ---: |
| Suspenso |      1,00 | **0,10** | 0,18 |
| Aprobado |      0,91 | **1,00** | 0,95 |

Aunque la accuracy es elevada, el modelo únicamente identifica correctamente aproximadamente el **10 % de los estudiantes que suspenden**.

La **Balanced Accuracy** es:

```text
0,55
```

Esto demuestra que una accuracy elevada puede resultar engañosa cuando las clases están fuertemente desequilibradas.

---

## Tratamiento del desbalanceo

Para mejorar la identificación de los alumnos suspensos se entrenó una regresión logística utilizando:

```python
class_weight='balanced'
```

Los resultados fueron:

| Clase    | Precision |   Recall |   F1 |
| -------- | --------: | -------: | ---: |
| Suspenso |      0,20 | **0,65** | 0,31 |
| Aprobado |      0,95 |     0,72 | 0,82 |

Las métricas globales son:

```text
Accuracy = 0,71
Balanced Accuracy = 0,683
```

El recall de los suspensos pasa del:

```text
10 % → 65 %
```

Por tanto, el modelo balanceado reduce la accuracy global, pero mejora considerablemente su capacidad para detectar estudiantes en riesgo de suspender.

---

## Variables más relevantes en clasificación

La regresión logística identifica nuevamente como principales predictores:

| Variable                 | Coeficiente aproximado |
| ------------------------ | ---------------------: |
| Horas de estudio semanal |                  +2,73 |
| Nota anterior            |                  +2,12 |
| Tasa de asistencia       |                  +1,76 |
| Nivel de dificultad      |                  −0,77 |

Existe, por tanto, una elevada coherencia entre los modelos de regresión y clasificación.

---

# Conclusiones

Los resultados obtenidos permiten concluir que las variables con mayor capacidad predictiva del rendimiento académico son:

1. **Horas semanales de estudio.**
2. **Nota anterior.**
3. **Tasa de asistencia.**

En regresión, el modelo consigue explicar aproximadamente un **36-37 % de la variabilidad de la nota final**, con un error absoluto medio próximo a **5,8 puntos**.

Los modelos Linear, Ridge y Lasso presentan resultados muy similares, por lo que la regularización no aporta una mejora sustancial.

En clasificación, la regresión logística L2 obtiene una accuracy próxima al **91 %**, aunque esta cifra está condicionada por el fuerte predominio de los estudiantes aprobados.

El análisis de las clases por separado demuestra que el modelo convencional presenta dificultades para identificar suspensos.

La utilización de `class_weight='balanced'` permite aumentar el recall de los suspensos desde aproximadamente el **10 % hasta el 65 %**, incrementando además la Balanced Accuracy de **0,55 a 0,683**.

Por ello, la selección del modelo debe depender del objetivo:

* Si se busca maximizar el número total de predicciones correctas, el modelo L2 convencional ofrece mejores resultados.
* Si el objetivo es detectar estudiantes potencialmente en riesgo de suspender, resulta más informativo utilizar el modelo balanceado.

El proyecto muestra así la importancia de no evaluar los modelos únicamente mediante una métrica, especialmente cuando existen problemas de **clases desbalanceadas**.

---

## Tecnologías utilizadas

El proyecto ha sido desarrollado en **Python** utilizando principalmente:

* `NumPy`
* `Pandas`
* `Matplotlib`
* `Seaborn`
* `SciPy`
* `Scikit-learn`
* `category_encoders`
* `Joblib`

---

## Instalación

Para instalar las principales dependencias:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn category-encoders joblib jupyter
```

---

## Ejecución

1. Clonar o descargar el repositorio.
2. Instalar las dependencias.
3. Añadir el dataset al directorio del proyecto.
4. Abrir el notebook de Jupyter.
5. Ejecutar las celdas en orden.

```bash
jupyter notebook
```

### Importante

En el notebook original la carga del dataset utiliza una ruta local de Windows:

```python
df = pd.read_csv(r"C:\Users\pchic\Downloads\dataset_estudiantes (2).csv")
```

Para que el proyecto pueda ejecutarse desde cualquier ordenador y funcionar correctamente en GitHub, se recomienda sustituirla por una ruta relativa, por ejemplo:

```python
df = pd.read_csv("dataset_estudiantes.csv")
```

---

## Archivos generados

Durante el proyecto se generan distintos archivos:

```text
df_regresion.csv
modelo_regresion.pkl
modelo_clasificacion.pkl
scaler_clasificacion.pkl
```

`modelo_regresion.pkl` contiene el modelo final de regresión.

`modelo_clasificacion.pkl` contiene el modelo final de clasificación mediante regresión logística L2.

`scaler_clasificacion.pkl` almacena el escalador utilizado para las variables numéricas del modelo de clasificación.

---

## Estructura recomendada del repositorio

```text
PROYECTO-ML/
│
├── PROYECTO OPCIONAL ML.ipynb
├── dataset_estudiantes.csv
├── README.md
│
├── modelo_regresion.pkl
├── modelo_clasificacion.pkl
├── scaler_clasificacion.pkl
│
└── df_regresion.csv
```

---

## Autor

Proyecto realizado como ejercicio práctico de **Machine Learning**, incluyendo análisis exploratorio, preprocesamiento, regresión, clasificación, regularización, interpretación de modelos y tratamiento de clases desbalanceadas.
