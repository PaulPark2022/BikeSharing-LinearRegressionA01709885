# Bike Sharing Prediction - Regresión Lineal Manual

## Descripción
Este proyecto implementa una regresión lineal **desde cero**, sin frameworks de Machine Learning, para predecir la cantidad total de bicicletas rentadas diariamente en el sistema Capital Bikeshare de Washington D.C. Se trabajó con el dataset [Bike Sharing Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset).

El objetivo es comprender los fundamentos de aprendizaje automático, ajuste de parámetros y optimización mediante gradiente descendente.

---

## Dataset
- **Archivo:** `data/day.csv`
- **Registros:** 731 días (2011–2012)
- **Variables seleccionadas:**
  - `season`, `yr`, `mnth`, `weekday`, `weathersit` → codificadas con one-hot
  - `temp`, `atemp`, `hum`, `windspeed` → normalizadas
  - `cnt` → objetivo
- **Variables descartadas:** `casual`, `registered`, `instant`, `dteday`

---

## Preprocesamiento
1. One-hot encoding manual para variables categóricas (drop first para evitar multicolinealidad).
2. Normalización manual de variables continuas.
3. Separación en conjuntos de entrenamiento (80%) y prueba (20%).
4. Adición de término de bias a las entradas.

---

## Implementación
El código principal se encuentra en `main.py` y contiene:
- `hipotesis(X, w)` → calcula predicciones
- `costo(X, y, w)` → error cuadrático medio (MSE)
- `gradiente(X, y, w)` → gradiente para actualización de pesos
- `r2_score(y_true, y_pred)` → coeficiente de determinación R²

**Optimización:** Gradiente descendente con 1000 épocas y learning rate = 0.01.

---

## Resultados
- R² en conjunto de prueba: **0.8470**
- Curva de costo y predicciones vs reales generadas en `results/`:
  - `training_curve.png`
  - `predictions.png`

---

## Ejecución
1. Clonar el repositorio:
git clone https://github.com/PaulPark2022/BikeSharing-LinearRegressionA01709885.git

2. Instalar dependencias:
pip install numpy pandas matplotlib

3. Ejecutar:
python main.py

## Referencias
UC Irvine Machine Learning Repository – Bike Sharing Dataset. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset 
