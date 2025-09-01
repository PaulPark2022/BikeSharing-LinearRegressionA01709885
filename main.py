import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Carga de datos
# ==========================================================
data = pd.read_csv('data/day.csv')  # Ajusta ruta según tu repo

print("Primeras filas del dataset:")
print(data.head())

# ==========================================================
# 2. Preprocesamiento
# ==========================================================

def one_hot(df, column):
    """Codificación one-hot manual para variables categóricas."""
    uniques = sorted(df[column].unique())
    for val in uniques[1:]:  # drop_first=True
        df[f"{column}_{val}"] = (df[column] == val).astype(float)
    df.drop(column, axis=1, inplace=True)
    return df

# Variables categóricas
for col in ['season', 'yr', 'mnth', 'weekday', 'weathersit']:
    data = one_hot(data, col)

# Normalización manual
for col in ['temp', 'atemp', 'hum', 'windspeed']:
    min_val = data[col].min()
    max_val = data[col].max()
    data[col] = (data[col] - min_val) / (max_val - min_val)

# ==========================================================
# 3. Separación de datos
# ==========================================================
X = data.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1).values.astype(np.float64)
y = data['cnt'].values.reshape(-1, 1).astype(np.float64)

# Agregar bias
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Train/test split
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

train_size = int(0.8 * X.shape[0])
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# ==========================================================
# 4. Algoritmo: Regresión Lineal con Gradiente Descendente
# ==========================================================
def hipotesis(X, w):
    return X.dot(w)

def costo(X, y, w):
    m = len(y)
    error = hipotesis(X, w) - y
    return (1/m) * np.sum(error**2)

def gradiente(X, y, w):
    m = len(y)
    error = hipotesis(X, w) - y
    return (2/m) * X.T.dot(error)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Inicialización
w = np.random.randn(X.shape[1], 1) * 0.01
learning_rate = 0.01
epochs = 1000

train_costs = []

for i in range(epochs):
    w -= learning_rate * gradiente(X_train, y_train, w)
    if i % 100 == 0:
        c = costo(X_train, y_train, w)
        train_costs.append(c)
        print(f"Epoch {i}, Cost: {c:.2f}")

# ==========================================================
# 5. Evaluación
# ==========================================================
y_pred_test = hipotesis(X_test, w)
R2 = r2_score(y_test, y_pred_test)

print("\nPrimeras 10 predicciones vs reales:")
for real, pred in zip(y_test[:10], y_pred_test[:10]):
    print(f"Real: {real[0]:.0f}, Predicción: {pred[0]:.2f}")

print(f"\nR² en test: {R2:.4f}")

# ==========================================================
# 6. Gráficas
# ==========================================================
plt.figure(figsize=(6,4))
plt.plot(train_costs, marker='o')
plt.title("Evolución del costo (cada 100 épocas)")
plt.xlabel("Iteraciones (x100)")
plt.ylabel("Costo MSE")
plt.savefig("results/training_curve.png")
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.title("Valores reales vs predicciones")
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.savefig("results/predictions.png")
plt.close()
