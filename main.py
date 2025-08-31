import pandas as pd
import numpy as np

data = pd.read_csv('day.csv')
print("Primeras filas:")
print(data.head())

def one_hot(df, column):
    uniques = sorted(df[column].unique())
    for val in uniques[1:]:  # drop_first=True equivalente
        df[f"{column}_{val}"] = (df[column] == val).astype(float)
    df.drop(column, axis=1, inplace=True)
    return df

for col in ['season', 'yr', 'mnth', 'weekday', 'weathersit']:
    data = one_hot(data, col)

for col in ['temp', 'atemp', 'hum', 'windspeed']:
    min_val = data[col].min()
    max_val = data[col].max()
    data[col] = (data[col] - min_val) / (max_val - min_val) 

# Eliminamos columnas no numéricas o no predictoras
X = data.drop(['instant','dteday','casual','registered','cnt'], axis=1).values.astype(np.float64)
y = data['cnt'].values.reshape(-1,1).astype(np.float64)

# Agregar columna de 1 para bias
X = np.hstack((np.ones((X.shape[0],1)), X))

np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

train_size = int(0.8 * X.shape[0])
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

w = np.random.randn(X.shape[1],1) * 0.01
learning_rate = 0.01
epochs = 1000
m = X_train.shape[0]

for i in range(epochs):
    y_pred = X_train.dot(w)
    error = y_pred - y_train
    cost = (1/m) * np.sum(error**2)
    gradient = (2/m) * X_train.T.dot(error)
    w -= learning_rate * gradient

    if i % 100 == 0:
        print(f"Epoch {i}, Cost: {cost:.2f}")

y_pred_test = X_test.dot(w)

# Mostrar primeras 10 predicciones vs reales
print("\nPrimeras 10 predicciones vs reales:")
for real, pred in zip(y_test[:10], y_pred_test[:10]):
    print(f"Real: {real[0]:.0f}, Predicción: {pred[0]:.2f}")

# Cálculo de R² en test
R2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2)
print(f"\nR² en test: {R2:.4f}")

