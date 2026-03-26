import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CARGA DE DATOS ---
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. PREPROCESAMIENTO (EL SCALER) ---
# Aquí es donde se crea el objeto que viste en el texto de GitHub
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler como un archivo real por si lo necesitas entregar aparte
joblib.dump(scaler, 'scaler.pkl')

# --- 3. IMPORTACIÓN DE MODELOS ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# --- 4. ENTRENAMIENTO DE LOS 8 MODELOS ---
modelos = {
    "Ordinary Least Squares": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Bayesian Regression": BayesianRidge(),
    "Lasso Regression": Lasso(alpha=0.01),
    "Nearest Neighbors Regression": KNeighborsRegressor(n_neighbors=5),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM Regression": SVR(kernel='rbf'),
    "Neural Network MLP Regression": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# --- 5. EVALUACIÓN Y RESULTADOS ---
print(f"{'Modelo':<35} | {'MSE':<10} | {'R2 Score':<10}")
print("-" * 60)

for nombre, modelo in modelos.items():
    # Entrenar
    modelo.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{nombre:<35} | {mse:.4f}     | {r2:.4f}")

