"""
=============================================================================
  SISTEMA DE PREDICCIÓN DE PRECIOS - INTEL (INTC)
  Proyecto: Aprendizaje Automático - Regresión Lineal
=============================================================================

  Descripción:
    Script de predicción que carga un modelo entrenado y genera estimaciones
    del precio futuro del ticker INTC a partir de datos históricos en CSV.

  Modelos soportados:
    OLS, Ridge, Bayesian, Lasso, KNN, Random Forest, SVM, MLP

  Uso básico:
    python prediccion_intc.py --csv datos.csv

  Uso completo:
    python prediccion_intc.py --csv datos.csv --modelo models/lasso.pkl
                              --scaler models/scaler.pkl --salida resultado.csv
                              --mostrar-metricas

  El CSV de entrada puede tener:
    • Formato básico  (7 cols):  open, high, low, close, volume, trade_count, vwap
    • Formato completo (12 cols): las 7 anteriores + daily_return, ma_7, ma_21,
                                  volatility_7, daily_range
  La columna de fecha puede llamarse "fecha" o "timestamp".
=============================================================================
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO
# ─────────────────────────────────────────────────────────────────────────────

RUTA_CSV_DEFAULT    = "data/intc_processed.csv"
RUTA_SCALER_DEFAULT = "models/scaler.pkl"
RUTA_MODELO_DEFAULT = "models/lasso.pkl"
RUTA_SALIDA_DEFAULT = "predicciones.csv"

# Columnas que espera el modelo (en orden)
COLUMNAS_BASICAS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
COLUMNAS_DERIVADAS = ["daily_return", "ma_7", "ma_21", "volatility_7", "daily_range"]
COLUMNAS_MODELO = COLUMNAS_BASICAS + COLUMNAS_DERIVADAS   # 12 features en total

# Nombre de la columna objetivo (por si el CSV la trae para calcular métricas)
COLUMNA_OBJETIVO = "target"

# Nombre que puede tener la columna de fecha en el CSV
POSIBLES_COLS_FECHA = ["fecha", "timestamp"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    linea = "=" * 60
    print(f"\n{linea}")
    print("   SISTEMA DE PREDICCIÓN  —  INTEL (INTC)")
    print(f"{linea}\n")


def ok(msg):    print(f"  [OK]    {msg}")
def info(msg):  print(f"  [INFO]  {msg}")
def warn(msg):  print(f"  [WARN]  {msg}")
def error(msg): print(f"  [ERROR] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELO Y SCALER
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo(ruta_scaler: str, ruta_modelo: str):
    """
    Carga el scaler y el modelo entrenado desde archivos pickle.

    Parámetros
    ----------
    ruta_scaler : str  Ruta al archivo scaler.pkl
    ruta_modelo : str  Ruta al archivo del modelo (.pkl)

    Retorna
    -------
    (scaler, modelo) : tupla con los objetos cargados
    """
    for ruta, nombre in [(ruta_scaler, "scaler"), (ruta_modelo, "modelo")]:
        if not os.path.exists(ruta):
            error(f"No se encontró el {nombre} en: {ruta}")
            sys.exit(1)

    try:
        scaler = joblib.load(ruta_scaler)
        modelo = joblib.load(ruta_modelo)
        ok(f"Scaler cargado  →  {ruta_scaler}")
        ok(f"Modelo cargado  →  {ruta_modelo}  [{type(modelo).__name__}]")
        return scaler, modelo
    except Exception as exc:
        error(f"Error al cargar los archivos pickle: {exc}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def aplicar_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las columnas derivadas usadas durante el entrenamiento.

    Columnas generadas
    ------------------
    • daily_return  : retorno porcentual diario del precio de cierre
    • ma_7          : media móvil de 7 días del precio de cierre
    • ma_21         : media móvil de 21 días del precio de cierre
    • volatility_7  : desviación estándar móvil de 7 días del retorno diario
    • daily_range   : diferencia entre el precio máximo y mínimo del día

    Nota: las primeras 21 filas quedarán con NaN por los cálculos rolling
    y serán descartadas antes de escalar.
    """
    df = df.copy()
    df["daily_return"]  = df["close"].pct_change()
    df["ma_7"]          = df["close"].rolling(window=7).mean()
    df["ma_21"]         = df["close"].rolling(window=21).mean()
    df["volatility_7"]  = df["daily_return"].rolling(window=7).std()
    df["daily_range"]   = df["high"] - df["low"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPROCESAMIENTO DEL CSV
# ─────────────────────────────────────────────────────────────────────────────

def _detectar_columna_fecha(df: pd.DataFrame) -> str | None:
    """Devuelve el nombre de la columna de fecha si existe, o None."""
    for col in POSIBLES_COLS_FECHA:
        if col in df.columns:
            return col
    return None


def procesar_csv(ruta_csv: str, scaler) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Lee el CSV, aplica feature engineering si hace falta, elimina NaN
    y escala las características.

    Parámetros
    ----------
    ruta_csv : str   Ruta al archivo CSV de entrada
    scaler   : obj   Scaler entrenado (StandardScaler u otro)

    Retorna
    -------
    (X_scaled_df, y_real)
        X_scaled_df : DataFrame con las 12 features escaladas, indexado por fecha
        y_real      : Series con los valores reales de "target" si existen, o None
    """
    # ── Verificar existencia ──────────────────────────────────────────────────
    if not os.path.exists(ruta_csv):
        error(f"No se encontró el archivo: {ruta_csv}")
        sys.exit(1)

    # ── Leer CSV ──────────────────────────────────────────────────────────────
    try:
        col_fecha = None
        # Lectura provisional para detectar la columna de fecha
        temp = pd.read_csv(ruta_csv, nrows=0)
        col_fecha = _detectar_columna_fecha(temp)

        if col_fecha:
            df = pd.read_csv(ruta_csv, parse_dates=[col_fecha])
            df = df.set_index(col_fecha)
            df.index.name = "fecha"
        else:
            df = pd.read_csv(ruta_csv)
            warn("No se encontró columna de fecha ('fecha' / 'timestamp'). "
                 "El índice será numérico.")

        ok(f"CSV cargado: {len(df)} registros  |  columnas: {list(df.columns)}")
        if col_fecha:
            info(f"Rango de fechas: {df.index.min()}  →  {df.index.max()}")

    except Exception as exc:
        error(f"No se pudo leer el CSV: {exc}")
        sys.exit(1)

    # ── Extraer y_real si el CSV ya incluye la columna objetivo ──────────────
    y_real = None
    if COLUMNA_OBJETIVO in df.columns:
        y_real = df[COLUMNA_OBJETIVO].copy()
        info(f"Columna '{COLUMNA_OBJETIVO}' detectada. "
             "Se calcularán métricas de evaluación.")

    # ── Detectar formato y aplicar feature engineering si hace falta ─────────
    tiene_basicas  = all(c in df.columns for c in COLUMNAS_BASICAS)
    tiene_completo = all(c in df.columns for c in COLUMNAS_MODELO)

    if tiene_completo:
        info("CSV con las 12 columnas pre-procesadas. Usando directamente.")
        X = df[COLUMNAS_MODELO].copy()

    elif tiene_basicas:
        info("CSV con 7 columnas básicas. Aplicando feature engineering...")
        df = aplicar_feature_engineering(df)
        X = df[COLUMNAS_MODELO].copy()
        antes = len(X)
        X = X.dropna()
        if y_real is not None:
            y_real = y_real.loc[X.index]   # alinear target con filas válidas
        info(f"Filas eliminadas por NaN en rolling: {antes - len(X)}")

    else:
        faltantes = [c for c in COLUMNAS_BASICAS if c not in df.columns]
        error("El CSV no contiene las columnas mínimas requeridas.")
        error(f"Columnas faltantes: {faltantes}")
        info("Columnas mínimas necesarias: " + ", ".join(["fecha"] + COLUMNAS_BASICAS))
        sys.exit(1)

    # ── Validar que quedaron registros ────────────────────────────────────────
    if len(X) == 0:
        error("No quedaron registros válidos tras el preprocesamiento.")
        info("Se necesitan al menos 21 registros consecutivos para calcular "
             "las medias móviles.")
        sys.exit(1)

    # ── Escalar ───────────────────────────────────────────────────────────────
    try:
        X_scaled = scaler.transform(X)
        ok(f"Escalado aplicado  →  {X_scaled.shape[0]} registros listos.")
    except Exception as exc:
        error(f"Error al escalar los datos: {exc}")
        info("Verifica que el scaler fue entrenado con las mismas 12 columnas.")
        sys.exit(1)

    X_scaled_df = pd.DataFrame(X_scaled, columns=COLUMNAS_MODELO, index=X.index)
    return X_scaled_df, y_real


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calcula MAE, MSE, RMSE y R² comparando valores reales vs. predichos.

    Parámetros
    ----------
    y_real : Series  Valores reales del precio objetivo
    y_pred : array   Predicciones del modelo

    Retorna
    -------
    dict con las cuatro métricas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_real, y_pred)
    mse  = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_real, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


def imprimir_metricas(metricas: dict, nombre_modelo: str):
    """Imprime las métricas en formato tabla."""
    sep = "-" * 40
    print(f"\n{sep}")
    print(f"  Métricas de evaluación — {nombre_modelo}")
    print(sep)
    for nombre, valor in metricas.items():
        print(f"  {nombre:<6} :  {valor:.6f}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN Y GUARDADO
# ─────────────────────────────────────────────────────────────────────────────

def predecir_y_guardar(
    ruta_csv:       str,
    scaler,
    modelo,
    ruta_salida:    str,
    mostrar_metricas: bool = False,
) -> pd.DataFrame:
    """
    Flujo completo: lee CSV → feature engineering → escala → predice → guarda.

    Parámetros
    ----------
    ruta_csv         : Ruta al archivo CSV de entrada
    scaler           : Scaler cargado
    modelo           : Modelo cargado
    ruta_salida      : Ruta del CSV de salida con predicciones
    mostrar_metricas : Si True y el CSV contiene la columna 'target', calcula
                       y muestra MAE, MSE, RMSE, R²

    Retorna
    -------
    DataFrame con columnas [fecha, prediccion] y opcionalmente [target, error_abs]
    """
    # ── Procesar CSV ──────────────────────────────────────────────────────────
    X_scaled, y_real = procesar_csv(ruta_csv, scaler)

    # ── Predecir ──────────────────────────────────────────────────────────────
    try:
        y_pred = modelo.predict(X_scaled)
        ok(f"Predicción completada para {len(y_pred)} registros.")
    except Exception as exc:
        error(f"Error durante la predicción: {exc}")
        sys.exit(1)

    # ── Construir DataFrame de resultados ─────────────────────────────────────
    resultado = pd.DataFrame(
        {"prediccion": np.round(y_pred, 4)},
        index=X_scaled.index
    )
    resultado.index.name = "fecha"

    if y_real is not None and mostrar_metricas:
        y_real_aligned = y_real.loc[resultado.index]
        resultado["target"]     = y_real_aligned.values
        resultado["error_abs"]  = np.abs(y_real_aligned.values - y_pred).round(4)

        metricas = calcular_metricas(y_real_aligned, y_pred)
        imprimir_metricas(metricas, type(modelo).__name__)

    elif y_real is not None and not mostrar_metricas:
        info("Columna 'target' disponible. Usa --mostrar-metricas para evaluar "
             "el modelo contra los valores reales.")

    # ── Guardar CSV de salida ─────────────────────────────────────────────────
    try:
        resultado.reset_index().to_csv(ruta_salida, index=False)
        ok(f"Resultados guardados en: {ruta_salida}")
    except Exception as exc:
        error(f"No se pudo guardar el archivo de salida: {exc}")
        sys.exit(1)

    # ── Mostrar muestra de predicciones ──────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Muestra de predicciones (primeras 5 filas):")
    print(f"{'─'*55}")
    print(resultado.head(5).to_string())
    print(f"{'─'*55}\n")

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# LISTADO DE MODELOS DISPONIBLES
# ─────────────────────────────────────────────────────────────────────────────

MODELOS_CONOCIDOS = {
    "ols":          "Ordinary Least Squares Regression",
    "ridge":        "Ridge Regression",
    "bayesian":     "Bayesian Ridge Regression",
    "lasso":        "Lasso Regression",
    "knn":          "K-Nearest Neighbors Regression",
    "random_forest":"Random Forest Regression",
    "svm":          "Support Vector Machine Regression",
    "mlp":          "Neural Network MLP Regression",
}

def listar_modelos(directorio: str = "models"):
    """Imprime los modelos .pkl disponibles en el directorio indicado."""
    print(f"\nModelos disponibles en '{directorio}/':\n")
    if not os.path.isdir(directorio):
        warn(f"El directorio '{directorio}' no existe.")
        return
    for nombre, descripcion in MODELOS_CONOCIDOS.items():
        ruta = os.path.join(directorio, f"{nombre}.pkl")
        estado = "✓" if os.path.exists(ruta) else "✗"
        print(f"  [{estado}]  {nombre}.pkl  —  {descripcion}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="prediccion_intc.py",
        description="Sistema de predicción del precio de Intel (INTC) usando ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Predicción con el modelo por defecto (Lasso)
  python prediccion_intc.py --csv data/intc_processed.csv

  # Usar Random Forest y ver métricas
  python prediccion_intc.py --csv data/intc_processed.csv \\
      --modelo models/random_forest.pkl --mostrar-metricas

  # Usar MLP con salida personalizada
  python prediccion_intc.py --csv nuevos_datos.csv \\
      --modelo models/mlp.pkl --salida preds_mlp.csv

  # Ver modelos disponibles
  python prediccion_intc.py --listar-modelos
        """,
    )

    parser.add_argument(
        "--csv",
        default=RUTA_CSV_DEFAULT,
        metavar="RUTA",
        help=f"CSV de entrada con datos históricos (default: {RUTA_CSV_DEFAULT})",
    )
    parser.add_argument(
        "--scaler",
        default=RUTA_SCALER_DEFAULT,
        metavar="RUTA",
        help=f"Scaler entrenado .pkl (default: {RUTA_SCALER_DEFAULT})",
    )
    parser.add_argument(
        "--modelo",
        default=RUTA_MODELO_DEFAULT,
        metavar="RUTA",
        help=f"Modelo entrenado .pkl (default: {RUTA_MODELO_DEFAULT}). "
             "Opciones: ols, ridge, bayesian, lasso, knn, random_forest, svm, mlp",
    )
    parser.add_argument(
        "--salida",
        default=RUTA_SALIDA_DEFAULT,
        metavar="RUTA",
        help=f"CSV de salida con predicciones (default: {RUTA_SALIDA_DEFAULT})",
    )
    parser.add_argument(
        "--mostrar-metricas",
        action="store_true",
        help="Calcula MAE, MSE, RMSE y R² si el CSV incluye la columna 'target'",
    )
    parser.add_argument(
        "--listar-modelos",
        action="store_true",
        help="Muestra los modelos disponibles en la carpeta 'models/' y sale",
    )

    args = parser.parse_args()

    banner()

    # Listar modelos y salir si se pidió
    if args.listar_modelos:
        directorio = os.path.dirname(args.modelo) or "models"
        listar_modelos(directorio)
        sys.exit(0)

    # Cargar scaler y modelo
    scaler, modelo = cargar_modelo(args.scaler, args.modelo)

    # Ejecutar predicción
    predecir_y_guardar(
        ruta_csv=args.csv,
        scaler=scaler,
        modelo=modelo,
        ruta_salida=args.salida,
        mostrar_metricas=args.mostrar_metricas,
    )

    print("Proceso finalizado correctamente.\n")


if __name__ == "__main__":
    main()
