# -*- coding: utf-8 -*-
"""
Simulador Completo (con datos sintéticos) de Redes Híbridas de Energía Renovable
para Centros Educativos Rurales
"""

# %% 1. Importar Librerías
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog  # Para Programación Lineal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime, timedelta

# %% 2. Generación de Datos Sintéticos (¡¡REEMPLAZAR CON DATOS REALES!!)


def generar_datos_sinteticos(filepath, dias=365, institucion_id="sintetico"):
    """Genera un archivo CSV con datos sintéticos horarios."""
    print(f"Generando datos sintéticos para {institucion_id} en {filepath}...")
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(dias * 24)]
    df = pd.DataFrame({"timestamp": timestamps})
    df.set_index("timestamp", inplace=True)

    # Irradiancia (patrón diurno sinusoidal + ruido)
    hours = df.index.hour
    df["irradiancia"] = np.maximum(
        0, np.sin((hours - 6) * np.pi / 12) * 1000 + np.random.normal(0, 50, len(df))
    )
    df.loc[(hours < 6) | (hours > 18), "irradiancia"] = 0  # Noche

    # Temperatura (patrón diario simple + estacionalidad + ruido)
    day_of_year = df.index.dayofyear
    df["temp"] = (
        15
        + 5 * np.sin((hours - 9) * np.pi / 12)
        + 5 * np.sin((day_of_year - 90) * 2 * np.pi / 365)
        + np.random.normal(0, 1, len(df))
    )

    # Consumo (patrón escolar: alto durante el día laboral, bajo noches/fines de semana + ruido)
    base_consumo = 2.0  # kW
    consumo = base_consumo * np.ones(len(df))
    es_dia_laboral = df.index.dayofweek < 5  # Lunes a Viernes
    es_horario_escolar = (hours >= 8) & (hours < 16)
    consumo[es_dia_laboral & es_horario_escolar] *= 2.0 + np.random.uniform(
        -0.5, 0.5, sum(es_dia_laboral & es_horario_escolar)
    )  # Mayor consumo en horario escolar
    consumo[~es_dia_laboral] *= 0.5 + np.random.uniform(
        -0.1, 0.1, sum(~es_dia_laboral)
    )  # Menor consumo fines de semana
    consumo[(hours < 7) | (hours > 18)] *= 0.3 + np.random.uniform(
        -0.05, 0.05, sum((hours < 7) | (hours > 18))
    )  # Muy bajo de noche
    df["consumo_kw"] = np.maximum(
        0.1, consumo + np.random.normal(0, 0.2, len(df))
    )  # Asegurar consumo mínimo

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"Datos sintéticos guardados en {filepath}")
    return filepath


# %% 3. Definición de Parámetros y Configuraciones por Institución

# ¡¡AJUSTAR ESTOS PARÁMETROS A LA REALIDAD DE CADA INSTITUCIÓN!!
CONFIGURACIONES = {
    "centro_1": {
        "data_path": "data/centro_1_sintetico.csv",
        "params": {
            "panel_area": 60,
            "panel_efficiency": 0.19,
            "temp_coeff": 0.004,
            "battery_capacity_ah": 250,
            "battery_voltage_nom": 48,
            "battery_efficiency": 0.92,
            "soc_min": 0.20,
            "soc_max": 0.95,
            "initial_soc": 0.5,
            "inverter_efficiency": 0.95,
            "simulation_step_h": 1,
            "grid_cost_per_kwh": 0.10,  # Costo de comprar energía de la red (ejemplo)
        },
    },
    "centro_2": {
        "data_path": "data/centro_2_sintetico.csv",
        "params": {
            "panel_area": 40,
            "panel_efficiency": 0.18,
            "temp_coeff": 0.0045,
            "battery_capacity_ah": 150,
            "battery_voltage_nom": 48,
            "battery_efficiency": 0.90,
            "soc_min": 0.25,
            "soc_max": 0.90,
            "initial_soc": 0.6,
            "inverter_efficiency": 0.94,
            "simulation_step_h": 1,
            "grid_cost_per_kwh": 0.11,
        },
    },
    # Añadir aquí las configuraciones para centro_3, centro_4, centro_5
    # ...
}

# Generar datos sintéticos si no existen
for inst_id, config in CONFIGURACIONES.items():
    if not os.path.exists(config["data_path"]):
        generar_datos_sinteticos(config["data_path"], institucion_id=inst_id)

# %% 4. Funciones Auxiliares (Carga de Datos, Cálculo PV)


def load_data(filepath):
    """Carga y preprocesa datos de una institución."""
    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
        df = df.interpolate(method="time")  # Interpolar valores faltantes
        df = df.fillna(method="bfill").fillna(
            method="ffill"
        )  # Rellenar al inicio/final si es necesario

        # Asegurar columnas necesarias
        for col in ["irradiancia", "temp", "consumo_kw"]:
            if col not in df.columns:
                raise ValueError(
                    f"Columna requerida '{col}' no encontrada en {filepath}"
                )

        # Crear características temporales
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["dayofyear"] = df.index.dayofyear

        print(f"Datos cargados y preprocesados para {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error Crítico: Archivo no encontrado {filepath}. Deteniendo.")
        return None
    except ValueError as e:
        print(f"Error Crítico: {e}. Deteniendo.")
        return None


def calcular_potencia_pv(irradiancia, temp, params):
    """Calcula la potencia PV instantánea."""
    # Modelo simple con eficiencia y efecto de temperatura
    # Podría mejorarse con modelos más complejos (ej. Sandia, PVWatts)
    t_ref = 25.0  # Temperatura de referencia
    p_nominal = (
        params["panel_area"] * 1000 * params["panel_efficiency"]
    )  # Potencia nominal STC (aprox)
    p_pv = (
        params["panel_area"]
        * irradiancia
        * params["panel_efficiency"]
        * (1 - params["temp_coeff"] * np.maximum(0, temp - t_ref))
    )
    return np.maximum(0, p_pv / 1000)  # Convertir a kW y asegurar no negativo


# %% 5. Modelo Matemático (Ecuaciones Diferenciales)


def hybrid_system_ode(t, y, T, P_pv, P_load, params):
    """
    Define las EDOs del sistema híbrido para UN paso de tiempo dt.
    y: vector de estado [SOC]
    t: tiempo actual (no se usa directamente si los inputs son por paso)
    T: Temperatura actual
    P_pv: Potencia PV actual (kW)
    P_load: Potencia de Carga actual (kW)
    params: diccionario de parámetros del sistema
    """
    SOC = y[0]
    dt = params["simulation_step_h"]  # Paso de tiempo en horas

    # Potencia disponible de PV después de eficiencia del inversor (si aplica a carga DC/AC)
    P_pv_util = (
        P_pv  # Simplificación: Asumimos que la PV puede ir directo a la carga o batería
    )

    # Potencia neta (Generación - Carga)
    P_net = P_pv_util - P_load

    # Capacidad de la batería en kWh
    Cap_kwh = params["battery_capacity_ah"] * params["battery_voltage_nom"] / 1000

    P_bat = 0  # Potencia hacia (+) o desde (-) la batería en kW
    P_grid = 0  # Potencia desde la red (+)

    if P_net >= 0:  # Exceso de generación
        # Intentar cargar la batería
        P_charge_max_kwh = (params["soc_max"] - SOC) * Cap_kwh
        P_charge_max_kw = P_charge_max_kwh / dt  # Potencia máx para cargar en dt horas
        P_to_charge = min(P_net, P_charge_max_kw)

        P_bat = (
            -P_to_charge * params["battery_efficiency"]
        )  # Potencia que entra a la batería (negativa), considerando pérdidas
        # Podría haber excedente para vender a la red si P_net > P_to_charge (no modelado aquí)

    else:  # Déficit de generación (P_net < 0)
        P_needed = abs(P_net)
        # Intentar descargar la batería
        P_discharge_max_kwh = (SOC - params["soc_min"]) * Cap_kwh
        P_discharge_max_kw = P_discharge_max_kwh / dt
        P_from_discharge = min(P_needed, P_discharge_max_kw)

        if P_from_discharge > 0:
            P_bat = (
                P_from_discharge / params["battery_efficiency"]
            )  # Potencia que sale de la batería (positiva), considerando pérdidas
            P_needed -= P_from_discharge  # Potencia que aún falta

        # Si aún falta potencia, obtener de la red
        if P_needed > 0:
            P_grid = P_needed  # Asume que la red puede suplir todo lo necesario

    # Cambio en SOC (derivada discreta)
    dSOC = (P_bat * dt) / Cap_kwh  # P_bat es negativo si carga, positivo si descarga

    # Aplicar límites estrictos de SOC
    next_soc = SOC + dSOC
    if next_soc < params["soc_min"]:
        # Ajustar P_bat para que SOC quede en soc_min
        dSOC = params["soc_min"] - SOC
        P_bat = dSOC * Cap_kwh / dt  # Recalcular P_bat correspondiente
        # Recalcular P_grid si es necesario (si la reducción de descarga aumentó P_needed)
        # Esta parte puede volverse compleja dependiendo de la lógica exacta
    elif next_soc > params["soc_max"]:
        dSOC = params["soc_max"] - SOC
        P_bat = dSOC * Cap_kwh / dt  # P_bat será negativo

    # Devolvemos el cambio en SOC y la energía de red usada en este paso
    # solve_ivp espera la derivada, pero aquí trabajamos en pasos discretos dt
    # Para usar solve_ivp directamente, necesitaríamos interpolar P_pv, P_load
    # Enfoque más simple: simular paso a paso
    # Esta función devuelve el cambio en SOC y P_grid para el paso actual
    return dSOC, P_grid


# Función para simulación paso a paso (alternativa a solve_ivp para este caso)
def simulate_step_by_step(initial_soc, time_series_data, params):
    num_steps = len(time_series_data)
    soc = np.zeros(num_steps + 1)
    p_grid_usage = np.zeros(num_steps)
    p_bat_flow = np.zeros(num_steps)  # Positivo descarga, negativo carga
    p_pv_gen = np.zeros(num_steps)

    soc[0] = initial_soc

    for i in range(num_steps):
        # Obtener datos del paso actual
        current_data = time_series_data.iloc[i]
        temp = current_data["temp"]
        irradiancia = current_data["irradiancia"]
        p_load = current_data["consumo_kw"]

        # Calcular generación PV
        p_pv = calcular_potencia_pv(irradiancia, temp, params)
        p_pv_gen[i] = p_pv

        # Calcular cambio de SOC y uso de red
        # Usamos el SOC del *inicio* del paso para la decisión
        dSOC, p_grid = hybrid_system_ode(None, [soc[i]], temp, p_pv, p_load, params)

        # Actualizar SOC para el *final* del paso (inicio del siguiente)
        soc[i + 1] = soc[i] + dSOC
        # Almacenar resultados del paso
        p_grid_usage[i] = p_grid
        # Recalcular flujo de batería basado en dSOC final (para registro)
        Cap_kwh = params["battery_capacity_ah"] * params["battery_voltage_nom"] / 1000
        p_bat_flow[i] = dSOC * Cap_kwh / params["simulation_step_h"]

    results = pd.DataFrame(
        {
            "soc": soc[:-1],  # SOC al inicio de cada hora
            "p_grid_kw": p_grid_usage,
            "p_bat_kw": p_bat_flow,
            "p_pv_kw": p_pv_gen,
            "p_load_kw": time_series_data["consumo_kw"].values,
        },
        index=time_series_data.index,
    )

    return results


# %% 6. Machine Learning para Predicción


def train_predict_ml(df, features, target, institucion_id, save_path="modelos"):
    print(f"\n--- Entrenamiento/Predicción ML para {target} ({institucion_id}) ---")
    X = df[features]
    y = df[target]

    # Dividir datos (ordenados por tiempo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # Modelo (Random Forest como ejemplo robusto)
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
    )
    model.fit(X_train, y_train)

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluación
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print("Evaluación del Modelo:")
    print(f"  RMSE (Train/Test): {rmse_train:.4f} / {rmse_test:.4f}")
    print(f"  MAE  (Train/Test): {mae_train:.4f} / {mae_test:.4f}")
    print(f"  R²   (Train/Test): {r2_train:.4f} / {r2_test:.4f}")

    # Guardar modelo
    model_filename = os.path.join(save_path, f"{institucion_id}_{target}_model.pkl")
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {model_filename}")

    # Crear DataFrame con resultados de test para visualización/uso
    results_test = pd.DataFrame(
        {"real": y_test, "pred": y_pred_test}, index=y_test.index
    )

    return model, results_test, {"rmse": rmse_test, "mae": mae_test, "r2": r2_test}


# %% 7. Módulo de Optimización (Ejemplo Básico con Programación Lineal)


def optimize_operation_lp(soc_inicial, P_pv_pred, P_load_pred, params, horizonte_h=24):
    """
    Optimiza la operación para minimizar costos de red usando LP.
    Horizonte_h: Número de horas a optimizar.
    """
    print(f"\n--- Optimización LP para {horizonte_h} horas ---")

    # Parámetros clave
    dt = params["simulation_step_h"]
    Cap_kwh = params["battery_capacity_ah"] * params["battery_voltage_nom"] / 1000
    eta_bat = params["battery_efficiency"]
    eta_inv = params["inverter_efficiency"]  # Asumiendo conversión AC
    soc_min = params["soc_min"]
    soc_max = params["soc_max"]
    costo_grid = params["grid_cost_per_kwh"]

    # Asegurar que las predicciones tengan la longitud correcta
    P_pv_pred = P_pv_pred[:horizonte_h]
    P_load_pred = P_load_pred[:horizonte_h]
    T = len(P_load_pred)  # Horizonte efectivo

    # Variables de decisión para cada paso de tiempo t (0 a T-1):
    # p_bat_ch[t]: Potencia de carga de batería (kW) >= 0
    # p_bat_dis[t]: Potencia de descarga de batería (kW) >= 0
    # p_grid_buy[t]: Potencia comprada de la red (kW) >= 0
    # soc[t]: Estado de carga al *final* del paso t (fracción)

    num_vars_per_step = 3  # (p_bat_ch, p_bat_dis, p_grid_buy)
    num_soc_vars = T
    total_vars = T * num_vars_per_step + num_soc_vars

    # Función Objetivo: Minimizar costo total de energía de red
    # Costo = sum(p_grid_buy[t] * dt * costo_grid)
    c = np.zeros(total_vars)
    grid_indices = [t * num_vars_per_step + 2 for t in range(T)]
    c[grid_indices] = dt * costo_grid

    # Restricciones
    A_eq = []  # Restricciones de igualdad Ax = b
    b_eq = []
    A_ub = []  # Restricciones de desigualdad Ax <= b
    b_ub = []
    bounds = []  # Límites de las variables

    # Límites de variables (>= 0 por defecto para linprog, excepto SOC)
    for t in range(T):
        # p_bat_ch, p_bat_dis, p_grid_buy >= 0
        bounds.extend([(0, None), (0, None), (0, None)])
    # Límites SOC
    for t in range(T):
        bounds.append((soc_min, soc_max))  # soc_min <= soc[t] <= soc_max

    # Restricción 1: Balance de Potencia en cada paso t
    # P_pv(t) + p_bat_dis(t)/eta_inv + p_grid_buy(t) = P_load(t) + p_bat_ch(t)*eta_inv
    # -> p_bat_ch*eta_inv - p_bat_dis/eta_inv - p_grid_buy = P_pv(t) - P_load(t)
    for t in range(T):
        row = np.zeros(total_vars)
        row[t * num_vars_per_step + 0] = (
            eta_bat  # p_bat_ch (simplificado: eficiencia en carga)
        )
        row[t * num_vars_per_step + 1] = (
            -1 / eta_bat
        )  # p_bat_dis (simplificado: eficiencia en descarga)
        row[t * num_vars_per_step + 2] = -1  # p_grid_buy
        A_eq.append(row)
        b_eq.append(P_pv_pred[t] - P_load_pred[t])

    # Restricción 2: Evolución del SOC
    # soc[t] = soc[t-1] + (p_bat_ch[t]*eta_bat - p_bat_dis[t]/eta_bat) * dt / Cap_kwh
    # -> soc[t] - soc[t-1] - p_bat_ch*eta_bat*dt/Cap_kwh + p_bat_dis/eta_bat*dt/Cap_kwh = 0
    soc_start_index = T * num_vars_per_step
    for t in range(T):
        row = np.zeros(total_vars)
        row[soc_start_index + t] = 1  # soc[t]
        if t == 0:
            # soc[0] - soc_inicial - p_bat_ch[0]*... + p_bat_dis[0]*... = 0
            row[t * num_vars_per_step + 0] = -eta_bat * dt / Cap_kwh  # p_bat_ch[0]
            row[t * num_vars_per_step + 1] = (
                (1 / eta_bat) * dt / Cap_kwh
            )  # p_bat_dis[0]
            A_eq.append(row)
            b_eq.append(soc_inicial)
        else:
            # soc[t] - soc[t-1] - p_bat_ch[t]*... + p_bat_dis[t]*... = 0
            row[soc_start_index + t - 1] = -1  # soc[t-1]
            row[t * num_vars_per_step + 0] = -eta_bat * dt / Cap_kwh  # p_bat_ch[t]
            row[t * num_vars_per_step + 1] = (
                (1 / eta_bat) * dt / Cap_kwh
            )  # p_bat_dis[t]
            A_eq.append(row)
            b_eq.append(0)

    # (Opcional) Restricciones adicionales: límites de potencia de carga/descarga
    # p_bat_ch[t] <= P_ch_max
    # p_bat_dis[t] <= P_dis_max

    # Resolver el problema LP
    # Nota: method='highs' es generalmente recomendado si está disponible
    try:
        result = linprog(
            c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )  # o 'simplex', 'interior-point'

        if result.success:
            print("Optimización LP encontrada con éxito.")
            # Extraer resultados
            sol = result.x
            schedule = pd.DataFrame(index=P_load_pred.index[:T])
            schedule["p_bat_ch_kw"] = [sol[t * num_vars_per_step + 0] for t in range(T)]
            schedule["p_bat_dis_kw"] = [
                sol[t * num_vars_per_step + 1] for t in range(T)
            ]
            schedule["p_grid_buy_kw"] = [
                sol[t * num_vars_per_step + 2] for t in range(T)
            ]
            schedule["soc_opt"] = [sol[soc_start_index + t] for t in range(T)]
            schedule["p_pv_pred_kw"] = P_pv_pred[:T]
            schedule["p_load_pred_kw"] = P_load_pred[:T]
            return schedule, result.fun  # Devuelve el cronograma y el costo óptimo
        else:
            print(f"Error en la optimización LP: {result.message}")
            return None, None
    except ValueError as e:
        print(f"Error durante la configuración de linprog: {e}")
        # Esto puede ocurrir si las matrices/vectores no tienen dimensiones consistentes
        return None, None
    except Exception as e:
        print(f"Excepción inesperada durante la optimización: {e}")
        return None, None


# %% 8. Validación (Conceptual)


def validate_simulator(real_data_validation, simulation_results):
    """Compara resultados de simulación con datos reales de validación."""
    print("\n--- Validación del Simulador (Conceptual) ---")
    try:
        # Asegurar que los índices coincidan
        common_index = real_data_validation.index.intersection(simulation_results.index)
        if common_index.empty:
            print(
                "Error Validación: No hay índices comunes entre datos reales y simulados."
            )
            return None

        real_val = real_data_validation.loc[common_index]
        sim_val = simulation_results.loc[common_index]

        metrics = {}

        # Comparar consumo de red (si existe en datos reales)
        if "p_grid_real_kw" in real_val.columns:
            rmse_grid = np.sqrt(
                mean_squared_error(real_val["p_grid_real_kw"], sim_val["p_grid_kw"])
            )
            mae_grid = mean_absolute_error(
                real_val["p_grid_real_kw"], sim_val["p_grid_kw"]
            )
            metrics["grid_rmse"] = rmse_grid
            metrics["grid_mae"] = mae_grid
            print(
                f"  Validación Uso Red (kW): RMSE={rmse_grid:.4f}, MAE={mae_grid:.4f}"
            )
        else:
            print(
                "  Advertencia Validación: No hay datos reales de uso de red para comparar."
            )

        # Comparar SOC (si existe en datos reales)
        if "soc_real" in real_val.columns:
            rmse_soc = np.sqrt(mean_squared_error(real_val["soc_real"], sim_val["soc"]))
            mae_soc = mean_absolute_error(real_val["soc_real"], sim_val["soc"])
            metrics["soc_rmse"] = rmse_soc
            metrics["soc_mae"] = mae_soc
            print(f"  Validación SOC: RMSE={rmse_soc:.4f}, MAE={mae_soc:.4f}")
        else:
            print("  Advertencia Validación: No hay datos reales de SOC para comparar.")

        # Visualización (Ejemplo: SOC)
        if "soc_real" in real_val.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(
                real_val.index,
                real_val["soc_real"],
                label="SOC Real (Validación)",
                alpha=0.8,
            )
            plt.plot(
                sim_val.index,
                sim_val["soc"],
                label="SOC Simulado",
                linestyle="--",
                alpha=0.8,
            )
            plt.title("Validación: SOC Real vs Simulado")
            plt.xlabel("Tiempo")
            plt.ylabel("SOC")
            plt.legend()
            plt.ylim(0, 1)
            # Guardar figura
            # plt.savefig(f'resultados/{institucion_id}_validation_soc.png')
            plt.show()

        return metrics

    except Exception as e:
        print(f"Error durante la validación: {e}")
        return None


# %% 9. Bucle Principal de Procesamiento por Institución

resultados_globales = {}

# Crear directorios si no existen
os.makedirs("resultados", exist_ok=True)
os.makedirs("modelos", exist_ok=True)
os.makedirs("graficos", exist_ok=True)

for institucion_id, config in CONFIGURACIONES.items():
    print(f"\n{'='*20} Procesando Institución: {institucion_id} {'='*20}")

    # Cargar datos
    df_inst = load_data(config["data_path"])
    if df_inst is None:
        print(f"No se pudieron cargar los datos para {institucion_id}. Saltando...")
        continue

    params = config["params"]
    resultados_inst = {"params": params}

    # --- Análisis Estadístico ---
    print("\n--- Análisis Estadístico Descriptivo ---")
    desc_stats = df_inst[["consumo_kw", "irradiancia", "temp"]].describe()
    print(desc_stats)
    resultados_inst["desc_stats"] = desc_stats.to_dict()

    # Correlación
    try:
        correlation_matrix = df_inst[
            ["consumo_kw", "irradiancia", "temp", "hour"]
        ].corr()
        print("\nMatriz de Correlación:")
        print(correlation_matrix)
        resultados_inst["correlation"] = correlation_matrix.to_dict()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Matriz de Correlación - {institucion_id}")
        plt.tight_layout()
        plt.savefig(f"graficos/{institucion_id}_correlation_matrix.png")
        plt.close()  # Cerrar figura para no mostrarla ahora
    except Exception as e:
        print(f"Error generando heatmap de correlación: {e}")

    # --- Simulación Base (sin ML ni optimización aún) ---
    print("\n--- Simulación Base del Sistema (Paso a Paso) ---")
    # Usar todos los datos para esta simulación base
    sim_base_results = simulate_step_by_step(params["initial_soc"], df_inst, params)
    resultados_inst["simulacion_base"] = sim_base_results

    # Graficar SOC base
    plt.figure(figsize=(12, 6))
    plt.plot(
        sim_base_results.index, sim_base_results["soc"], label="SOC Simulado (Base)"
    )
    plt.title(f"Simulación Base SOC - {institucion_id}")
    plt.xlabel("Tiempo")
    plt.ylabel("SOC")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graficos/{institucion_id}_sim_base_soc.png")
    plt.close()

    # --- Machine Learning ---
    # Predicción de Consumo
    ml_features_consumo = [
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "temp",
        "irradiancia",
    ]  # Usar variables temporales y clima
    ml_target_consumo = "consumo_kw"
    model_consumo, ml_results_consumo, ml_metrics_consumo = train_predict_ml(
        df_inst, ml_features_consumo, ml_target_consumo, institucion_id
    )
    resultados_inst["ml_consumo"] = {
        "metrics": ml_metrics_consumo,
        "results_test": ml_results_consumo,
    }

    # Graficar predicciones de consumo
    plt.figure(figsize=(15, 7))
    plt.plot(
        ml_results_consumo.index, ml_results_consumo["real"], label="Real", alpha=0.9
    )
    plt.plot(
        ml_results_consumo.index,
        ml_results_consumo["pred"],
        label="Predicción",
        alpha=0.7,
        linestyle="--",
    )
    plt.title(f"Predicción Consumo (Test) - {institucion_id}")
    plt.xlabel("Fecha")
    plt.ylabel("Consumo (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graficos/{institucion_id}_ml_consumo_test.png")
    plt.close()

    # Predicción de Generación PV (usando el cálculo directo como 'real' y ML para predecir)
    # Creamos la columna de generación PV 'real' (calculada)
    df_inst["p_pv_kw_calc"] = calcular_potencia_pv(
        df_inst["irradiancia"], df_inst["temp"], params
    )
    ml_features_pv = [
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "temp",
        "irradiancia",
    ]  # Similar a consumo
    ml_target_pv = "p_pv_kw_calc"
    model_pv, ml_results_pv, ml_metrics_pv = train_predict_ml(
        df_inst, ml_features_pv, ml_target_pv, institucion_id, save_path="modelos"
    )
    resultados_inst["ml_pv"] = {"metrics": ml_metrics_pv, "results_test": ml_results_pv}
    # (Opcional) Graficar predicciones PV

    # --- Optimización ---
    # Usar las predicciones del conjunto de test para simular un horizonte futuro
    # ¡¡IMPORTANTE!! En un caso real, necesitarías predicciones futuras del clima para generar estas P_pv_pred y P_load_pred
    p_pv_pred_optim = ml_results_pv["pred"]  # Usando predicciones ML de PV
    p_load_pred_optim = ml_results_consumo["pred"]  # Usando predicciones ML de Consumo

    # Asegurar que los índices coincidan para la optimización
    common_idx_optim = p_pv_pred_optim.index.intersection(p_load_pred_optim.index)
    if not common_idx_optim.empty:
        p_pv_pred_optim = p_pv_pred_optim.loc[common_idx_optim]
        p_load_pred_optim = p_load_pred_optim.loc[common_idx_optim]

        # Ejecutar optimización para un horizonte (e.g., 1 día = 24 horas)
        horizonte = 24 * 7  # Optimizar para una semana
        if len(p_load_pred_optim) >= horizonte:
            optim_schedule, optim_cost = optimize_operation_lp(
                params["initial_soc"],
                p_pv_pred_optim.values[:horizonte],
                p_load_pred_optim.values[:horizonte],
                params,
                horizonte_h=horizonte,
            )
            if optim_schedule is not None:
                resultados_inst["optimizacion"] = {
                    "schedule": optim_schedule,
                    "costo_total": optim_cost,
                }
                print(f"Costo optimizado estimado para {horizonte}h: ${optim_cost:.2f}")

                # Graficar cronograma de optimización
                plt.figure(figsize=(15, 10))
                plt.subplot(3, 1, 1)
                plt.plot(
                    optim_schedule.index,
                    optim_schedule["p_pv_pred_kw"],
                    label="PV Predicha",
                    color="orange",
                )
                plt.plot(
                    optim_schedule.index,
                    optim_schedule["p_load_pred_kw"],
                    label="Carga Predicha",
                    color="blue",
                )
                plt.plot(
                    optim_schedule.index,
                    optim_schedule["p_grid_buy_kw"],
                    label="Red Comprada (Opt)",
                    color="red",
                    linestyle="--",
                )
                plt.ylabel("Potencia (kW)")
                plt.legend()
                plt.title(f"Optimización Potencias - {institucion_id}")

                plt.subplot(3, 1, 2)
                plt.plot(
                    optim_schedule.index,
                    optim_schedule["p_bat_ch_kw"],
                    label="Carga Bat (Opt)",
                    color="green",
                )
                plt.plot(
                    optim_schedule.index,
                    -optim_schedule["p_bat_dis_kw"],
                    label="Descarga Bat (Opt)",
                    color="purple",
                )  # Negativo para visualización
                plt.ylabel("Potencia Bat (kW)")
                plt.legend()

                plt.subplot(3, 1, 3)
                plt.plot(
                    optim_schedule.index,
                    optim_schedule["soc_opt"],
                    label="SOC Optimizado",
                    color="cyan",
                )
                plt.ylabel("SOC")
                plt.xlabel("Tiempo (Pasos)")
                plt.ylim(0, 1)
                plt.legend()

                plt.tight_layout()
                plt.savefig(f"graficos/{institucion_id}_optimizacion_schedule.png")
                plt.close()
            else:
                print("La optimización no tuvo éxito.")
                resultados_inst["optimizacion"] = None
        else:
            print(
                f"No hay suficientes datos predichos ({len(p_load_pred_optim)}) para el horizonte de optimización ({horizonte})."
            )
            resultados_inst["optimizacion"] = None
    else:
        print(
            "Índices de predicciones PV y Consumo no coinciden. Saltando optimización."
        )
        resultados_inst["optimizacion"] = None

    # --- Validación (Usando la simulación base como 'simulado' y datos originales como 'real') ---
    # ¡¡IMPORTANTE!! Esto NO es una validación real. Necesitarías un conjunto de datos
    # completamente separado (no usado en entrenamiento/test) para una validación adecuada.
    # Aquí solo se compara la simulación paso a paso (que usó datos 'reales' como input)
    # con esos mismos datos, lo cual no mide el error de predicción.
    # Se necesitaría añadir columnas 'soc_real', 'p_grid_real_kw' a los datos si se tuvieran mediciones.
    # validation_metrics = validate_simulator(df_inst, sim_base_results) # Conceptual
    # resultados_inst['validacion'] = validation_metrics
    print("\nADVERTENCIA: La validación real requiere un conjunto de datos separado.")
    resultados_inst["validacion"] = "Requiere datos reales separados"

    # --- Guardar Resultados de la Institución ---
    resultados_globales[institucion_id] = resultados_inst
    # Opcional: Guardar DataFrames de resultados en CSV/Excel
    sim_base_results.to_csv(f"resultados/{institucion_id}_simulacion_base_results.csv")
    if (
        "optimizacion" in resultados_inst
        and resultados_inst["optimizacion"] is not None
    ):
        resultados_inst["optimizacion"]["schedule"].to_csv(
            f"resultados/{institucion_id}_optimizacion_schedule.csv"
        )


# %% 10. Resumen Final y Conclusiones

print(f"\n{'='*20} Procesamiento Finalizado {'='*20}")

# Crear un resumen de métricas clave
summary_list = []
for inst_id, results in resultados_globales.items():
    metrics = {"institucion": inst_id}
    # Añadir estadísticas descriptivas clave
    if "desc_stats" in results and "consumo_kw" in results["desc_stats"]:
        metrics["consumo_medio_kw"] = results["desc_stats"]["consumo_kw"].get("mean")
        metrics["consumo_std_kw"] = results["desc_stats"]["consumo_kw"].get("std")
    # Añadir métricas ML (ej. R2 de test para consumo)
    if "ml_consumo" in results and results["ml_consumo"]:
        metrics["ml_consumo_r2_test"] = results["ml_consumo"]["metrics"].get("r2")
        metrics["ml_consumo_rmse_test"] = results["ml_consumo"]["metrics"].get("rmse")
    if "ml_pv" in results and results["ml_pv"]:
        metrics["ml_pv_r2_test"] = results["ml_pv"]["metrics"].get("r2")
    # Añadir costo optimizado
    if "optimizacion" in results and results["optimizacion"]:
        metrics["costo_optim_total"] = results["optimizacion"].get("costo_total")
        metrics["costo_optim_por_hora"] = (
            results["optimizacion"].get("costo_total")
            / len(results["optimizacion"]["schedule"])
            if results["optimizacion"].get("costo_total") is not None
            else None
        )

    summary_list.append(metrics)

summary_df = pd.DataFrame(summary_list)
print("\n--- Resumen de Métricas Clave ---")
print(summary_df)
summary_df.to_csv("resultados/summary_metrics_global.csv", index=False)

print("\n--- Notas Importantes ---")
print(
    "1. Este script utilizó DATOS SINTÉTICOS. Reemplace la generación de datos con la carga de sus archivos CSV reales."
)
print(
    "2. Los PARÁMETROS del sistema (paneles, batería, etc.) deben ajustarse a los valores específicos de cada institución."
)
print(
    "3. Los modelos (ODE, Optimización) son SIMPLIFICADOS. Refínelos según sea necesario."
)
print(
    "4. La OPTIMIZACIÓN se realizó con predicciones del conjunto de test. Para operación real, se requieren pronósticos futuros."
)
print(
    "5. La VALIDACIÓN requiere un conjunto de datos real y separado que no se usó para entrenamiento o test."
)
print(
    f"\nResultados detallados, modelos y gráficos guardados en las carpetas: 'resultados', 'modelos', 'graficos'."
)
