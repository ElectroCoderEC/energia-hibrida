import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Configuración de la simulación
np.random.seed(42)  # Para reproducibilidad
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 4, 30)
dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]
num_schools = 5  # Número de centros educativos a simular

# Parámetros para diferentes centros educativos (tamaño, equipamiento, ubicación)
school_params = {
    "Escuela_1": {
        "size": "small",
        "students": 50,
        "altitude": 2800,
        "terrain": "montañoso",
    },
    "Escuela_2": {
        "size": "medium",
        "students": 120,
        "altitude": 3200,
        "terrain": "valle",
    },
    "Escuela_3": {
        "size": "small",
        "students": 40,
        "altitude": 3500,
        "terrain": "montañoso",
    },
    "Escuela_4": {
        "size": "large",
        "students": 200,
        "altitude": 2600,
        "terrain": "planicie",
    },
    "Escuela_5": {
        "size": "medium",
        "students": 100,
        "altitude": 3000,
        "terrain": "valle",
    },
}

# Crear directorio para guardar datos si no existe
if not os.path.exists("datos_simulacion"):
    os.makedirs("datos_simulacion")


# 1. Generar datos climáticos
def generate_climate_data():
    # Parámetros para modelar variación estacional
    # En Ecuador (hemisferio sur), mayor radiación en diciembre-febrero, menor en junio-agosto
    days = np.arange(len(dates))

    climate_data = []

    for school_name, params in school_params.items():
        # Base de radiación según ubicación (más alta en montañas)
        base_radiation = 4.5 + (params["altitude"] - 2500) * 0.0003

        # Base de viento según terreno
        if params["terrain"] == "montañoso":
            base_wind = 4.0
        elif params["terrain"] == "valle":
            base_wind = 2.8
        else:  # planicie
            base_wind = 3.5

        for day, date in enumerate(dates):
            # Modelar variación estacional con una función sinusoidal (ciclo anual)
            seasonal_factor = (
                np.sin(2 * np.pi * (day - 15) / 365) + 1
            )  # +1 para mantenerlo positivo

            # Agregar variación diaria con ruido
            daily_radiation = base_radiation * (
                1 + 0.2 * seasonal_factor
            ) + np.random.normal(0, 0.5)
            daily_radiation = max(
                daily_radiation, 0.5
            )  # Asegurar valores mínimos razonables

            # Variación del viento (más variable que la radiación)
            daily_wind = base_wind * (1 + 0.15 * seasonal_factor) + np.random.normal(
                0, 1.0
            )
            daily_wind = max(daily_wind, 0.8)  # Asegurar valores mínimos razonables

            # Temperatura influye en el rendimiento de los paneles
            temp_base = (
                22 - (params["altitude"] - 2500) * 0.004
            )  # Disminuye con altitud
            daily_temp = temp_base + 5 * seasonal_factor + np.random.normal(0, 2)

            # Nubosidad (afecta a la radiación directa)
            cloud_cover = (
                np.random.beta(2, 5) if daily_radiation > 3 else np.random.beta(5, 2)
            )

            # Precipitación (puede afectar limpieza de paneles)
            rain_prob = (
                0.3 if seasonal_factor > 1 else 0.1
            )  # Más lluvia en temporada húmeda
            precipitation = (
                np.random.exponential(5) if np.random.random() < rain_prob else 0
            )

            climate_data.append(
                {
                    "fecha": date,
                    "escuela": school_name,
                    "radiacion_solar_kwh_m2": round(daily_radiation, 2),
                    "velocidad_viento_m_s": round(daily_wind, 2),
                    "temperatura_c": round(daily_temp, 1),
                    "nubosidad": round(cloud_cover, 2),
                    "precipitacion_mm": round(precipitation, 1),
                }
            )

    df_climate = pd.DataFrame(climate_data)
    return df_climate


# 2. Generar datos de consumo energético
def generate_consumption_data():
    consumption_data = []

    for school_name, params in school_params.items():
        # Consumo base según tamaño y número de estudiantes
        if params["size"] == "small":
            base_consumption = 15  # kWh/día para escuela pequeña
        elif params["size"] == "medium":
            base_consumption = 30  # kWh/día para escuela mediana
        else:  # large
            base_consumption = 50  # kWh/día para escuela grande

        # Ajustar por estudiante
        student_factor = params["students"] / 100  # Normalizar

        # Horarios de consumo (patrón típico escolar)
        morning_hours = [7, 8, 9, 10, 11, 12]  # Horario matutino
        afternoon_hours = [13, 14, 15, 16, 17]  # Horario vespertino
        evening_hours = [18, 19, 20]  # Actividades nocturnas ocasionales

        # Matriz de consumo (día de la semana x hora)
        # Las escuelas rurales generalmente operan de lunes a viernes
        weekday_patterns = np.zeros((7, 24))

        # Patrón de lunes a viernes (días 0-4)
        for day in range(5):
            for hour in range(24):
                if hour in morning_hours:
                    weekday_patterns[day, hour] = (
                        0.8 + 0.4 * np.random.random()
                    )  # Consumo alto
                elif hour in afternoon_hours:
                    weekday_patterns[day, hour] = (
                        0.6 + 0.3 * np.random.random()
                    )  # Consumo medio
                elif hour in evening_hours:
                    weekday_patterns[day, hour] = (
                        0.3 + 0.3 * np.random.random()
                    )  # Consumo bajo
                else:
                    weekday_patterns[day, hour] = (
                        0.05 + 0.05 * np.random.random()
                    )  # Consumo base

        # Fines de semana (días 5-6): consumo muy bajo, solo sistemas de seguridad
        for day in [5, 6]:
            for hour in range(24):
                weekday_patterns[day, hour] = 0.05 + 0.05 * np.random.random()

        # Generar consumo para cada día y hora
        for date in dates:
            # Vacaciones escolares (diciembre-febrero y julio-agosto en Ecuador)
            is_vacation = date.month in [12, 1, 2] or date.month in [7, 8]
            vacation_factor = (
                0.2 if is_vacation else 1.0
            )  # Reducción durante vacaciones

            day_of_week = date.weekday()

            for hour in range(24):
                # Consumo base * factor estudiantes * patrón diario * factor vacaciones
                hourly_consumption = (
                    base_consumption
                    * student_factor
                    * weekday_patterns[day_of_week, hour]
                    * vacation_factor
                )

                # Agregar variabilidad aleatoria (±10%)
                hourly_consumption *= 0.9 + 0.2 * np.random.random()

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "consumo_kwh": round(hourly_consumption, 3),
                    }
                )

    df_consumption = pd.DataFrame(consumption_data)
    return df_consumption


# 3. Generar datos de sistemas de energía renovable
def generate_renewable_system_data():
    systems_data = []

    # Parámetros para los sistemas de energía renovable
    for school_name, params in school_params.items():
        # Capacidad instalada según tamaño de la escuela
        if params["size"] == "small":
            solar_capacity = np.random.uniform(3, 5)  # kWp
            wind_capacity = np.random.uniform(1, 2)  # kW
            battery_capacity = np.random.uniform(5, 10)  # kWh
        elif params["size"] == "medium":
            solar_capacity = np.random.uniform(6, 10)  # kWp
            wind_capacity = np.random.uniform(2, 4)  # kW
            battery_capacity = np.random.uniform(12, 20)  # kWh
        else:  # large
            solar_capacity = np.random.uniform(12, 20)  # kWp
            wind_capacity = np.random.uniform(5, 8)  # kW
            battery_capacity = np.random.uniform(25, 40)  # kWh

        # Eficiencias y características técnicas
        solar_efficiency = np.random.uniform(0.14, 0.20)  # 14-20% eficiencia paneles
        wind_efficiency = np.random.uniform(
            0.25, 0.40
        )  # 25-40% eficiencia aerogeneradores
        battery_efficiency = np.random.uniform(0.85, 0.95)  # 85-95% eficiencia baterías

        # Edad del sistema (afecta rendimiento)
        system_age = np.random.randint(0, 5)  # 0-5 años

        # Degradación anual
        solar_degradation = 0.005 * system_age  # 0.5% degradación anual
        battery_degradation = 0.02 * system_age  # 2% degradación anual

        # Características de los componentes
        panel_type = np.random.choice(["monocristalino", "policristalino"])
        battery_type = np.random.choice(["litio", "plomo-ácido"])
        wind_type = np.random.choice(["eje horizontal", "eje vertical"])

        # Superficie disponible
        available_area = params["students"] * 2  # m² (estimación aproximada)

        systems_data.append(
            {
                "escuela": school_name,
                "capacidad_solar_kWp": round(solar_capacity, 2),
                "capacidad_eolica_kW": round(wind_capacity, 2),
                "capacidad_bateria_kWh": round(battery_capacity, 2),
                "eficiencia_solar": round(solar_efficiency, 3),
                "eficiencia_eolica": round(wind_efficiency, 3),
                "eficiencia_bateria": round(battery_efficiency, 3),
                "edad_sistema_años": system_age,
                "degradacion_solar": round(solar_degradation, 3),
                "degradacion_bateria": round(battery_degradation, 3),
                "tipo_panel": panel_type,
                "tipo_bateria": battery_type,
                "tipo_aerogenerador": wind_type,
                "area_disponible_m2": available_area,
                "altitud_m": params["altitude"],
                "terreno": params["terrain"],
            }
        )

    df_systems = pd.DataFrame(systems_data)
    return df_systems


# 4. Generar datos de generación de energía
def generate_power_generation_data(df_climate, df_systems):
    generation_data = []

    # Agrupar datos climáticos por escuela y fecha
    climate_grouped = df_climate.groupby(["escuela", "fecha"])

    # Iterar por cada escuela y día
    for (school_name, date), climate_group in climate_grouped:
        # Obtener parámetros del sistema para esta escuela
        system_params = df_systems[df_systems["escuela"] == school_name].iloc[0]

        # Datos climáticos para este día
        climate_day = climate_group.iloc[0]

        # Calcular generación solar
        solar_capacity = system_params["capacidad_solar_kWp"]
        solar_efficiency = system_params["eficiencia_solar"] * (
            1 - system_params["degradacion_solar"]
        )
        radiation = climate_day["radiacion_solar_kwh_m2"]
        temperature = climate_day["temperatura_c"]

        # Efecto de temperatura en eficiencia solar (disminuye ~0.4% por cada grado sobre 25°C)
        temp_factor = 1 - max(0, (temperature - 25) * 0.004)

        # Efecto de nubosidad
        cloud_factor = 1 - climate_day["nubosidad"] * 0.3

        # Generación solar diaria
        solar_generation = (
            solar_capacity * radiation * solar_efficiency * temp_factor * cloud_factor
        )

        # Calcular generación eólica
        wind_capacity = system_params["capacidad_eolica_kW"]
        wind_efficiency = system_params["eficiencia_eolica"]
        wind_speed = climate_day["velocidad_viento_m_s"]

        # Modelo simplificado de curva de potencia (cúbica entre cut-in y rated, constante en rated)
        cut_in_speed = 2.5  # m/s
        rated_speed = 10.0  # m/s

        if wind_speed < cut_in_speed:
            wind_factor = 0
        elif wind_speed < rated_speed:
            # Relación cúbica entre velocidad y potencia
            wind_factor = (
                (wind_speed - cut_in_speed) / (rated_speed - cut_in_speed)
            ) ** 3
        else:
            wind_factor = 1.0

        # Horas equivalentes (estimación)
        wind_hours = 10 if wind_speed > cut_in_speed else 6

        # Generación eólica diaria
        wind_generation = wind_capacity * wind_factor * wind_efficiency * wind_hours

        # Efecto de la altitud en densidad del aire (afecta generación eólica)
        altitude_factor = 1 - (system_params["altitud_m"] - 2500) * 0.00008
        wind_generation *= altitude_factor

        # Datos de generación agregados
        generation_data.append(
            {
                "fecha": date,
                "escuela": school_name,
                "generacion_solar_kwh": round(max(0, solar_generation), 2),
                "generacion_eolica_kwh": round(max(0, wind_generation), 2),
                "generacion_total_kwh": round(
                    max(0, solar_generation + wind_generation), 2
                ),
            }
        )

    df_generation = pd.DataFrame(generation_data)
    return df_generation


# 5. Generar datos de balance energético
def generate_energy_balance(df_consumption, df_generation):
    # Agregar consumo diario
    daily_consumption = (
        df_consumption.groupby(["fecha", "escuela"])["consumo_kwh"].sum().reset_index()
    )
    daily_consumption.rename(
        columns={"consumo_kwh": "consumo_diario_kwh"}, inplace=True
    )

    # Unir con datos de generación
    df_balance = pd.merge(df_generation, daily_consumption, on=["fecha", "escuela"])

    # Calcular balance y métricas
    df_balance["excedente_energetico_kwh"] = (
        df_balance["generacion_total_kwh"] - df_balance["consumo_diario_kwh"]
    )
    df_balance["deficit_energetico_kwh"] = np.where(
        df_balance["excedente_energetico_kwh"] < 0,
        -df_balance["excedente_energetico_kwh"],
        0,
    )
    df_balance["excedente_energetico_kwh"] = np.where(
        df_balance["excedente_energetico_kwh"] > 0,
        df_balance["excedente_energetico_kwh"],
        0,
    )

    # Calcular autonomía del sistema y dependencia de red
    df_balance["porcentaje_autosuficiencia"] = np.minimum(
        100, df_balance["generacion_total_kwh"] / df_balance["consumo_diario_kwh"] * 100
    )
    df_balance["energia_requerida_red_kwh"] = df_balance["deficit_energetico_kwh"]

    # Redondear valores
    for col in [
        "excedente_energetico_kwh",
        "deficit_energetico_kwh",
        "porcentaje_autosuficiencia",
        "energia_requerida_red_kwh",
    ]:
        df_balance[col] = df_balance[col].round(2)

    return df_balance


# 6. Generar datos de costos y emisiones
def generate_economics_data(df_systems, df_balance):
    # Parámetros económicos
    electricity_cost = 0.095  # USD/kWh (tarifa Ecuador)
    co2_per_kwh_grid = 0.385  # kg CO2/kWh (factor de emisión Ecuador)

    # Costos de inversión inicial (USD)
    system_costs = []
    for _, system in df_systems.iterrows():
        solar_unit_cost = np.random.uniform(1000, 1400)  # USD/kWp
        wind_unit_cost = np.random.uniform(1800, 2500)  # USD/kW
        battery_unit_cost = (
            np.random.uniform(400, 800)
            if system["tipo_bateria"] == "plomo-ácido"
            else np.random.uniform(800, 1200)
        )  # USD/kWh

        solar_investment = system["capacidad_solar_kWp"] * solar_unit_cost
        wind_investment = system["capacidad_eolica_kW"] * wind_unit_cost
        battery_investment = system["capacidad_bateria_kWh"] * battery_unit_cost

        installation_cost = 0.2 * (
            solar_investment + wind_investment + battery_investment
        )  # 20% del costo de equipos
        total_investment = (
            solar_investment + wind_investment + battery_investment + installation_cost
        )

        # Costos anuales de O&M (2-4% de la inversión inicial)
        annual_om_cost = np.random.uniform(0.02, 0.04) * total_investment

        # Vida útil de componentes
        solar_lifespan = np.random.randint(20, 25)  # años
        wind_lifespan = np.random.randint(15, 20)  # años
        battery_lifespan = (
            np.random.randint(5, 10)
            if system["tipo_bateria"] == "plomo-ácido"
            else np.random.randint(8, 12)
        )  # años

        system_costs.append(
            {
                "escuela": system["escuela"],
                "inversion_solar_usd": round(solar_investment, 2),
                "inversion_eolica_usd": round(wind_investment, 2),
                "inversion_baterias_usd": round(battery_investment, 2),
                "inversion_instalacion_usd": round(installation_cost, 2),
                "inversion_total_usd": round(total_investment, 2),
                "costo_om_anual_usd": round(annual_om_cost, 2),
                "vida_util_solar_años": solar_lifespan,
                "vida_util_eolica_años": wind_lifespan,
                "vida_util_baterias_años": battery_lifespan,
            }
        )

    df_costs = pd.DataFrame(system_costs)

    # Calcular costos de operación diarios y emisiones
    operational_data = []
    for _, balance in df_balance.iterrows():
        school_name = balance["escuela"]
        grid_energy = balance["energia_requerida_red_kwh"]

        # Costo diario de energía de la red
        daily_grid_cost = grid_energy * electricity_cost

        # Emisiones CO2 evitadas y emitidas
        co2_avoided = balance["generacion_total_kwh"] * co2_per_kwh_grid
        co2_emitted = grid_energy * co2_per_kwh_grid

        operational_data.append(
            {
                "fecha": balance["fecha"],
                "escuela": school_name,
                "costo_energia_red_usd": round(daily_grid_cost, 2),
                "co2_evitado_kg": round(co2_avoided, 2),
                "co2_emitido_kg": round(co2_emitted, 2),
            }
        )

    df_operational = pd.DataFrame(operational_data)

    return df_costs, df_operational


# Ejecutar generación de datos
print("Generando datos climáticos...")
df_climate = generate_climate_data()

print("Generando datos de consumo...")
df_consumption = generate_consumption_data()

print("Generando datos de sistemas renovables...")
df_systems = generate_renewable_system_data()

print("Generando datos de producción energética...")
df_generation = generate_power_generation_data(df_climate, df_systems)

print("Generando balance energético...")
df_balance = generate_energy_balance(df_consumption, df_generation)

print("Generando datos económicos...")
df_costs, df_operational = generate_economics_data(df_systems, df_balance)

# Guardar datos en Excel
print("Guardando datos en Excel...")
with pd.ExcelWriter("datos_simulacion/datos_sinteticos_redes_hibridas.xlsx") as writer:
    df_climate.to_excel(writer, sheet_name="Datos_Climaticos", index=False)
    df_consumption.to_excel(writer, sheet_name="Consumo_Energia", index=False)
    df_systems.to_excel(writer, sheet_name="Sistemas_Renovables", index=False)
    df_generation.to_excel(writer, sheet_name="Generacion_Energia", index=False)
    df_balance.to_excel(writer, sheet_name="Balance_Energetico", index=False)
    df_costs.to_excel(writer, sheet_name="Costos_Sistemas", index=False)
    df_operational.to_excel(writer, sheet_name="Costos_Operacion", index=False)

print("Datos generados y guardados con éxito.")


# Visualización básica de datos (ejemplos)
def plot_sample_data():
    # 1. Comparación de generación vs consumo para una escuela
    plt.figure(figsize=(12, 6))
    school_sample = "Escuela_1"
    school_data = df_balance[df_balance["escuela"] == school_sample].copy()
    school_data["fecha"] = pd.to_datetime(school_data["fecha"])
    school_data = school_data.sort_values("fecha")

    # Tomar solo los primeros 90 días para mejor visualización
    school_data = school_data.head(90)

    # Convertir fechas a numpy arrays para evitar problemas de indexación
    fechas = school_data["fecha"].values
    gen_total = school_data["generacion_total_kwh"].values
    consumo = school_data["consumo_diario_kwh"].values

    plt.plot(fechas, gen_total, "g-", label="Generación Total (kWh)")
    plt.plot(fechas, consumo, "r-", label="Consumo (kWh)")
    plt.title(f"Generación vs Consumo: {school_sample} (Primeros 90 días)")
    plt.xlabel("Fecha")
    plt.ylabel("Energía (kWh)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("datos_simulacion/generacion_vs_consumo.png")

    # 2. Distribución de fuentes de generación
    plt.figure(figsize=(12, 6))
    generation_by_source = df_generation[
        df_generation["escuela"] == school_sample
    ].copy()
    generation_by_source["fecha"] = pd.to_datetime(generation_by_source["fecha"])
    generation_by_source = generation_by_source.sort_values("fecha").head(90)

    fechas_gen = generation_by_source["fecha"].values
    solar_gen = generation_by_source["generacion_solar_kwh"].values
    eolica_gen = generation_by_source["generacion_eolica_kwh"].values

    plt.stackplot(
        fechas_gen,
        [solar_gen, eolica_gen],
        labels=["Solar", "Eólica"],
        colors=["gold", "skyblue"],
    )
    plt.title(
        f"Composición de Generación Renovable: {school_sample} (Primeros 90 días)"
    )
    plt.xlabel("Fecha")
    plt.ylabel("Generación (kWh)")
    plt.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("datos_simulacion/composicion_generacion.png")

    # 3. Porcentaje de autosuficiencia
    plt.figure(figsize=(12, 6))
    autosuficiencia = school_data["porcentaje_autosuficiencia"].values

    plt.plot(fechas, autosuficiencia, "b-")
    plt.axhline(y=100, color="r", linestyle="--", label="Autosuficiencia completa")
    plt.title(f"Porcentaje de Autosuficiencia: {school_sample} (Primeros 90 días)")
    plt.xlabel("Fecha")
    plt.ylabel("Autosuficiencia (%)")
    plt.ylim(0, 120)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("datos_simulacion/autosuficiencia.png")

    # 4. Comparación de inversión por escuela
    plt.figure(figsize=(10, 6))
    investment_data = df_costs.copy()
    schools = investment_data["escuela"]
    solar_inv = investment_data["inversion_solar_usd"]
    wind_inv = investment_data["inversion_eolica_usd"]
    battery_inv = investment_data["inversion_baterias_usd"]
    installation_inv = investment_data["inversion_instalacion_usd"]

    width = 0.6
    plt.bar(schools, solar_inv, width, label="Solar")
    plt.bar(schools, wind_inv, width, bottom=solar_inv, label="Eólica")
    plt.bar(schools, battery_inv, width, bottom=solar_inv + wind_inv, label="Baterías")
    plt.bar(
        schools,
        installation_inv,
        width,
        bottom=solar_inv + wind_inv + battery_inv,
        label="Instalación",
    )

    plt.title("Inversión por Componente y Escuela")
    plt.xlabel("Escuela")
    plt.ylabel("Inversión (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("datos_simulacion/inversion_por_escuela.png")


# Generar y guardar visualizaciones
print("Generando visualizaciones de muestra...")
plot_sample_data()

print(
    "Proceso completado. Los archivos se encuentran en la carpeta 'datos_simulacion'."
)
