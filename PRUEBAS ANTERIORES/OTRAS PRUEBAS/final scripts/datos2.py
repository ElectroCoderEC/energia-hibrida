import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings("ignore")

# Configuración de la simulación
np.random.seed(42)  # Para reproducibilidad
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 6, 30)
dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]

print(
    f"Generando datos para {len(dates)} días ({start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')})"
)

# Parámetros para centros educativos rurales en Chimborazo, Ecuador
# Basado en datos reales de altitudes y características geográficas de la provincia
school_params = {
    "Escuela Intercultural Bilingüe Mushuk Pakari": {
        "tipo": "Educación Básica",
        "tamaño": "pequeña",
        "estudiantes": 85,
        "docentes": 6,
        "personal_admin": 2,
        "altitud": 3200,  # Riobamba zona rural
        "terreno": "valle_andino",
        "ubicacion": "Riobamba Rural",
        "area_total_m2": 2400,
        "area_techada_m2": 800,
        "area_disponible_paneles_m2": 450,
        "distancia_red_km": 15,
        "acceso_vehicular": "limitado",
    },
    "Unidad Educativa Sultana de Los Andes": {
        "tipo": "Educación Básica y Bachillerato",
        "tamaño": "mediana",
        "estudiantes": 180,
        "docentes": 14,
        "personal_admin": 4,
        "altitud": 2850,  # Zona de Guano
        "terreno": "planicie_andina",
        "ubicacion": "Guano Rural",
        "area_total_m2": 4200,
        "area_techada_m2": 1400,
        "area_disponible_paneles_m2": 750,
        "distancia_red_km": 8,
        "acceso_vehicular": "bueno",
    },
    "Escuela Comunitaria del Chimborazo": {
        "tipo": "Educación Básica",
        "tamaño": "pequeña",
        "estudiantes": 45,
        "docentes": 4,
        "personal_admin": 1,
        "altitud": 3800,  # Zona alta cerca del Chimborazo
        "terreno": "montañoso",
        "ubicacion": "Chimborazo Alto",
        "area_total_m2": 1800,
        "area_techada_m2": 600,
        "area_disponible_paneles_m2": 320,
        "distancia_red_km": 25,
        "acceso_vehicular": "muy_limitado",
    },
    "Colegio Técnico Agropecuario Chambo": {
        "tipo": "Bachillerato Técnico",
        "tamaño": "grande",
        "estudiantes": 280,
        "docentes": 22,
        "personal_admin": 8,
        "altitud": 2650,  # Zona de Chambo
        "terreno": "valle_productivo",
        "ubicacion": "Chambo Rural",
        "area_total_m2": 6500,
        "area_techada_m2": 2200,
        "area_disponible_paneles_m2": 1100,
        "distancia_red_km": 5,
        "acceso_vehicular": "excelente",
    },
    "Instituto Intercultural Puruhá": {
        "tipo": "Educación Superior",
        "tamaño": "mediana",
        "estudiantes": 150,
        "docentes": 18,
        "personal_admin": 6,
        "altitud": 3100,  # Zona de Colta
        "terreno": "páramo_bajo",
        "ubicacion": "Colta Rural",
        "area_total_m2": 3800,
        "area_techada_m2": 1200,
        "area_disponible_paneles_m2": 650,
        "distancia_red_km": 12,
        "acceso_vehicular": "regular",
    },
}


# Función para generar datos climáticos ajustados a Chimborazo
def generate_climate_data():
    """
    Genera datos climáticos basados en patrones reales de Chimborazo, Ecuador
    Considera: estacionalidad, altitud, efectos orográficos, y variabilidad climática andina
    """
    climate_data = []

    for school_name, params in school_params.items():
        print(f"Generando datos climáticos para {school_name}...")

        # Parámetros base según altitud y ubicación en Chimborazo
        if params["altitud"] > 3500:  # Zona muy alta
            base_radiation = 5.2  # Mayor radiación en altura
            base_wind = 6.5  # Más viento en zonas altas
            temp_base = 8  # Más frío
        elif params["altitud"] > 3000:  # Zona alta
            base_radiation = 4.8
            base_wind = 4.2
            temp_base = 12
        else:  # Zona media
            base_radiation = 4.3
            base_wind = 3.1
            temp_base = 16

        # Ajustes por terreno
        terrain_factors = {
            "valle_andino": {"rad": 1.0, "wind": 0.8, "temp": 1.1},
            "planicie_andina": {"rad": 1.05, "wind": 1.2, "temp": 1.0},
            "montañoso": {"rad": 1.15, "wind": 1.4, "temp": 0.9},
            "valle_productivo": {"rad": 0.95, "wind": 0.9, "temp": 1.05},
            "páramo_bajo": {"rad": 1.1, "wind": 1.3, "temp": 0.85},
        }

        terrain = params["terreno"]
        t_factor = terrain_factors[terrain]

        for day, date in enumerate(dates):
            # Variación estacional (Ecuador está en el ecuador, pero hay estaciones secas/húmedas)
            day_of_year = date.timetuple().tm_yday

            # Estación seca (junio-septiembre) vs húmeda (octubre-mayo)
            if 150 <= day_of_year <= 270:  # Estación seca
                seasonal_rad = 1.2
                seasonal_rain = 0.3
                seasonal_cloud = 0.6
            else:  # Estación húmeda
                seasonal_rad = 0.85
                seasonal_rain = 1.8
                seasonal_cloud = 1.4

            # Radiación solar diaria
            daily_radiation = (
                base_radiation
                * t_factor["rad"]
                * seasonal_rad
                * (0.9 + 0.2 * np.random.random())
            )
            daily_radiation = max(daily_radiation, 1.5)

            # Velocidad del viento
            daily_wind = base_wind * t_factor["wind"] * (0.8 + 0.4 * np.random.random())
            daily_wind = max(daily_wind, 0.5)

            # Temperatura
            daily_temp = (
                temp_base * t_factor["temp"]
                + 3 * np.sin(2 * np.pi * day_of_year / 365)
                + np.random.normal(0, 2)
            )

            # Humedad relativa (alta en zonas andinas)
            humidity = (
                np.random.uniform(60, 85)
                if params["altitud"] > 3000
                else np.random.uniform(55, 75)
            )

            # Nubosidad
            cloud_cover = min(0.9, seasonal_cloud * np.random.beta(2, 3))

            # Precipitación
            if np.random.random() < (
                0.4 * seasonal_rain / (1 + 0.1 * params["altitud"] / 1000)
            ):
                precipitation = np.random.exponential(8) * seasonal_rain
            else:
                precipitation = 0

            # Presión atmosférica (decrece con altitud)
            pressure = 1013.25 * (1 - 0.0065 * params["altitud"] / 288.15) ** 5.255

            climate_data.append(
                {
                    "fecha": date,
                    "escuela": school_name,
                    "radiacion_solar_kwh_m2": round(daily_radiation, 2),
                    "velocidad_viento_m_s": round(daily_wind, 2),
                    "temperatura_c": round(daily_temp, 1),
                    "humedad_relativa_pct": round(humidity, 1),
                    "nubosidad": round(cloud_cover, 2),
                    "precipitacion_mm": round(precipitation, 1),
                    "presion_atmosferica_hpa": round(pressure, 1),
                    "altitud_m": params["altitud"],
                    "terreno": terrain,
                }
            )

    return pd.DataFrame(climate_data)


# Función para generar datos de consumo energético
def generate_consumption_data():
    """
    Genera patrones de consumo realistas para centros educativos rurales
    Considera: tipo de institución, horarios, equipamiento, y actividades
    """
    consumption_data = []

    for school_name, params in school_params.items():
        print(f"Generando datos de consumo para {school_name}...")

        # Consumo base según tipo y tamaño
        if params["tipo"] == "Educación Básica":
            base_daily = 25 + params["estudiantes"] * 0.15  # kWh/día
        elif params["tipo"] == "Educación Básica y Bachillerato":
            base_daily = 35 + params["estudiantes"] * 0.18
        elif params["tipo"] == "Bachillerato Técnico":
            base_daily = 45 + params["estudiantes"] * 0.25  # Más equipos técnicos
        else:  # Educación Superior
            base_daily = 40 + params["estudiantes"] * 0.22

        # Factor por personal
        base_daily += (params["docentes"] + params["personal_admin"]) * 1.5

        # Patrones horarios según tipo de institución
        if params["tipo"] == "Educación Superior":
            # Horarios más amplios
            morning_hours = list(range(7, 12))
            afternoon_hours = list(range(13, 18))
            evening_hours = list(range(18, 21))
        else:
            # Horarios escolares tradicionales
            morning_hours = list(range(7, 12))
            afternoon_hours = list(range(13, 17))
            evening_hours = list(range(17, 19))

        for date in dates:
            # Factores estacionales
            is_vacation = (
                date.month in [12, 1, 2]
                or (date.month == 7 and date.day > 15)
                or date.month == 8
            )
            vacation_factor = 0.15 if is_vacation else 1.0

            # Factor por día de la semana
            day_of_week = date.weekday()

            for hour in range(24):
                if day_of_week < 5:  # Lunes a viernes
                    if hour in morning_hours:
                        hourly_factor = 0.15 * (0.8 + 0.4 * np.random.random())
                    elif hour in afternoon_hours:
                        hourly_factor = 0.12 * (0.7 + 0.3 * np.random.random())
                    elif hour in evening_hours:
                        hourly_factor = 0.08 * (0.5 + 0.3 * np.random.random())
                    else:
                        hourly_factor = 0.02 * (0.3 + 0.2 * np.random.random())
                else:  # Fines de semana
                    if 8 <= hour <= 17:
                        hourly_factor = 0.03 * (0.2 + 0.3 * np.random.random())
                    else:
                        hourly_factor = 0.01 * (0.1 + 0.1 * np.random.random())

                hourly_consumption = base_daily * hourly_factor * vacation_factor

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "consumo_kwh": round(hourly_consumption, 3),
                        "tipo_institucion": params["tipo"],
                        "estudiantes": params["estudiantes"],
                        "personal_total": params["docentes"] + params["personal_admin"],
                    }
                )

    return pd.DataFrame(consumption_data)


# Función para generar características de sistemas renovables
def generate_renewable_systems():
    """
    Define sistemas híbridos dimensionados según necesidades y recursos disponibles
    """
    systems_data = []

    for school_name, params in school_params.items():
        print(f"Dimensionando sistema renovable para {school_name}...")

        # Dimensionamiento según área disponible y demanda
        area_panels = params["area_disponible_paneles_m2"]

        # Capacidad solar (considerando 200W/m² y factor de ocupación 80%)
        solar_capacity = area_panels * 0.2 * 0.8  # kWp

        # Capacidad eólica según viento promedio y espacio
        if params["terreno"] in ["montañoso", "páramo_bajo"]:
            wind_capacity = np.random.uniform(2, 5)  # Mejor recurso eólico
        else:
            wind_capacity = np.random.uniform(1, 3)

        # Capacidad de almacenamiento (2-3 días de autonomía)
        daily_consumption = 25 + params["estudiantes"] * 0.2  # Estimación
        battery_capacity = daily_consumption * np.random.uniform(2, 3)

        # Características técnicas actualizadas
        systems_data.append(
            {
                "escuela": school_name,
                "capacidad_solar_kWp": round(solar_capacity, 2),
                "capacidad_eolica_kW": round(wind_capacity, 2),
                "capacidad_bateria_kWh": round(battery_capacity, 2),
                "eficiencia_solar": round(
                    np.random.uniform(0.18, 0.22), 3
                ),  # Paneles modernos
                "eficiencia_eolica": round(np.random.uniform(0.35, 0.45), 3),
                "eficiencia_bateria": round(
                    np.random.uniform(0.90, 0.95), 3
                ),  # Baterías litio
                "eficiencia_inversor": round(np.random.uniform(0.94, 0.97), 3),
                "edad_sistema_años": np.random.randint(0, 3),
                "tipo_panel": np.random.choice(
                    ["monocristalino", "policristalino", "bifacial"]
                ),
                "tipo_bateria": np.random.choice(["litio_ferro", "litio_ion", "gel"]),
                "tipo_aerogenerador": np.random.choice(
                    ["eje_horizontal", "eje_vertical"]
                ),
                "area_total_m2": params["area_total_m2"],
                "area_techada_m2": params["area_techada_m2"],
                "area_paneles_m2": params["area_disponible_paneles_m2"],
                "altitud_m": params["altitud"],
                "terreno": params["terreno"],
                "distancia_red_km": params["distancia_red_km"],
                "acceso_vehicular": params["acceso_vehicular"],
            }
        )

    return pd.DataFrame(systems_data)


# Función para calcular generación de energía
def calculate_energy_generation(df_climate, df_systems):
    """
    Calcula generación solar y eólica con modelos físicos mejorados
    """
    generation_data = []

    for _, system in df_systems.iterrows():
        school_name = system["escuela"]
        school_climate = df_climate[df_climate["escuela"] == school_name]

        print(f"Calculando generación para {school_name}...")

        for _, climate in school_climate.iterrows():
            date = climate["fecha"]

            # === GENERACIÓN SOLAR ===
            radiation = climate["radiacion_solar_kwh_m2"]
            temperature = climate["temperatura_c"]
            cloud_cover = climate["nubosidad"]

            # Efecto de temperatura (coef. -0.4%/°C sobre 25°C)
            temp_factor = 1 - max(0, (temperature - 25) * 0.004)

            # Efecto de nubosidad
            cloud_factor = 1 - cloud_cover * 0.4

            # Efecto de altitud (mayor radiación UV)
            altitude_factor = 1 + (system["altitud_m"] - 2500) * 0.00005

            # Degradación por edad
            degradation = 1 - system["edad_sistema_años"] * 0.007

            solar_generation = (
                system["capacidad_solar_kWp"]
                * radiation
                * system["eficiencia_solar"]
                * temp_factor
                * cloud_factor
                * altitude_factor
                * degradation
                * system["eficiencia_inversor"]
            )

            # === GENERACIÓN EÓLICA ===
            wind_speed = climate["velocidad_viento_m_s"]
            pressure = climate["presion_atmosferica_hpa"]

            # Corrección por densidad del aire
            air_density_factor = pressure / 1013.25

            # Curva de potencia simplificada
            cut_in = 2.5  # m/s
            rated_speed = 12.0  # m/s
            cut_out = 25.0  # m/s

            if wind_speed < cut_in or wind_speed > cut_out:
                wind_factor = 0
            elif wind_speed < rated_speed:
                wind_factor = ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3
            else:
                wind_factor = 1.0

            # Horas equivalentes diarias
            daily_hours = 18 if wind_speed > cut_in else 12

            wind_generation = (
                system["capacidad_eolica_kW"]
                * wind_factor
                * system["eficiencia_eolica"]
                * air_density_factor
                * daily_hours
                / 24
            )

            generation_data.append(
                {
                    "fecha": date,
                    "escuela": school_name,
                    "generacion_solar_kwh": round(max(0, solar_generation), 3),
                    "generacion_eolica_kwh": round(max(0, wind_generation), 3),
                    "generacion_total_kwh": round(
                        max(0, solar_generation + wind_generation), 3
                    ),
                    "radiacion_utilizada": round(radiation * cloud_factor, 2),
                    "factor_temperatura": round(temp_factor, 3),
                    "factor_viento": round(wind_factor, 3),
                }
            )

    return pd.DataFrame(generation_data)


# Función para calcular balance energético y almacenamiento
def calculate_energy_balance(df_consumption, df_generation, df_systems):
    """
    Calcula balance energético considerando almacenamiento en baterías
    """
    # Consumo diario total
    daily_consumption = (
        df_consumption.groupby(["fecha", "escuela"])["consumo_kwh"].sum().reset_index()
    )
    daily_consumption.rename(
        columns={"consumo_kwh": "consumo_diario_kwh"}, inplace=True
    )

    # Merge con generación
    df_balance = pd.merge(df_generation, daily_consumption, on=["fecha", "escuela"])

    # Añadir datos del sistema
    df_balance = pd.merge(
        df_balance,
        df_systems[["escuela", "capacidad_bateria_kWh", "eficiencia_bateria"]],
        on="escuela",
    )

    balance_data = []

    for school in df_systems["escuela"].unique():
        school_data = df_balance[df_balance["escuela"] == school].sort_values("fecha")
        battery_capacity = school_data.iloc[0]["capacidad_bateria_kWh"]
        battery_efficiency = school_data.iloc[0]["eficiencia_bateria"]

        # Estado inicial de batería (50%)
        battery_soc = battery_capacity * 0.5

        for _, row in school_data.iterrows():
            generation = row["generacion_total_kwh"]
            consumption = row["consumo_diario_kwh"]

            # Balance energético instantáneo
            net_energy = generation - consumption

            if net_energy > 0:  # Exceso de generación
                # Cargar batería
                max_charge = (battery_capacity - battery_soc) / battery_efficiency
                energy_to_battery = min(net_energy, max_charge)
                battery_soc += energy_to_battery * battery_efficiency
                excess_energy = net_energy - energy_to_battery
                grid_feed = excess_energy
                grid_consumption = 0
            else:  # Déficit de generación
                deficit = abs(net_energy)
                # Descargar batería
                available_battery = battery_soc * battery_efficiency
                energy_from_battery = min(deficit, available_battery)
                battery_soc -= energy_from_battery / battery_efficiency
                grid_consumption = deficit - energy_from_battery
                grid_feed = 0
                excess_energy = 0

            # Límites de SOC (10% - 90% para preservar batería)
            battery_soc = max(
                battery_capacity * 0.1, min(battery_capacity * 0.9, battery_soc)
            )

            # Cálculo de autosuficiencia
            self_sufficiency = (
                min(100, (generation / consumption * 100)) if consumption > 0 else 100
            )

            balance_data.append(
                {
                    "fecha": row["fecha"],
                    "escuela": row["escuela"],
                    "generacion_total_kwh": row["generacion_total_kwh"],
                    "consumo_diario_kwh": row["consumo_diario_kwh"],
                    "balance_neto_kwh": round(net_energy, 3),
                    "energia_bateria_kwh": round(battery_soc, 2),
                    "soc_bateria_pct": round((battery_soc / battery_capacity) * 100, 1),
                    "energia_red_kwh": round(grid_consumption, 3),
                    "energia_exportada_kwh": round(grid_feed, 3),
                    "excedente_kwh": round(excess_energy, 3),
                    "autosuficiencia_pct": round(self_sufficiency, 1),
                }
            )

    return pd.DataFrame(balance_data)


# Función para calcular económicos y ambientales
def calculate_economics_environment(df_balance, df_systems):
    """
    Calcula costos, ahorros y impacto ambiental
    """
    # Parámetros ecuatorianos actualizados
    electricity_cost = 0.10  # USD/kWh (costo promedio Ecuador rural)
    co2_factor = 0.385  # kg CO2/kWh (factor de emisión Ecuador)

    # Costos de inversión (USD 2024-2025)
    cost_data = []
    operational_data = []

    for _, system in df_systems.iterrows():
        school_name = system["escuela"]

        # Costos unitarios actualizados
        solar_cost = np.random.uniform(800, 1200)  # USD/kWp instalado
        wind_cost = np.random.uniform(1500, 2200)  # USD/kW instalado
        battery_cost = np.random.uniform(400, 700)  # USD/kWh (litio)

        # Inversión total
        solar_investment = system["capacidad_solar_kWp"] * solar_cost
        wind_investment = system["capacidad_eolica_kW"] * wind_cost
        battery_investment = system["capacidad_bateria_kWh"] * battery_cost

        # Costos adicionales
        installation_cost = 0.15 * (
            solar_investment + wind_investment + battery_investment
        )
        engineering_cost = 0.08 * (
            solar_investment + wind_investment + battery_investment
        )
        total_investment = (
            solar_investment
            + wind_investment
            + battery_investment
            + installation_cost
            + engineering_cost
        )

        # Costos O&M anuales
        annual_om = total_investment * np.random.uniform(0.02, 0.035)

        cost_data.append(
            {
                "escuela": school_name,
                "inversion_solar_usd": round(solar_investment, 2),
                "inversion_eolica_usd": round(wind_investment, 2),
                "inversion_baterias_usd": round(battery_investment, 2),
                "costo_instalacion_usd": round(installation_cost, 2),
                "costo_ingenieria_usd": round(engineering_cost, 2),
                "inversion_total_usd": round(total_investment, 2),
                "costo_om_anual_usd": round(annual_om, 2),
                "vida_util_solar_años": 25,
                "vida_util_eolica_años": 20,
                "vida_util_baterias_años": 12,
                "costo_kwh_sistema_usd": round(
                    total_investment
                    / (system["capacidad_solar_kWp"] + system["capacidad_eolica_kW"])
                    / 20
                    / 365,
                    4,
                ),
            }
        )

        # Datos operacionales diarios
        school_balance = df_balance[df_balance["escuela"] == school_name]

        for _, balance in school_balance.iterrows():
            daily_grid_cost = balance["energia_red_kwh"] * electricity_cost
            daily_savings = balance["generacion_total_kwh"] * electricity_cost
            co2_avoided = balance["generacion_total_kwh"] * co2_factor
            co2_emitted = balance["energia_red_kwh"] * co2_factor

            operational_data.append(
                {
                    "fecha": balance["fecha"],
                    "escuela": school_name,
                    "costo_energia_red_usd": round(daily_grid_cost, 3),
                    "ahorro_diario_usd": round(daily_savings, 3),
                    "co2_evitado_kg": round(co2_avoided, 3),
                    "co2_emitido_kg": round(co2_emitted, 3),
                    "beneficio_ambiental_usd": round(
                        co2_avoided * 0.025, 3
                    ),  # Precio carbono estimado
                }
            )

    return pd.DataFrame(cost_data), pd.DataFrame(operational_data)


# Función principal para generar todos los datos
def generate_complete_dataset():
    """
    Genera el dataset completo y lo guarda en Excel
    """
    print("=== SIMULADOR DE SISTEMAS HÍBRIDOS RENOVABLES ===")
    print("Provincia de Chimborazo - Ecuador")
    print("Centros Educativos Rurales")
    print("=" * 50)

    # 1. Generar datos climáticos
    print("\n1. Generando datos climáticos...")
    df_climate = generate_climate_data()

    # 2. Generar datos de consumo
    print("\n2. Generando patrones de consumo...")
    df_consumption = generate_consumption_data()

    # 3. Definir sistemas renovables
    print("\n3. Dimensionando sistemas renovables...")
    df_systems = generate_renewable_systems()

    # 4. Calcular generación
    print("\n4. Calculando generación de energía...")
    df_generation = calculate_energy_generation(df_climate, df_systems)

    # 5. Balance energético
    print("\n5. Calculando balance energético...")
    df_balance = calculate_energy_balance(df_consumption, df_generation, df_systems)

    # 6. Análisis económico y ambiental
    print("\n6. Calculando impactos económicos y ambientales...")
    df_costs, df_operational = calculate_economics_environment(df_balance, df_systems)

    # 7. Crear dataset consolidado para ML
    print("\n7. Consolidando datos para Machine Learning...")

    # Merge de todos los datos
    df_ml = pd.merge(df_balance, df_climate, on=["fecha", "escuela"])
    df_ml = pd.merge(df_ml, df_operational, on=["fecha", "escuela"])

    # Añadir características del sistema
    system_features = df_systems[
        [
            "escuela",
            "capacidad_solar_kWp",
            "capacidad_eolica_kW",
            "capacidad_bateria_kWh",
            "area_paneles_m2",
            "altitud_m",
            "distancia_red_km",
        ]
    ]
    df_ml = pd.merge(df_ml, system_features, on="escuela")

    # Añadir características temporales
    df_ml["año"] = df_ml["fecha"].dt.year
    df_ml["mes"] = df_ml["fecha"].dt.month
    df_ml["dia_año"] = df_ml["fecha"].dt.dayofyear
    df_ml["dia_semana"] = df_ml["fecha"].dt.dayofweek
    df_ml["es_fin_semana"] = (df_ml["dia_semana"] >= 5).astype(int)

    # Variables categóricas a numéricas
    df_ml["terreno_codigo"] = pd.Categorical(df_ml["terreno"]).codes
    df_ml["escuela_codigo"] = pd.Categorical(df_ml["escuela"]).codes

    # Estadísticas móviles (ventanas de 7 y 30 días)
    df_ml = df_ml.sort_values(["escuela", "fecha"])

    for col in ["radiacion_solar_kwh_m2", "velocidad_viento_m_s", "temperatura_c"]:
        df_ml[f"{col}_ma7"] = (
            df_ml.groupby("escuela")[col]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df_ml[f"{col}_ma30"] = (
            df_ml.groupby("escuela")[col]
            .rolling(30, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    # Índices de eficiencia
    df_ml["eficiencia_sistema"] = np.where(
        df_ml["generacion_total_kwh"] > 0,
        df_ml["generacion_total_kwh"]
        / (df_ml["capacidad_solar_kWp"] + df_ml["capacidad_eolica_kW"]),
        0,
    )

    df_ml["ratio_consumo_generacion"] = np.where(
        df_ml["generacion_total_kwh"] > 0,
        df_ml["consumo_diario_kwh"] / df_ml["generacion_total_kwh"],
        np.inf,
    )

    # 8. Guardar en Excel
    print("\n8. Guardando datos en archivo Excel...")

    filename = f"simulacion_sistemas_hibridos_chimborazo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Hoja principal para Machine Learning
        df_ml.to_excel(writer, sheet_name="Datos_ML", index=False)

        # Hojas individuales para análisis detallado
        df_climate.to_excel(writer, sheet_name="Datos_Climaticos", index=False)
        df_consumption.to_excel(writer, sheet_name="Consumo_Energia", index=False)
        df_systems.to_excel(writer, sheet_name="Sistemas_Renovables", index=False)
        df_generation.to_excel(writer, sheet_name="Generacion_Energia", index=False)
        df_balance.to_excel(writer, sheet_name="Balance_Energetico", index=False)
        df_costs.to_excel(writer, sheet_name="Costos_Inversion", index=False)
        df_operational.to_excel(writer, sheet_name="Costos_Operacion", index=False)

        # Resumen ejecutivo
        create_executive_summary(writer, df_ml, df_systems, df_costs)

    print(f"\n✅ Dataset generado exitosamente: {filename}")
    print(f"📊 Total de registros ML: {len(df_ml):,}")
    print(f"🏫 Centros educativos: {len(df_systems)}")
    print(
        f"📅 Período: {df_ml['fecha'].min().strftime('%Y-%m-%d')} a {df_ml['fecha'].max().strftime('%Y-%m-%d')}"
    )

    print("\n📈 ESTADÍSTICAS GENERALES:")
    """
    # Estadísticas básicas
    print("\n📈 ESTADÍSTICAS GENERALES:")
    print(
        f"  • Generación solar promedio: {df_ml['generacion_solar_kwh'].mean():.2f} kWh/día"
    )
    print(
        f"  • Generación eólica promedio: {df_ml['generacion_eolica_kwh'].mean():.2f} kWh/día"
    )
    print(f"  • Consumo promedio: {df_ml['consumo_diario_kwh'].mean():.2f} kWh/día")
    print(f"  • Autosuficiencia promedio: {df_ml['autosuficiencia_pct'].mean():.1f}%")
    print(f"  • Ahorro promedio diario: ${df_ml['ahorro_diario_usd'].mean():.2f}")
    """

    return df_ml, filename


def create_executive_summary(writer, df_ml, df_systems, df_costs):
    """
    Crea un resumen ejecutivo con indicadores clave
    """
    summary_data = []

    for school in df_systems["escuela"].unique():
        school_data = df_ml[df_ml["escuela"] == school]
        school_system = df_systems[df_systems["escuela"] == school].iloc[0]
        school_cost = df_costs[df_costs["escuela"] == school].iloc[0]

        # Métricas anuales (promedio de todos los años)
        annual_generation = (
            school_data["generacion_total_kwh"].sum() / 4.5
        )  # Promedio anual
        annual_consumption = school_data["consumo_diario_kwh"].sum() / 4.5
        annual_savings = school_data["ahorro_diario_usd"].sum() / 4.5
        annual_co2_avoided = school_data["co2_evitado_kg"].sum() / 4.5

        # ROI y payback
        roi_years = (
            school_cost["inversion_total_usd"] / annual_savings
            if annual_savings > 0
            else 999
        )

        summary_data.append(
            {
                "Escuela": school,
                "Tipo": (
                    school_system["tipo_institucion"]
                    if "tipo_institucion" in school_system
                    else "N/A"
                ),
                "Estudiantes": school_system.get("estudiantes", 0),
                "Altitud_m": school_system["altitud_m"],
                "Capacidad_Solar_kWp": school_system["capacidad_solar_kWp"],
                "Capacidad_Eolica_kW": school_system["capacidad_eolica_kW"],
                "Capacidad_Bateria_kWh": school_system["capacidad_bateria_kWh"],
                "Inversion_Total_USD": school_cost["inversion_total_usd"],
                "Generacion_Anual_kWh": round(annual_generation, 0),
                "Consumo_Anual_kWh": round(annual_consumption, 0),
                "Autosuficiencia_Promedio_%": round(
                    school_data["autosuficiencia_pct"].mean(), 1
                ),
                "Ahorro_Anual_USD": round(annual_savings, 0),
                "CO2_Evitado_Anual_kg": round(annual_co2_avoided, 0),
                "Payback_Años": round(roi_years, 1),
                "Area_Paneles_m2": school_system["area_paneles_m2"],
                "Distancia_Red_km": school_system["distancia_red_km"],
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name="Resumen_Ejecutivo", index=False)


# Función para análisis estadístico básico
def perform_basic_analysis(df_ml):
    """
    Realiza análisis estadístico básico para validar datos
    """
    print("\n🔍 ANÁLISIS DE VALIDACIÓN DE DATOS:")

    # Verificar valores faltantes
    missing_data = df_ml.isnull().sum()
    if missing_data.sum() > 0:
        print(f"⚠️  Valores faltantes encontrados: {missing_data.sum()}")
    else:
        print("✅ No hay valores faltantes")

    # Verificar rangos de valores
    validations = [
        ("Radiación solar", "radiacion_solar_kwh_m2", 0, 8),
        ("Velocidad viento", "velocidad_viento_m_s", 0, 30),
        ("Temperatura", "temperatura_c", -5, 35),
        ("Generación solar", "generacion_solar_kwh", 0, 100),
        ("Consumo diario", "consumo_diario_kwh", 0, 150),
        ("Autosuficiencia", "autosuficiencia_pct", 0, 100),
    ]

    for name, col, min_val, max_val in validations:
        if col in df_ml.columns:
            out_of_range = ((df_ml[col] < min_val) | (df_ml[col] > max_val)).sum()
            if out_of_range > 0:
                print(
                    f"⚠️  {name}: {out_of_range} valores fuera de rango [{min_val}, {max_val}]"
                )
            else:
                print(f"✅ {name}: todos los valores en rango válido")


# Función para preparar datos para ML
def prepare_ml_features(df_ml):
    """
    Prepara características específicas para modelos de ML
    """
    print("\n🤖 PREPARANDO DATOS PARA MACHINE LEARNING:")

    # Seleccionar características numéricas para ML
    ml_features = [
        # Variables climáticas
        "radiacion_solar_kwh_m2",
        "velocidad_viento_m_s",
        "temperatura_c",
        "humedad_relativa_pct",
        "nubosidad",
        "precipitacion_mm",
        "presion_atmosferica_hpa",
        # Variables del sistema
        "capacidad_solar_kWp",
        "capacidad_eolica_kW",
        "capacidad_bateria_kWh",
        "area_paneles_m2",
        "altitud_m",
        "distancia_red_km",
        # Variables temporales
        "mes",
        "dia_año",
        "dia_semana",
        "es_fin_semana",
        # Variables móviles
        "radiacion_solar_kwh_m2_ma7",
        "velocidad_viento_m_s_ma7",
        "temperatura_c_ma7",
        # Variables objetivo (para predicción)
        "generacion_total_kwh",
        "consumo_diario_kwh",
        "autosuficiencia_pct",
        "energia_red_kwh",
        "ahorro_diario_usd",
    ]

    # Filtrar columnas que existen
    available_features = [col for col in ml_features if col in df_ml.columns]
    df_ml_features = df_ml[["fecha", "escuela"] + available_features].copy()

    print(f"✅ Dataset ML preparado con {len(available_features)} características")
    print(f"📊 Registros para entrenamiento: {len(df_ml_features):,}")

    return df_ml_features


# Función principal de ejecución
def main():
    """
    Función principal que ejecuta toda la simulación
    """
    try:
        # Generar dataset completo
        df_ml, filename = generate_complete_dataset()

        # Análisis de validación
        perform_basic_analysis(df_ml)

        # Preparar datos para ML
        df_ml_prepared = prepare_ml_features(df_ml)

        print(f"\n🎯 CASOS DE USO PARA MACHINE LEARNING:")
        print("1. Predicción de generación solar/eólica basada en clima")
        print("2. Optimización de dimensionamiento de sistemas")
        print("3. Predicción de consumo energético")
        print("4. Análisis de autosuficiencia energética")
        print("5. Optimización económica de sistemas híbridos")
        print("6. Detección de anomalías en rendimiento")

        print(f"\n🏆 SIMULACIÓN COMPLETADA EXITOSAMENTE")
        print(f"📁 Archivo generado: {filename}")

        return df_ml, df_ml_prepared, filename

    except Exception as e:
        print(f"❌ Error durante la simulación: {str(e)}")
        raise


# Ejecutar simulación
if __name__ == "__main__":
    df_ml, df_ml_prepared, output_file = main()
