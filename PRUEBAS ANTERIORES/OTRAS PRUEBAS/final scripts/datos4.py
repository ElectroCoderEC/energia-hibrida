import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import math

# Configuración de la simulación
np.random.seed(42)  # Para reproducibilidad
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 6, 30)
dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]

# Nombres específicos de las escuelas con parámetros detallados
school_params = {
    "Escuela 21 de Abril": {
        "students": 85,
        "altitude": 2950,
        "terrain": "montañoso",
        "area_hectares": 1.2,
        "lat": -1.4678,
        "lon": -78.8718,
        "infrastructure_level": "básica",
        "type": "primaria",
    },
    "Escuela Sangay": {
        "students": 165,
        "altitude": 3100,
        "terrain": "valle",
        "area_hectares": 2.1,
        "lat": -1.5234,
        "lon": -78.9012,
        "infrastructure_level": "intermedia",
        "type": "primaria-secundaria",
    },
    "Politécnica de Chimborazo": {
        "students": 280,
        "altitude": 2758,
        "terrain": "planicie",
        "area_hectares": 4.5,
        "lat": -1.6500,
        "lon": -78.6833,
        "infrastructure_level": "avanzada",
        "type": "superior",
    },
    "Colegio Condorazo": {
        "students": 220,
        "altitude": 3250,
        "terrain": "montañoso",
        "area_hectares": 3.2,
        "lat": -1.4123,
        "lon": -78.7456,
        "infrastructure_level": "intermedia",
        "type": "secundaria",
    },
    "Colegio Victor Proaño": {
        "students": 135,
        "altitude": 3050,
        "terrain": "valle",
        "area_hectares": 1.8,
        "lat": -1.5789,
        "lon": -78.8234,
        "infrastructure_level": "básica",
        "type": "secundaria",
    },
}

# Crear directorio para guardar datos si no existe
if not os.path.exists("datos_simulacion"):
    os.makedirs("datos_simulacion")


# 1. Generar datos climáticos específicos para Chimborazo
def generate_climate_data():
    climate_data = []
    days = np.arange(len(dates))

    for school_name, params in school_params.items():
        # Radiación solar base ajustada para Ecuador (ecuatorial)
        # Mayor radiación en septiembre-noviembre y marzo-mayo
        base_radiation = 4.8 + (params["altitude"] - 2750) * 0.0002

        # Ajuste por ubicación geográfica y terreno
        if params["terrain"] == "montañoso":
            base_wind = 5.2
            radiation_modifier = 1.1  # Mayor radiación en montañas
        elif params["terrain"] == "valle":
            base_wind = 3.1
            radiation_modifier = 0.95  # Algo menos por sombras
        else:  # planicie
            base_wind = 4.3
            radiation_modifier = 1.0

        for day, date in enumerate(dates):
            # Estacionalidad ecuatorial (dos estaciones secas y dos lluviosas)
            day_of_year = date.timetuple().tm_yday

            # Patrón bimodal de radiación (Ecuador)
            seasonal_radiation = (
                0.8
                + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                + 0.15 * np.sin(4 * np.pi * day_of_year / 365)
            )

            # Patrón de vientos (más fuertes en agosto-octubre)
            seasonal_wind = (
                0.9
                + 0.3 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
                + 0.1 * np.sin(4 * np.pi * day_of_year / 365)
            )

            # Radiación solar diaria
            daily_radiation = (
                base_radiation * radiation_modifier * seasonal_radiation
                + np.random.normal(0, 0.4)
            )
            daily_radiation = max(daily_radiation, 1.2)

            # Velocidad del viento
            daily_wind = base_wind * seasonal_wind + np.random.normal(0, 0.8)
            daily_wind = max(daily_wind, 0.5)

            # Temperatura basada en altitud y estacionalidad
            temp_base = 16 - (params["altitude"] - 2750) * 0.0065  # Gradiente térmico
            seasonal_temp = 3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_temp = temp_base + seasonal_temp + np.random.normal(0, 1.5)

            # Humedad relativa (mayor en valles)
            base_humidity = 75 if params["terrain"] == "valle" else 65
            seasonal_humidity = 15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
            daily_humidity = base_humidity + seasonal_humidity + np.random.normal(0, 5)
            daily_humidity = np.clip(daily_humidity, 30, 95)

            # Nubosidad (mayor en época lluviosa: dic-may)
            is_rainy_season = date.month in [12, 1, 2, 3, 4, 5]
            cloud_base = 0.6 if is_rainy_season else 0.3
            cloud_cover = np.clip(cloud_base + np.random.beta(2, 3) * 0.4, 0, 1)

            # Precipitación (patrón bimodal ecuatorial)
            rain_prob = 0.45 if is_rainy_season else 0.15
            precipitation = (
                np.random.exponential(8) if np.random.random() < rain_prob else 0
            )

            # Presión atmosférica (función de altitud)
            pressure = 1013.25 * (1 - 0.0065 * params["altitude"] / 288.15) ** 5.255

            climate_data.append(
                {
                    "fecha": date,
                    "escuela": school_name,
                    "radiacion_solar_kwh_m2": round(daily_radiation, 2),
                    "velocidad_viento_m_s": round(daily_wind, 2),
                    "temperatura_c": round(daily_temp, 1),
                    "humedad_relativa_pct": round(daily_humidity, 1),
                    "nubosidad": round(cloud_cover, 2),
                    "precipitacion_mm": round(precipitation, 1),
                    "presion_atmosferica_hpa": round(pressure, 1),
                    "altitud_m": params["altitude"],
                    "latitud": params["lat"],
                    "longitud": params["lon"],
                }
            )

    return pd.DataFrame(climate_data)


# 2. Generar datos de consumo energético detallado
def generate_consumption_data():
    consumption_data = []

    # Definir equipos típicos por tipo de institución
    equipment_profiles = {
        "primaria": {
            "base_consumption": 12,  # kWh/día
            "equipment": ["iluminación", "ventiladores", "computadoras_básicas"],
            "peak_hours": [8, 9, 10, 11, 14, 15, 16],
        },
        "secundaria": {
            "base_consumption": 25,
            "equipment": [
                "iluminación",
                "ventiladores",
                "computadoras",
                "laboratorio_básico",
            ],
            "peak_hours": [7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
        },
        "primaria-secundaria": {
            "base_consumption": 35,
            "equipment": [
                "iluminación",
                "ventiladores",
                "computadoras",
                "laboratorio_básico",
                "biblioteca",
            ],
            "peak_hours": [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18],
        },
        "superior": {
            "base_consumption": 55,
            "equipment": [
                "iluminación",
                "ventiladores",
                "computadoras",
                "laboratorios",
                "talleres",
                "cafetería",
            ],
            "peak_hours": [6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    }

    for school_name, params in school_params.items():
        profile = equipment_profiles[params["type"]]

        # Factor de escala por estudiantes
        student_factor = params["students"] / 150
        base_consumption = profile["base_consumption"] * student_factor

        # Factor de infraestructura
        infra_factors = {"básica": 0.8, "intermedia": 1.0, "avanzada": 1.3}
        infra_factor = infra_factors[params["infrastructure_level"]]

        # Generar patrones horarios
        hourly_patterns = np.zeros(24)
        for hour in range(24):
            if hour in profile["peak_hours"]:
                hourly_patterns[hour] = np.random.uniform(0.7, 1.2)
            elif hour in [6, 12, 18, 19]:  # Horarios de transición
                hourly_patterns[hour] = np.random.uniform(0.3, 0.6)
            else:  # Horarios de baja demanda
                hourly_patterns[hour] = np.random.uniform(0.05, 0.15)

        for date in dates:
            # Factores estacionales
            is_vacation = date.month in [12, 1, 2, 7, 8]
            vacation_factor = 0.25 if is_vacation else 1.0

            # Factor de día de semana
            weekday = date.weekday()
            if weekday < 5:  # Lunes a viernes
                weekday_factor = 1.0
            elif weekday == 5:  # Sábado
                weekday_factor = 0.3
            else:  # Domingo
                weekday_factor = 0.1

            for hour in range(24):
                hourly_consumption = (
                    base_consumption
                    * infra_factor
                    * hourly_patterns[hour]
                    * vacation_factor
                    * weekday_factor
                )

                # Variabilidad aleatoria
                hourly_consumption *= np.random.uniform(0.85, 1.15)

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "consumo_kwh": round(max(hourly_consumption, 0.01), 3),
                        "tipo_institucion": params["type"],
                        "num_estudiantes": params["students"],
                        "factor_infraestructura": infra_factor,
                    }
                )

    return pd.DataFrame(consumption_data)


# 3. Generar datos de sistemas de energía renovable
def generate_renewable_system_data():
    systems_data = []

    for school_name, params in school_params.items():
        # Dimensionamiento basado en estudiantes y área disponible
        students = params["students"]
        area_m2 = params["area_hectares"] * 10000

        # Capacidad solar (considerando 30% del área disponible)
        solar_area = min(area_m2 * 0.3, students * 8)  # m²
        solar_capacity = solar_area * 0.2  # kWp (200W/m²)

        # Capacidad eólica (basada en viento disponible y espacio)
        if params["terrain"] == "montañoso":
            wind_capacity = np.random.uniform(3, 8)
        elif params["terrain"] == "valle":
            wind_capacity = np.random.uniform(1, 4)
        else:  # planicie
            wind_capacity = np.random.uniform(2, 6)

        # Capacidad de baterías (2-3 días de autonomía)
        daily_consumption_est = students * 0.2  # kWh estimado
        battery_capacity = daily_consumption_est * np.random.uniform(2, 3)

        # Características técnicas
        panel_types = ["monocristalino", "policristalino"]
        battery_types = ["litio", "plomo-ácido", "gel"]
        inverter_types = ["string", "microinversor", "optimizador"]

        # Eficiencias realistas
        solar_efficiency = np.random.uniform(0.16, 0.22)
        wind_efficiency = np.random.uniform(0.28, 0.42)
        battery_efficiency = np.random.uniform(0.88, 0.96)
        inverter_efficiency = np.random.uniform(0.94, 0.98)

        # Edad y degradación
        system_age = np.random.randint(0, 6)
        solar_degradation = 0.005 * system_age
        battery_degradation = 0.03 * system_age

        systems_data.append(
            {
                "escuela": school_name,
                "capacidad_solar_kWp": round(solar_capacity, 2),
                "area_paneles_m2": round(solar_area, 1),
                "capacidad_eolica_kW": round(wind_capacity, 2),
                "capacidad_bateria_kWh": round(battery_capacity, 2),
                "capacidad_inversor_kW": round(
                    max(solar_capacity, wind_capacity) * 1.2, 2
                ),
                "eficiencia_solar": round(solar_efficiency, 3),
                "eficiencia_eolica": round(wind_efficiency, 3),
                "eficiencia_bateria": round(battery_efficiency, 3),
                "eficiencia_inversor": round(inverter_efficiency, 3),
                "edad_sistema_años": system_age,
                "degradacion_solar_anual": round(solar_degradation, 4),
                "degradacion_bateria_anual": round(battery_degradation, 4),
                "tipo_panel": np.random.choice(panel_types),
                "tipo_bateria": np.random.choice(battery_types),
                "tipo_inversor": np.random.choice(inverter_types),
                "area_total_disponible_m2": round(area_m2, 1),
                "orientacion_paneles_grados": np.random.randint(
                    -10, 11
                ),  # Desviación del sur
                "inclinacion_paneles_grados": round(
                    abs(params["lat"]) + np.random.uniform(-5, 5), 1
                ),
                "altura_aerogenerador_m": np.random.randint(15, 30),
                "factor_sombreado": round(np.random.uniform(0.02, 0.08), 3),
            }
        )

    return pd.DataFrame(systems_data)


# 4. Generar datos de generación de energía
def generate_power_generation_data(df_climate, df_systems):
    generation_data = []

    for _, climate in df_climate.iterrows():
        school_name = climate["escuela"]
        system = df_systems[df_systems["escuela"] == school_name].iloc[0]

        # Generación solar
        radiation = climate["radiacion_solar_kwh_m2"]
        temperature = climate["temperatura_c"]
        cloud_cover = climate["nubosidad"]

        # Efectos en la generación solar
        temp_coeff = -0.004  # %/°C
        temp_factor = 1 + temp_coeff * (temperature - 25)
        cloud_factor = 1 - cloud_cover * 0.7
        degradation_factor = 1 - system["degradacion_solar_anual"]
        shading_factor = 1 - system["factor_sombreado"]

        solar_generation = (
            system["capacidad_solar_kWp"]
            * radiation
            * system["eficiencia_solar"]
            * temp_factor
            * cloud_factor
            * degradation_factor
            * shading_factor
        )

        # Generación eólica
        wind_speed = climate["velocidad_viento_m_s"]
        air_density = (
            climate["presion_atmosferica_hpa"] * 100 / (287 * (temperature + 273.15))
        )
        density_factor = air_density / 1.225  # Densidad al nivel del mar

        # Curva de potencia del aerogenerador
        cut_in = 2.5
        rated_speed = 12.0
        cut_out = 25.0

        if wind_speed < cut_in or wind_speed > cut_out:
            wind_power_factor = 0
        elif wind_speed <= rated_speed:
            wind_power_factor = ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            wind_power_factor = 1.0

        # Horas equivalentes (estimación basada en velocidad promedio)
        equivalent_hours = min(24, wind_speed * 1.5)

        wind_generation = (
            system["capacidad_eolica_kW"]
            * wind_power_factor
            * system["eficiencia_eolica"]
            * density_factor
            * equivalent_hours
            / 24
        )

        # Pérdidas del sistema
        inverter_loss = 1 - system["eficiencia_inversor"]
        cable_loss = 0.02  # 2% pérdidas en cables
        system_loss = inverter_loss + cable_loss

        solar_net = solar_generation * (1 - system_loss)
        wind_net = wind_generation * (1 - system_loss)

        generation_data.append(
            {
                "fecha": climate["fecha"],
                "escuela": school_name,
                "generacion_solar_bruta_kwh": round(max(0, solar_generation), 3),
                "generacion_eolica_bruta_kwh": round(max(0, wind_generation), 3),
                "generacion_solar_neta_kwh": round(max(0, solar_net), 3),
                "generacion_eolica_neta_kwh": round(max(0, wind_net), 3),
                "generacion_total_kwh": round(max(0, solar_net + wind_net), 3),
                "perdidas_sistema_kwh": round(
                    max(0, (solar_generation + wind_generation) * system_loss), 3
                ),
                "factor_capacidad_solar": round(
                    max(0, solar_generation / (system["capacidad_solar_kWp"] * 24)), 3
                ),
                "factor_capacidad_eolica": round(
                    max(0, wind_generation / (system["capacidad_eolica_kW"] * 24)), 3
                ),
            }
        )

    return pd.DataFrame(generation_data)


# 5. Balance energético y almacenamiento
def generate_energy_balance(df_consumption, df_generation, df_systems):
    # Consumo diario
    daily_consumption = (
        df_consumption.groupby(["fecha", "escuela"])["consumo_kwh"].sum().reset_index()
    )
    daily_consumption.rename(
        columns={"consumo_kwh": "consumo_diario_kwh"}, inplace=True
    )

    # Unir datos
    df_balance = pd.merge(df_generation, daily_consumption, on=["fecha", "escuela"])

    # Simular sistema de baterías
    balance_data = []

    energy_from_battery = 0

    for school_name in school_params.keys():
        school_data = df_balance[df_balance["escuela"] == school_name].sort_values(
            "fecha"
        )
        system = df_systems[df_systems["escuela"] == school_name].iloc[0]

        battery_capacity = system["capacidad_bateria_kWh"]
        battery_efficiency = system["eficiencia_bateria"]
        battery_soc = battery_capacity * 0.5  # Estado inicial 50%

        for _, row in school_data.iterrows():
            generation = row["generacion_total_kwh"]
            consumption = row["consumo_diario_kwh"]

            # Balance instantáneo
            net_energy = generation - consumption

            if net_energy > 0:  # Exceso de generación
                # Cargar batería
                energy_to_battery = min(
                    net_energy, (battery_capacity - battery_soc) / battery_efficiency
                )
                battery_soc += energy_to_battery * battery_efficiency
                excess_energy = net_energy - energy_to_battery
                grid_injection = excess_energy
                grid_consumption = 0
            else:  # Déficit de generación
                deficit = abs(net_energy)
                # Descargar batería
                energy_from_battery = min(deficit, battery_soc * battery_efficiency)
                battery_soc -= energy_from_battery / battery_efficiency
                remaining_deficit = deficit - energy_from_battery
                grid_consumption = remaining_deficit
                grid_injection = 0
                excess_energy = 0

            # Autodescarga de batería (0.1% diario)
            battery_soc *= 0.999
            battery_soc = max(0, min(battery_soc, battery_capacity))

            # Calcular métricas
            autosuficiencia = min(
                100, (generation / consumption * 100) if consumption > 0 else 100
            )

            balance_data.append(
                {
                    "fecha": row["fecha"],
                    "escuela": school_name,
                    "generacion_total_kwh": generation,
                    "consumo_diario_kwh": consumption,
                    "balance_energetico_kwh": net_energy,
                    "energia_excedente_kwh": round(excess_energy, 3),
                    "energia_deficit_kwh": round(
                        abs(net_energy) if net_energy < 0 else 0, 3
                    ),
                    "energia_bateria_carga_kwh": round(
                        energy_to_battery if net_energy > 0 else 0, 3
                    ),
                    "energia_bateria_descarga_kwh": round(
                        energy_from_battery if net_energy < 0 else 0, 3
                    ),
                    "estado_carga_bateria_kwh": round(battery_soc, 3),
                    "estado_carga_bateria_pct": round(
                        (battery_soc / battery_capacity) * 100, 1
                    ),
                    "energia_red_inyectada_kwh": round(grid_injection, 3),
                    "energia_red_consumida_kwh": round(grid_consumption, 3),
                    "porcentaje_autosuficiencia": round(autosuficiencia, 1),
                    "factor_utilizacion_bateria": round(
                        (energy_to_battery + energy_from_battery) / battery_capacity, 3
                    ),
                }
            )

    return pd.DataFrame(balance_data)


# 6. Análisis económico y ambiental
def generate_economics_data(df_systems, df_balance):
    # Parámetros económicos para Ecuador
    electricity_cost = 0.10  # USD/kWh
    co2_factor = 0.385  # kg CO2/kWh
    diesel_cost = 1.05  # USD/litro
    diesel_efficiency = 3.5  # kWh/litro

    # Costos de inversión
    cost_data = []
    for _, system in df_systems.iterrows():
        # Costos unitarios (USD)
        solar_cost_per_kw = np.random.uniform(900, 1300)
        wind_cost_per_kw = np.random.uniform(1600, 2200)
        battery_cost_per_kwh = 600 if system["tipo_bateria"] == "litio" else 350
        inverter_cost_per_kw = np.random.uniform(200, 400)

        # Inversiones
        solar_investment = system["capacidad_solar_kWp"] * solar_cost_per_kw
        wind_investment = system["capacidad_eolica_kW"] * wind_cost_per_kw
        battery_investment = system["capacidad_bateria_kWh"] * battery_cost_per_kwh
        inverter_investment = system["capacidad_inversor_kW"] * inverter_cost_per_kw

        # Costos adicionales
        installation_cost = 0.15 * (
            solar_investment
            + wind_investment
            + battery_investment
            + inverter_investment
        )
        engineering_cost = 0.08 * (
            solar_investment
            + wind_investment
            + battery_investment
            + inverter_investment
        )
        contingency = 0.05 * (
            solar_investment
            + wind_investment
            + battery_investment
            + inverter_investment
        )

        total_investment = (
            solar_investment
            + wind_investment
            + battery_investment
            + inverter_investment
            + installation_cost
            + engineering_cost
            + contingency
        )

        # Costos anuales O&M
        annual_om = total_investment * np.random.uniform(0.02, 0.035)

        cost_data.append(
            {
                "escuela": system["escuela"],
                "inversion_solar_usd": round(solar_investment, 2),
                "inversion_eolica_usd": round(wind_investment, 2),
                "inversion_baterias_usd": round(battery_investment, 2),
                "inversion_inversor_usd": round(inverter_investment, 2),
                "costo_instalacion_usd": round(installation_cost, 2),
                "costo_ingenieria_usd": round(engineering_cost, 2),
                "contingencia_usd": round(contingency, 2),
                "inversion_total_usd": round(total_investment, 2),
                "costo_om_anual_usd": round(annual_om, 2),
                "costo_unitario_usd_kw": round(
                    total_investment
                    / (system["capacidad_solar_kWp"] + system["capacidad_eolica_kW"]),
                    2,
                ),
                "payback_simple_años": 0,  # Se calculará después
                "tir_pct": 0,  # Se calculará después
                "van_usd": 0,  # Se calculará después
            }
        )

    df_costs = pd.DataFrame(cost_data)

    # Análisis operacional diario
    operational_data = []
    for _, balance in df_balance.iterrows():
        grid_energy = balance["energia_red_consumida_kwh"]
        injected_energy = balance["energia_red_inyectada_kwh"]
        generation = balance["generacion_total_kwh"]

        # Costos y beneficios
        grid_cost = grid_energy * electricity_cost
        injection_benefit = (
            injected_energy * electricity_cost * 0.8
        )  # 80% del precio de venta

        # Costo equivalente en diesel
        diesel_liters = generation / diesel_efficiency
        diesel_cost_equivalent = diesel_liters * diesel_cost

        # Emisiones
        co2_avoided = generation * co2_factor
        co2_emitted = grid_energy * co2_factor

        operational_data.append(
            {
                "fecha": balance["fecha"],
                "escuela": balance["escuela"],
                "costo_energia_red_usd": round(grid_cost, 3),
                "beneficio_inyeccion_red_usd": round(injection_benefit, 3),
                "ahorro_neto_diario_usd": round(injection_benefit - grid_cost, 3),
                "costo_diesel_equivalente_usd": round(diesel_cost_equivalent, 3),
                "ahorro_vs_diesel_usd": round(diesel_cost_equivalent - grid_cost, 3),
                "co2_evitado_kg": round(co2_avoided, 3),
                "co2_emitido_kg": round(co2_emitted, 3),
                "co2_neto_evitado_kg": round(co2_avoided - co2_emitted, 3),
                "diesel_equivalente_litros": round(diesel_liters, 2),
            }
        )

    df_operational = pd.DataFrame(operational_data)

    return df_costs, df_operational


# Función principal para generar todos los datos
def generate_all_data():
    print("Generando datos climáticos...")
    df_climate = generate_climate_data()

    print("Generando datos de consumo...")
    df_consumption = generate_consumption_data()

    print("Generando datos de sistemas...")
    df_systems = generate_renewable_system_data()

    print("Generando datos de generación...")
    df_generation = generate_power_generation_data(df_climate, df_systems)

    print("Calculando balance energético...")
    df_balance = generate_energy_balance(df_consumption, df_generation, df_systems)

    print("Generando análisis económico...")
    df_costs, df_operational = generate_economics_data(df_systems, df_balance)

    # Crear información de las escuelas
    df_schools = pd.DataFrame(
        [
            {
                "escuela": name,
                "estudiantes": params["students"],
                "altitud_m": params["altitude"],
                "terreno": params["terrain"],
                "area_hectareas": params["area_hectares"],
                "latitud": params["lat"],
                "longitud": params["lon"],
                "nivel_infraestructura": params["infrastructure_level"],
                "tipo_institucion": params["type"],
                "provincia": "Chimborazo",
                "pais": "Ecuador",
            }
            for name, params in school_params.items()
        ]
    )

    return {
        "escuelas": df_schools,
        "clima": df_climate,
        "consumo": df_consumption,
        "sistemas": df_systems,
        "generacion": df_generation,
        "balance": df_balance,
        "costos": df_costs,
        "operacional": df_operational,
    }


# Función para calcular métricas avanzadas
def calculate_advanced_metrics(df_balance, df_costs):
    """Calcula métricas avanzadas como TIR, VAN, y payback"""

    # Agrupar datos anuales para análisis financiero
    df_balance["año"] = df_balance["fecha"].dt.year
    annual_savings = (
        df_balance.groupby(["escuela", "año"])
        .agg(
            {
                "energia_red_consumida_kwh": "sum",
                "energia_red_inyectada_kwh": "sum",
                "generacion_total_kwh": "sum",
            }
        )
        .reset_index()
    )

    # Parámetros financieros
    electricity_cost = 0.10  # USD/kWh
    injection_price = 0.08  # USD/kWh
    discount_rate = 0.08  # 8% tasa de descuento
    project_life = 20  # años

    advanced_metrics = []

    for school in school_params.keys():
        school_costs = df_costs[df_costs["escuela"] == school].iloc[0]
        school_annual = annual_savings[annual_savings["escuela"] == school]

        if len(school_annual) > 0:
            # Calcular ahorro anual promedio
            avg_grid_consumption = school_annual["energia_red_consumida_kwh"].mean()
            avg_injection = school_annual["energia_red_inyectada_kwh"].mean()
            avg_generation = school_annual["generacion_total_kwh"].mean()

            annual_grid_cost = avg_grid_consumption * electricity_cost
            annual_injection_benefit = avg_injection * injection_price
            annual_fuel_savings = avg_generation * 0.30  # Estimación ahorro combustible

            total_annual_savings = (
                annual_injection_benefit - annual_grid_cost + annual_fuel_savings
            )

            # Calcular payback simple
            initial_investment = school_costs["inversion_total_usd"]
            annual_om = school_costs["costo_om_anual_usd"]
            net_annual_savings = total_annual_savings - annual_om

            payback_years = (
                initial_investment / net_annual_savings
                if net_annual_savings > 0
                else 999
            )

            # Calcular VAN (Valor Actual Neto)
            cash_flows = [-initial_investment]  # Año 0
            for year in range(1, project_life + 1):
                annual_cf = net_annual_savings * (0.98**year)  # Degradación 2% anual
                cash_flows.append(annual_cf)

            # VAN
            van = sum(
                [cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows)]
            )

            # TIR (aproximación)
            tir = 0
            for rate in np.arange(0, 0.5, 0.001):
                npv_test = sum(
                    [cf / (1 + rate) ** i for i, cf in enumerate(cash_flows)]
                )
                if npv_test <= 0:
                    tir = rate
                    break

            advanced_metrics.append(
                {
                    "escuela": school,
                    "ahorro_anual_promedio_usd": round(total_annual_savings, 2),
                    "costo_om_anual_usd": round(annual_om, 2),
                    "ahorro_neto_anual_usd": round(net_annual_savings, 2),
                    "payback_simple_años": round(payback_years, 1),
                    "van_usd": round(van, 2),
                    "tir_pct": round(tir * 100, 1),
                    "energia_generada_anual_kwh": round(avg_generation, 1),
                    "factor_capacidad_promedio": round(
                        avg_generation / (8760 * 10), 3
                    ),  # Estimación
                    "co2_evitado_anual_kg": round(avg_generation * 0.385, 1),
                }
            )

    return pd.DataFrame(advanced_metrics)


# Función para crear resumen ejecutivo
def create_executive_summary(data_dict):
    """Crea un resumen ejecutivo con KPIs principales"""

    df_balance = data_dict["balance"]
    df_systems = data_dict["sistemas"]
    df_schools = data_dict["escuelas"]

    # KPIs por escuela
    summary_data = []

    for school in school_params.keys():
        school_balance = df_balance[df_balance["escuela"] == school]
        school_system = df_systems[df_systems["escuela"] == school].iloc[0]
        school_info = df_schools[df_schools["escuela"] == school].iloc[0]

        # Métricas anuales (últimos 12 meses)
        recent_data = school_balance[school_balance["fecha"] >= "2024-06-30"]

        if len(recent_data) > 0:
            total_generation = recent_data["generacion_total_kwh"].sum()
            total_consumption = recent_data["consumo_diario_kwh"].sum()
            avg_autosuficiencia = recent_data["porcentaje_autosuficiencia"].mean()
            total_grid_consumption = recent_data["energia_red_consumida_kwh"].sum()
            total_injection = recent_data["energia_red_inyectada_kwh"].sum()

            # Costos evitados
            grid_cost_avoided = total_generation * 0.10
            co2_avoided = total_generation * 0.385

            summary_data.append(
                {
                    "escuela": school,
                    "estudiantes": school_info["estudiantes"],
                    "capacidad_instalada_kw": round(
                        school_system["capacidad_solar_kWp"]
                        + school_system["capacidad_eolica_kW"],
                        1,
                    ),
                    "generacion_anual_kwh": round(total_generation, 1),
                    "consumo_anual_kwh": round(total_consumption, 1),
                    "autosuficiencia_promedio_pct": round(avg_autosuficiencia, 1),
                    "energia_red_consumida_kwh": round(total_grid_consumption, 1),
                    "energia_red_inyectada_kwh": round(total_injection, 1),
                    "ahorro_energetico_usd": round(grid_cost_avoided, 2),
                    "co2_evitado_kg": round(co2_avoided, 1),
                    "factor_capacidad_pct": round(
                        (
                            total_generation
                            / (school_system["capacidad_solar_kWp"] * 8760)
                        )
                        * 100,
                        1,
                    ),
                    "inversion_total_usd": school_system["inversion_total_usd"],
                    "costo_por_estudiante_usd": round(
                        school_system["inversion_total_usd"]
                        / school_info["estudiantes"],
                        2,
                    ),
                }
            )

    return pd.DataFrame(summary_data)


# Función para exportar todos los datos a Excel
def export_to_excel(data_dict, filename="simulacion_energia_renovable_chimborazo.xlsx"):
    """Exporta todos los DataFrames a un archivo Excel con múltiples hojas"""

    print(f"Exportando datos a {filename}...")

    # Calcular métricas avanzadas
    df_advanced = calculate_advanced_metrics(data_dict["balance"], data_dict["costos"])
    df_summary = create_executive_summary(data_dict)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Hoja de resumen ejecutivo
        df_summary.to_excel(writer, sheet_name="Resumen_Ejecutivo", index=False)

        # Información de las escuelas
        data_dict["escuelas"].to_excel(writer, sheet_name="Escuelas", index=False)

        # Datos climáticos (muestra los últimos 365 días para reducir tamaño)
        recent_climate = data_dict["clima"][data_dict["clima"]["fecha"] >= "2024-06-30"]
        recent_climate.to_excel(writer, sheet_name="Datos_Climaticos", index=False)

        # Sistemas de energía renovable
        data_dict["sistemas"].to_excel(
            writer, sheet_name="Sistemas_Renovables", index=False
        )

        # Balance energético (últimos 365 días)
        recent_balance = data_dict["balance"][
            data_dict["balance"]["fecha"] >= "2024-06-30"
        ]
        recent_balance.to_excel(writer, sheet_name="Balance_Energetico", index=False)

        # Análisis económico
        data_dict["costos"].to_excel(writer, sheet_name="Costos_Inversion", index=False)
        df_advanced.to_excel(writer, sheet_name="Metricas_Financieras", index=False)

        # Datos operacionales (últimos 365 días)
        recent_operational = data_dict["operacional"][
            data_dict["operacional"]["fecha"] >= "2024-06-30"
        ]
        recent_operational.to_excel(
            writer, sheet_name="Analisis_Operacional", index=False
        )

        # Consumo detallado (muestra de 30 días)
        sample_consumption = data_dict["consumo"][
            (data_dict["consumo"]["fecha"] >= "2024-12-01")
            & (data_dict["consumo"]["fecha"] < "2024-12-31")
        ]
        sample_consumption.to_excel(writer, sheet_name="Consumo_Detallado", index=False)

        # Generación detallada (últimos 365 días)
        recent_generation = data_dict["generacion"][
            data_dict["generacion"]["fecha"] >= "2024-06-30"
        ]
        recent_generation.to_excel(
            writer, sheet_name="Generacion_Renovable", index=False
        )

    print(f"✅ Archivo {filename} creado exitosamente!")
    print(
        f"📊 Contiene {len(writer.sheets)} hojas con datos completos para análisis de ML"
    )


# Función principal
def main():
    print(
        "🌞 Iniciando simulación de sistemas híbridos para centros educativos rurales"
    )
    print("📍 Ubicación: Provincia de Chimborazo, Ecuador")
    print(
        f"📅 Período: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
    )
    print(f"🏫 Instituciones: {len(school_params)}")
    print("-" * 70)

    # Generar todos los datos
    data_dict = generate_all_data()

    # Mostrar estadísticas básicas
    print("\n📈 Estadísticas generadas:")
    for key, df in data_dict.items():
        print(f"  • {key.capitalize()}: {len(df):,} registros")

    # Exportar a Excel
    print("\n💾 Exportando datos...")
    export_to_excel(data_dict)

    print("\n✅ Proceso completado exitosamente!")
    print("\n📋 Estructura del archivo Excel:")
    sheets_info = [
        "• Resumen_Ejecutivo: KPIs principales por escuela",
        "• Escuelas: Información general de las instituciones",
        "• Datos_Climaticos: Variables meteorológicas (último año)",
        "• Sistemas_Renovables: Especificaciones técnicas",
        "• Balance_Energetico: Análisis diario de energía (último año)",
        "• Costos_Inversion: Análisis de inversión inicial",
        "• Metricas_Financieras: TIR, VAN, payback",
        "• Analisis_Operacional: Costos y beneficios diarios",
        "• Consumo_Detallado: Patrones horarios (muestra)",
        "• Generacion_Renovable: Producción solar y eólica",
    ]

    for info in sheets_info:
        print(f"  {info}")

    print(f"\n🎯 Listo para entrenar modelos de Machine Learning")
    print(f"💡 Variables objetivo sugeridas:")
    print(f"  - Predicción de generación renovable")
    print(f"  - Optimización de dimensionamiento")
    print(f"  - Pronóstico de consumo energético")
    print(f"  - Análisis de viabilidad económica")


# Ejecutar el programa
if __name__ == "__main__":
    main()
