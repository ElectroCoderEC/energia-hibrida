#!/usr/bin/env python3
"""
Simulador de Datos Sintéticos para Centros Educativos Rurales
Provincia de Chimborazo - Ecuador

Genera datos coherentes para sistemas híbridos solar-eólicos
considerando características geográficas y climáticas reales de la región.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

metadata = None

# Configuración de la simulación
np.random.seed(42)  # Para reproducibilidad
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 6, 30)
dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]

# Parámetros económicos actualizados Ecuador 2025
ELECTRICITY_COST = 0.10  # USD/kWh actualizado
CO2_FACTOR_ECUADOR = 0.385  # kg CO2/kWh factor de emisión Ecuador

# Parámetros para centros educativos rurales de Chimborazo
# Basado en ubicaciones reales de la provincia
school_params = {
    "UE Yaruquíes": {
        "size": "medium",
        "students": 180,
        "teachers": 12,
        "altitude": 2850,
        "terrain": "valle",
        "canton": "Riobamba",
        "area_terreno_m2": 2500,
        "area_construida_m2": 800,
        "tipo_institucion": "unidad_educativa",
        "jornada": "matutina",
        "niveles": ["inicial", "basica", "bachillerato"],
        "laboratorios": 2,
        "biblioteca": True,
        "cocina": True,
        "internet": True,
        "coord_lat": -1.6342,
        "coord_lon": -78.6407,
    },
    "Escuela San Andrés": {
        "size": "small",
        "students": 85,
        "teachers": 6,
        "altitude": 3200,
        "terrain": "montañoso",
        "canton": "Guano",
        "area_terreno_m2": 1200,
        "area_construida_m2": 400,
        "tipo_institucion": "escuela",
        "jornada": "matutina",
        "niveles": ["inicial", "basica"],
        "laboratorios": 0,
        "biblioteca": False,
        "cocina": True,
        "internet": False,
        "coord_lat": -1.6089,
        "coord_lon": -78.6308,
    },
    "Colegio Técnico Chambo": {
        "size": "large",
        "students": 320,
        "teachers": 22,
        "altitude": 2650,
        "terrain": "planicie",
        "canton": "Chambo",
        "area_terreno_m2": 4000,
        "area_construida_m2": 1500,
        "tipo_institucion": "colegio_tecnico",
        "jornada": "completa",
        "niveles": ["basica", "bachillerato_tecnico"],
        "laboratorios": 4,
        "biblioteca": True,
        "cocina": True,
        "internet": True,
        "coord_lat": -1.7200,
        "coord_lon": -78.5800,
    },
    "Escuela Intercultural Guarguallá": {
        "size": "small",
        "students": 65,
        "teachers": 4,
        "altitude": 3800,
        "terrain": "montañoso",
        "canton": "Guamote",
        "area_terreno_m2": 800,
        "area_construida_m2": 300,
        "tipo_institucion": "escuela_intercultural",
        "jornada": "matutina",
        "niveles": ["inicial", "basica"],
        "laboratorios": 0,
        "biblioteca": False,
        "cocina": True,
        "internet": False,
        "coord_lat": -2.0500,
        "coord_lon": -78.7200,
    },
    "UE San Juan": {
        "size": "medium",
        "students": 145,
        "teachers": 10,
        "altitude": 3100,
        "terrain": "valle",
        "canton": "Riobamba",
        "area_terreno_m2": 1800,
        "area_construida_m2": 650,
        "tipo_institucion": "unidad_educativa",
        "jornada": "matutina",
        "niveles": ["inicial", "basica", "bachillerato"],
        "laboratorios": 1,
        "biblioteca": True,
        "cocina": True,
        "internet": True,
        "coord_lat": -1.6500,
        "coord_lon": -78.6200,
    },
    "Escuela Comunitaria Punín": {
        "size": "small",
        "students": 45,
        "teachers": 3,
        "altitude": 3400,
        "terrain": "montañoso",
        "canton": "Riobamba",
        "area_terreno_m2": 600,
        "area_construida_m2": 200,
        "tipo_institucion": "escuela_comunitaria",
        "jornada": "matutina",
        "niveles": ["inicial", "basica"],
        "laboratorios": 0,
        "biblioteca": False,
        "cocina": False,
        "internet": False,
        "coord_lat": -1.7800,
        "coord_lon": -78.7100,
    },
}

# Crear directorio para guardar datos
if not os.path.exists("datos_simulacion_chimborazo"):
    os.makedirs("datos_simulacion_chimborazo")


def generate_climate_data():
    """
    Genera datos climáticos coherentes con las condiciones de Chimborazo
    Considera patrones estacionales ecuatoriales y efectos de altitud
    """
    climate_data = []

    for school_name, params in school_params.items():
        # Parámetros base según ubicación específica de Chimborazo
        base_radiation = (
            4.2 + (params["altitude"] - 2500) * 0.0004
        )  # Mayor radiación en altura

        # Velocidad del viento según topografía de Chimborazo
        if params["terrain"] == "montañoso":
            base_wind = 5.2 if params["altitude"] > 3000 else 4.5
        elif params["terrain"] == "valle":
            base_wind = 3.2
        else:  # planicie
            base_wind = 4.0

        # Factores de corrección por cantón (microclimas)
        canton_factors = {
            "Riobamba": {"temp": 0, "wind": 1.0, "radiation": 1.0},
            "Guano": {"temp": -1, "wind": 1.1, "radiation": 1.05},
            "Chambo": {"temp": 1, "wind": 0.9, "radiation": 0.98},
            "Guamote": {"temp": -2, "wind": 1.2, "radiation": 1.1},
        }

        canton_factor = canton_factors.get(
            params["canton"], {"temp": 0, "wind": 1.0, "radiation": 1.0}
        )

        for day, date in enumerate(dates):
            # Variación estacional ecuatorial (dos picos de lluvia)
            day_of_year = date.timetuple().tm_yday

            # Estaciones en Ecuador: seca (junio-septiembre), húmeda (octubre-mayo)
            dry_season = 6 <= date.month <= 9

            # Radiación solar (mayor en temporada seca)
            seasonal_radiation = 1.2 if dry_season else 0.9
            daily_radiation = base_radiation * seasonal_radiation * canton_factor[
                "radiation"
            ] + np.random.normal(0, 0.6)
            daily_radiation = max(daily_radiation, 1.0)

            # Viento (más fuerte en temporada seca)
            seasonal_wind = 1.15 if dry_season else 0.85
            daily_wind = base_wind * seasonal_wind * canton_factor[
                "wind"
            ] + np.random.normal(0, 1.2)
            daily_wind = max(daily_wind, 0.5)

            # Temperatura (efecto de altitud más pronunciado)
            temp_base = (
                20 - (params["altitude"] - 2500) * 0.0065
            )  # Gradiente térmico real
            temp_base += canton_factor["temp"]
            seasonal_temp = 2 if dry_season else -1
            daily_temp = temp_base + seasonal_temp + np.random.normal(0, 2.5)

            # Humedad relativa (mayor en temporada húmeda)
            humidity = (
                np.random.uniform(60, 80)
                if not dry_season
                else np.random.uniform(40, 65)
            )

            # Precipitación (patrón bimodal ecuatorial)
            rain_months = [3, 4, 5, 10, 11, 12]  # Meses más lluviosos
            rain_prob = 0.4 if date.month in rain_months else 0.15
            precipitation = (
                np.random.exponential(8) if np.random.random() < rain_prob else 0
            )

            # Nubosidad correlacionada con precipitación
            if precipitation > 0:
                cloud_cover = np.random.uniform(0.6, 0.9)
            else:
                cloud_cover = np.random.uniform(0.1, 0.5)

            # Presión atmosférica (función de altitud)
            pressure = 1013.25 * (1 - 0.0065 * params["altitude"] / 288.15) ** 5.255

            climate_data.append(
                {
                    "fecha": date,
                    "escuela": school_name,
                    "radiacion_solar_kwh_m2": round(daily_radiation, 2),
                    "velocidad_viento_m_s": round(daily_wind, 2),
                    "temperatura_c": round(daily_temp, 1),
                    "humedad_relativa": round(humidity, 1),
                    "nubosidad": round(cloud_cover, 2),
                    "precipitacion_mm": round(precipitation, 1),
                    "presion_atmosferica_hpa": round(pressure, 1),
                    "horas_sol": round(max(0, 12 - 4 * cloud_cover), 1),
                }
            )

    return pd.DataFrame(climate_data)


def generate_consumption_data():
    """
    Genera patrones de consumo específicos para centros educativos rurales ecuatorianos
    """
    consumption_data = []

    for school_name, params in school_params.items():
        # Consumo base según características específicas
        base_consumption = calculate_base_consumption(params)

        # Patrones horarios según jornada
        if params["jornada"] == "matutina":
            active_hours = list(range(7, 13))  # 7:00 - 12:00
        elif params["jornada"] == "vespertina":
            active_hours = list(range(13, 18))  # 13:00 - 17:00
        else:  # completa
            active_hours = list(range(7, 17))  # 7:00 - 16:00

        # Factor de consumo por hora
        hourly_factors = create_hourly_consumption_pattern(params, active_hours)

        for date in dates:
            # Factores especiales
            is_weekend = date.weekday() >= 5
            is_holiday = is_school_holiday(date)
            is_exam_period = is_exam_time(date)

            # Factor de ocupación
            if is_weekend or is_holiday:
                occupancy_factor = 0.1  # Solo seguridad y mantenimiento
            elif is_exam_period:
                occupancy_factor = 1.2  # Mayor uso durante exámenes
            else:
                occupancy_factor = 1.0

            for hour in range(24):
                # Consumo horario
                hourly_consumption = (
                    base_consumption * hourly_factors[hour] * occupancy_factor
                )

                # Variabilidad aleatoria
                hourly_consumption *= np.random.uniform(0.85, 1.15)

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "consumo_kwh": round(max(0, hourly_consumption), 3),
                        "factor_ocupacion": round(occupancy_factor, 2),
                    }
                )

    return pd.DataFrame(consumption_data)


def calculate_base_consumption(params):
    """Calcula el consumo base según características de la institución"""
    # Consumo base por estudiante (kWh/día)
    consumption_per_student = 0.08 if params["internet"] else 0.05

    # Consumo por área construida
    consumption_per_m2 = 0.02  # kWh/m²/día

    # Consumo por equipamiento
    equipment_consumption = 0
    equipment_consumption += params["laboratorios"] * 2  # 2 kWh/día por laboratorio
    equipment_consumption += 1 if params["biblioteca"] else 0
    equipment_consumption += 3 if params["cocina"] else 0
    equipment_consumption += 2 if params["internet"] else 0

    total_consumption = (
        params["students"] * consumption_per_student
        + params["area_construida_m2"] * consumption_per_m2
        + equipment_consumption
    )

    return max(total_consumption, 2.0)  # Mínimo 2 kWh/día


def create_hourly_consumption_pattern(params, active_hours):
    """Crea patrón de consumo horario"""
    factors = np.zeros(24)

    # Horas activas
    for hour in active_hours:
        if hour in [7, 8]:  # Inicio de jornada
            factors[hour] = 0.8
        elif hour in [10, 11, 14, 15]:  # Horas pico
            factors[hour] = 1.0
        else:
            factors[hour] = 0.6

    # Consumo base nocturno (seguridad, refrigeración)
    for hour in range(24):
        if factors[hour] == 0:
            factors[hour] = 0.05

    # Ajuste por tipo de institución
    if params["tipo_institucion"] == "colegio_tecnico":
        factors *= 1.3  # Mayor consumo por talleres
    elif params["tipo_institucion"] == "escuela_comunitaria":
        factors *= 0.7  # Menor consumo

    return factors


def is_school_holiday(date):
    """Determina si es período de vacaciones escolares en Ecuador"""
    # Vacaciones de verano (costa): diciembre-marzo
    # Vacaciones de invierno (sierra): julio-agosto
    return date.month in [12, 1, 2, 7, 8]


def is_exam_time(date):
    """Determina si es época de exámenes"""
    # Exámenes quimestrales: febrero, julio, noviembre
    return date.month in [2, 7, 11] and date.day <= 15


def generate_renewable_system_data():
    """Genera especificaciones de sistemas renovables optimizados para Chimborazo"""
    systems_data = []

    for school_name, params in school_params.items():
        # Dimensionamiento según área disponible y demanda
        available_roof_area = (
            params["area_construida_m2"] * 0.7
        )  # 70% del techo utilizable
        available_ground_area = min(
            params["area_terreno_m2"] * 0.3, 500
        )  # Máximo 500 m²

        # Capacidad solar (considerando área disponible)
        solar_density = 6  # kWp/100m² (densidad realista)
        max_solar_capacity = (
            (available_roof_area + available_ground_area) * solar_density / 100
        )

        # Dimensionar según tamaño y necesidades
        if params["size"] == "small":
            solar_capacity = min(np.random.uniform(3, 8), max_solar_capacity)
            wind_capacity = np.random.uniform(1, 3)
            battery_capacity = np.random.uniform(8, 15)
        elif params["size"] == "medium":
            solar_capacity = min(np.random.uniform(8, 15), max_solar_capacity)
            wind_capacity = np.random.uniform(3, 6)
            battery_capacity = np.random.uniform(15, 25)
        else:  # large
            solar_capacity = min(np.random.uniform(15, 25), max_solar_capacity)
            wind_capacity = np.random.uniform(6, 12)
            battery_capacity = np.random.uniform(25, 45)

        # Ajuste por altitud (mejor para eólico, menor densidad aire)
        if params["altitude"] > 3000:
            wind_capacity *= 1.2  # Más viento en altura
            solar_capacity *= 1.1  # Mejor radiación

        # Especificaciones técnicas
        solar_efficiency = np.random.uniform(0.18, 0.22)  # Paneles modernos
        wind_efficiency = np.random.uniform(0.30, 0.45)
        battery_efficiency = np.random.uniform(0.90, 0.96)

        # Edad y degradación
        system_age = np.random.randint(0, 3)  # Sistemas relativamente nuevos
        solar_degradation = 0.005 * system_age  # 0.5% anual
        battery_degradation = 0.015 * system_age  # 1.5% anual

        # Tecnologías apropiadas para la región
        panel_types = ["monocristalino", "policristalino"]
        panel_weights = [0.7, 0.3]  # Preferencia por monocristalino

        battery_types = ["litio", "plomo-ácido"]
        battery_weights = [0.6, 0.4] if params["size"] != "small" else [0.3, 0.7]

        wind_types = ["eje_horizontal", "eje_vertical"]
        wind_weights = [0.8, 0.2]  # Preferencia por eje horizontal

        # Inversor (crucial para sistemas híbridos)
        inverter_capacity = max(solar_capacity, wind_capacity) * 1.2
        inverter_efficiency = np.random.uniform(0.94, 0.98)

        systems_data.append(
            {
                "escuela": school_name,
                "capacidad_solar_kWp": round(solar_capacity, 2),
                "capacidad_eolica_kW": round(wind_capacity, 2),
                "capacidad_bateria_kWh": round(battery_capacity, 2),
                "capacidad_inversor_kW": round(inverter_capacity, 2),
                "eficiencia_solar": round(solar_efficiency, 3),
                "eficiencia_eolica": round(wind_efficiency, 3),
                "eficiencia_bateria": round(battery_efficiency, 3),
                "eficiencia_inversor": round(inverter_efficiency, 3),
                "edad_sistema_años": system_age,
                "degradacion_solar_anual": round(solar_degradation, 4),
                "degradacion_bateria_anual": round(battery_degradation, 4),
                "tipo_panel": np.random.choice(panel_types, p=panel_weights),
                "tipo_bateria": np.random.choice(battery_types, p=battery_weights),
                "tipo_aerogenerador": np.random.choice(wind_types, p=wind_weights),
                "area_paneles_m2": round(solar_capacity * 6, 1),  # ~6 m²/kWp
                "area_disponible_techo_m2": round(available_roof_area, 1),
                "area_disponible_terreno_m2": round(available_ground_area, 1),
                "factor_inclinacion_paneles": round(abs(params["coord_lat"]), 1),
                "orientacion_paneles": "sur",
                "altura_aerogenerador_m": round(np.random.uniform(8, 15), 1),
                "controlador_carga": "MPPT" if solar_capacity > 5 else "PWM",
            }
        )

    return pd.DataFrame(systems_data)


def generate_power_generation_data(df_climate, df_systems):
    """Genera datos de generación con modelos físicos mejorados"""
    generation_data = []

    # Agrupar datos climáticos
    climate_grouped = df_climate.groupby(["escuela", "fecha"])

    for (school_name, date), climate_group in climate_grouped:
        system_params = df_systems[df_systems["escuela"] == school_name].iloc[0]
        climate_day = climate_group.iloc[0]

        # Generación solar mejorada
        solar_gen = calculate_solar_generation(system_params, climate_day)

        # Generación eólica mejorada
        wind_gen = calculate_wind_generation(system_params, climate_day)

        # Eficiencia del inversor
        inverter_eff = system_params["eficiencia_inversor"]
        total_gen = (solar_gen + wind_gen) * inverter_eff

        generation_data.append(
            {
                "fecha": date,
                "escuela": school_name,
                "generacion_solar_kwh": round(max(0, solar_gen), 2),
                "generacion_eolica_kwh": round(max(0, wind_gen), 2),
                "generacion_total_kwh": round(max(0, total_gen), 2),
                "eficiencia_sistema": round(inverter_eff, 3),
                "factor_planta_solar": round(
                    (
                        solar_gen / (system_params["capacidad_solar_kWp"] * 24)
                        if system_params["capacidad_solar_kWp"] > 0
                        else 0
                    ),
                    3,
                ),
                "factor_planta_eolico": round(
                    (
                        wind_gen / (system_params["capacidad_eolica_kW"] * 24)
                        if system_params["capacidad_eolica_kW"] > 0
                        else 0
                    ),
                    3,
                ),
            }
        )

    return pd.DataFrame(generation_data)


def calculate_solar_generation(system_params, climate_data):
    """Modelo físico mejorado para generación solar"""
    capacity = system_params["capacidad_solar_kWp"]
    efficiency = system_params["eficiencia_solar"] * (
        1 - system_params["degradacion_solar_anual"]
    )
    radiation = climate_data["radiacion_solar_kwh_m2"]
    temperature = climate_data["temperatura_c"]
    cloud_cover = climate_data["nubosidad"]

    # Efecto de temperatura (coeficiente real -0.004/°C)
    temp_coefficient = -0.004
    temp_factor = 1 + temp_coefficient * (temperature - 25)

    # Efecto de nubosidad (modelo logarítmico)
    cloud_factor = 1 - 0.75 * cloud_cover

    # Horas de sol efectivas
    sun_hours = climate_data["horas_sol"]

    # Generación diaria
    daily_generation = (
        capacity * radiation * efficiency * temp_factor * cloud_factor * sun_hours / 12
    )

    return max(0, daily_generation)


def calculate_wind_generation(system_params, climate_data):
    """Modelo físico mejorado para generación eólica"""
    capacity = system_params["capacidad_eolica_kW"]
    efficiency = system_params["eficiencia_eolica"]
    wind_speed = climate_data["velocidad_viento_m_s"]
    pressure = climate_data["presion_atmosferica_hpa"]

    # Corrección por densidad del aire (altitud)
    air_density_factor = pressure / 1013.25

    # Curva de potencia mejorada
    cut_in = 2.5  # m/s
    rated_speed = 12.0  # m/s
    cut_out = 25.0  # m/s

    if wind_speed < cut_in or wind_speed > cut_out:
        power_factor = 0
    elif wind_speed < rated_speed:
        # Curva cúbica hasta velocidad nominal
        power_factor = ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3
    else:
        # Potencia constante hasta cut-out
        power_factor = 1.0

    # Generación diaria (asumiendo 24 horas de operación)
    daily_generation = capacity * power_factor * efficiency * air_density_factor * 24

    return max(0, daily_generation)


def generate_energy_balance(df_consumption, df_generation, df_systems):
    """Genera balance energético incluyendo gestión de baterías"""
    # Consumo diario
    daily_consumption = (
        df_consumption.groupby(["fecha", "escuela"])["consumo_kwh"].sum().reset_index()
    )
    daily_consumption.rename(
        columns={"consumo_kwh": "consumo_diario_kwh"}, inplace=True
    )

    # Unir con generación
    df_balance = pd.merge(df_generation, daily_consumption, on=["fecha", "escuela"])

    # Agregar capacidad de batería
    df_balance = pd.merge(
        df_balance,
        df_systems[["escuela", "capacidad_bateria_kWh", "eficiencia_bateria"]],
        on="escuela",
    )

    # Simular gestión de baterías (simplificado)
    balance_data = []

    for school in df_balance["escuela"].unique():
        school_data = df_balance[df_balance["escuela"] == school].sort_values("fecha")
        battery_capacity = school_data["capacidad_bateria_kWh"].iloc[0]
        battery_efficiency = school_data["eficiencia_bateria"].iloc[0]
        battery_charge = battery_capacity * 0.5  # Estado inicial 50%

        for _, row in school_data.iterrows():
            generation = row["generacion_total_kwh"]
            consumption = row["consumo_diario_kwh"]
            net_energy = generation - consumption

            # Gestión de batería
            if net_energy > 0:  # Excedente
                # Cargar batería
                max_charge = min(
                    net_energy * battery_efficiency, battery_capacity - battery_charge
                )
                battery_charge += max_charge
                excess_energy = net_energy - max_charge / battery_efficiency
                grid_injection = excess_energy
                grid_consumption = 0
            else:  # Déficit
                deficit = -net_energy
                # Descargar batería
                max_discharge = min(battery_charge * battery_efficiency, deficit)
                battery_charge -= max_discharge / battery_efficiency
                remaining_deficit = deficit - max_discharge
                grid_consumption = remaining_deficit
                grid_injection = 0
                excess_energy = 0

            # Degradación diaria de batería (muy pequeña)
            battery_charge *= 0.9999  # 0.01% diario

            balance_data.append(
                {
                    "fecha": row["fecha"],
                    "escuela": row["escuela"],
                    "generacion_total_kwh": row["generacion_total_kwh"],
                    "consumo_diario_kwh": row["consumo_diario_kwh"],
                    "balance_neto_kwh": round(net_energy, 2),
                    "estado_bateria_kwh": round(battery_charge, 2),
                    "excedente_red_kwh": round(grid_injection, 2),
                    "consumo_red_kwh": round(grid_consumption, 2),
                    "autosuficiencia_pct": round(
                        min(100, (generation + max_discharge) / consumption * 100), 1
                    ),
                    "soc_bateria_pct": round(
                        battery_charge / battery_capacity * 100, 1
                    ),
                }
            )

    return pd.DataFrame(balance_data)


def generate_economics_data(df_systems, df_balance):
    """Genera análisis económico detallado"""
    # Costos de inversión actualizados Ecuador 2025
    cost_factors = {
        "solar_usd_per_kWp": np.random.uniform(800, 1200),
        "wind_usd_per_kW": np.random.uniform(1500, 2200),
        "battery_litio_usd_per_kWh": np.random.uniform(600, 900),
        "battery_plomo_usd_per_kWh": np.random.uniform(200, 350),
        "inverter_usd_per_kW": np.random.uniform(300, 500),
        "installation_factor": 0.25,  # 25% del costo de equipos
        "om_factor": 0.03,  # 3% anual de la inversión
    }

    # Datos de inversión
    investment_data = []
    for _, system in df_systems.iterrows():
        solar_cost = system["capacidad_solar_kWp"] * cost_factors["solar_usd_per_kWp"]
        wind_cost = system["capacidad_eolica_kW"] * cost_factors["wind_usd_per_kW"]

        if system["tipo_bateria"] == "litio":
            battery_cost = (
                system["capacidad_bateria_kWh"]
                * cost_factors["battery_litio_usd_per_kWh"]
            )
        else:
            battery_cost = (
                system["capacidad_bateria_kWh"]
                * cost_factors["battery_plomo_usd_per_kWh"]
            )

        inverter_cost = (
            system["capacidad_inversor_kW"] * cost_factors["inverter_usd_per_kW"]
        )
        equipment_cost = solar_cost + wind_cost + battery_cost + inverter_cost
        installation_cost = equipment_cost * cost_factors["installation_factor"]
        total_investment = equipment_cost + installation_cost

        annual_om = total_investment * cost_factors["om_factor"]

        # Vida útil por tecnología
        solar_life = 25
        wind_life = 20
        battery_life = 12 if system["tipo_bateria"] == "litio" else 8
        inverter_life = 15

        investment_data.append(
            {
                "escuela": system["escuela"],
                "inversion_solar_usd": round(solar_cost, 2),
                "inversion_eolica_usd": round(wind_cost, 2),
                "inversion_baterias_usd": round(battery_cost, 2),
                "inversion_inversor_usd": round(inverter_cost, 2),
                "inversion_instalacion_usd": round(installation_cost, 2),
                "inversion_total_usd": round(total_investment, 2),
                "costo_om_anual_usd": round(annual_om, 2),
                "vida_util_solar_años": solar_life,
                "vida_util_eolica_años": wind_life,
                "vida_util_baterias_años": battery_life,
                "vida_util_inversor_años": inverter_life,
                "lcoe_usd_per_kwh": round(
                    calculate_lcoe(total_investment, annual_om, system), 4
                ),
            }
        )

    df_investment = pd.DataFrame(investment_data)

    # Datos operacionales diarios
    operational_data = []
    for _, balance in df_balance.iterrows():
        grid_cost = balance["consumo_red_kwh"] * ELECTRICITY_COST

        # Ingresos por inyección a red (si aplica)
        injection_revenue = (
            balance["excedente_red_kwh"] * ELECTRICITY_COST * 0.7
        )  # 70% del precio

        # Emisiones
        co2_avoided = balance["generacion_total_kwh"] * CO2_FACTOR_ECUADOR
        co2_emitted = balance["consumo_red_kwh"] * CO2_FACTOR_ECUADOR

        # Ahorros diarios
        daily_savings = (
            balance["generacion_total_kwh"] - balance["excedente_red_kwh"]
        ) * ELECTRICITY_COST

        operational_data.append(
            {
                "fecha": balance["fecha"],
                "escuela": balance["escuela"],
                "costo_energia_red_usd": round(grid_cost, 3),
                "ingresos_inyeccion_usd": round(injection_revenue, 3),
                "ahorro_diario_usd": round(daily_savings, 3),
                "co2_evitado_kg": round(co2_avoided, 2),
                "co2_emitido_kg": round(co2_emitted, 2),
                "co2_neto_evitado_kg": round(co2_avoided - co2_emitted, 2),
            }
        )

    df_operational = pd.DataFrame(operational_data)

    return df_investment, df_operational


def calculate_lcoe(investment, annual_om, system_params):
    """Calcula el Costo Nivelado de Energía (LCOE)"""
    # Parámetros financieros
    discount_rate = 0.08  # 8% tasa de descuento
    system_life = 20  # años

    # Generación anual estimada (kWh)
    annual_generation = (
        system_params["capacidad_solar_kWp"] * 1800
        + system_params["capacidad_eolica_kW"] * 2200
    )  # Horas equivalentes

    # Valor presente neto de costos e ingresos
    npv_costs = investment
    npv_generation = 0

    for year in range(1, system_life + 1):
        # Costos anuales
        annual_cost = annual_om
        npv_costs += annual_cost / (1 + discount_rate) ** year

        # Generación anual (con degradación)
        degradation_factor = (1 - 0.005) ** year  # 0.5% degradación anual
        year_generation = annual_generation * degradation_factor
        npv_generation += year_generation / (1 + discount_rate) ** year

    return npv_costs / npv_generation if npv_generation > 0 else 0


def save_all_data():

    global metadata
    """Función principal para generar y guardar todos los datos"""
    print(
        "🌱 Generando datos sintéticos para centros educativos rurales de Chimborazo..."
    )

    # 1. Datos climáticos
    print("☀️ Generando datos climáticos...")
    df_climate = generate_climate_data()
    df_climate.to_csv("datos_simulacion_chimborazo/datos_climaticos.csv", index=False)

    # 2. Datos de consumo
    print("⚡ Generando datos de consumo energético...")
    df_consumption = generate_consumption_data()
    df_consumption.to_csv("datos_simulacion_chimborazo/datos_consumo.csv", index=False)

    # 3. Sistemas de energía renovable
    print("🔋 Generando especificaciones de sistemas renovables...")
    df_systems = generate_renewable_system_data()
    df_systems.to_csv(
        "datos_simulacion_chimborazo/sistemas_renovables.csv", index=False
    )

    # 4. Generación de energía
    print("🌬️ Calculando generación de energía...")
    df_generation = generate_power_generation_data(df_climate, df_systems)
    df_generation.to_csv(
        "datos_simulacion_chimborazo/generacion_energia.csv", index=False
    )

    # 5. Balance energético
    print("⚖️ Calculando balance energético...")
    df_balance = generate_energy_balance(df_consumption, df_generation, df_systems)
    df_balance.to_csv("datos_simulacion_chimborazo/balance_energetico.csv", index=False)

    # 6. Análisis económico
    print("💰 Generando análisis económico...")
    df_investment, df_operational = generate_economics_data(df_systems, df_balance)
    df_investment.to_csv(
        "datos_simulacion_chimborazo/inversion_costos.csv", index=False
    )
    df_operational.to_csv(
        "datos_simulacion_chimborazo/datos_operacionales.csv", index=False
    )

    # 7. Metadatos del proyecto
    print("📊 Guardando metadatos...")
    metadata = {
        "proyecto": "Simulación Sistemas Híbridos - Centros Educativos Rurales Chimborazo",
        "fecha_generacion": datetime.now().isoformat(),
        "periodo_simulacion": {
            "inicio": start_date.isoformat(),
            "fin": end_date.isoformat(),
            "dias_total": len(dates),
        },
        "parametros_economicos": {
            "costo_electricidad_usd_per_kwh": ELECTRICITY_COST,
            "factor_co2_kg_per_kwh": CO2_FACTOR_ECUADOR,
            "tasa_descuento": 0.08,
        },
        "centros_educativos": school_params,
        "archivos_generados": [
            "datos_climaticos.csv",
            "datos_consumo.csv",
            "sistemas_renovables.csv",
            "generacion_energia.csv",
            "balance_energetico.csv",
            "inversion_costos.csv",
            "datos_operacionales.csv",
        ],
    }

    with open("datos_simulacion_chimborazo/metadatos.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 8. Resumen estadístico
    print("📈 Generando resumen estadístico...")
    generate_summary_report(
        df_climate,
        df_consumption,
        df_generation,
        df_balance,
        df_investment,
        df_operational,
    )

    print("\n✅ ¡Simulación completada exitosamente!")
    print(f"📁 Archivos guardados en: datos_simulacion_chimborazo/")
    print(f"🏫 Centros educativos simulados: {len(school_params)}")
    print(
        f"📅 Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"📊 Total de registros generados: {len(df_balance):,}")


def generate_summary_report(
    df_climate, df_consumption, df_generation, df_balance, df_investment, df_operational
):

    global metadata
    """Genera un reporte resumen con estadísticas clave"""

    summary = {
        "RESUMEN EJECUTIVO": {
            "periodo_analisis": f"{start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
            "centros_educativos": len(school_params),
            "dias_simulados": len(dates),
            "provincia": "Chimborazo, Ecuador",
        },
        "ESTADÍSTICAS CLIMÁTICAS": {
            "radiacion_solar_promedio_kwh_m2": round(
                df_climate["radiacion_solar_kwh_m2"].mean(), 2
            ),
            "radiacion_solar_maxima_kwh_m2": round(
                df_climate["radiacion_solar_kwh_m2"].max(), 2
            ),
            "velocidad_viento_promedio_m_s": round(
                df_climate["velocidad_viento_m_s"].mean(), 2
            ),
            "temperatura_promedio_c": round(df_climate["temperatura_c"].mean(), 1),
            "precipitacion_anual_promedio_mm": round(
                df_climate["precipitacion_mm"].sum() / len(school_params) / 4.5, 0
            ),
        },
        "CONSUMO ENERGÉTICO": {
            "consumo_total_simulado_mwh": round(
                df_consumption["consumo_kwh"].sum() / 1000, 1
            ),
            "consumo_promedio_diario_kwh": round(
                df_balance["consumo_diario_kwh"].mean(), 1
            ),
            "consumo_maximo_diario_kwh": round(
                df_balance["consumo_diario_kwh"].max(), 1
            ),
            "consumo_minimo_diario_kwh": round(
                df_balance["consumo_diario_kwh"].min(), 1
            ),
        },
        "GENERACIÓN RENOVABLE": {
            "generacion_total_simulada_mwh": round(
                df_generation["generacion_total_kwh"].sum() / 1000, 1
            ),
            "contribucion_solar_pct": round(
                df_generation["generacion_solar_kwh"].sum()
                / df_generation["generacion_total_kwh"].sum()
                * 100,
                1,
            ),
            "contribucion_eolica_pct": round(
                df_generation["generacion_eolica_kwh"].sum()
                / df_generation["generacion_total_kwh"].sum()
                * 100,
                1,
            ),
            "factor_planta_solar_promedio": round(
                df_generation["factor_planta_solar"].mean(), 3
            ),
            "factor_planta_eolico_promedio": round(
                df_generation["factor_planta_eolico"].mean(), 3
            ),
        },
        "BALANCE ENERGÉTICO": {
            "autosuficiencia_promedio_pct": round(
                df_balance["autosuficiencia_pct"].mean(), 1
            ),
            "energia_excedente_total_mwh": round(
                df_balance["excedente_red_kwh"].sum() / 1000, 1
            ),
            "energia_red_requerida_mwh": round(
                df_balance["consumo_red_kwh"].sum() / 1000, 1
            ),
            "soc_promedio_baterias_pct": round(df_balance["soc_bateria_pct"].mean(), 1),
        },
        "ANÁLISIS ECONÓMICO": {
            "inversion_total_promedio_usd": round(
                df_investment["inversion_total_usd"].mean(), 0
            ),
            "inversion_total_sistema_usd": round(
                df_investment["inversion_total_usd"].sum(), 0
            ),
            "lcoe_promedio_usd_per_kwh": round(
                df_investment["lcoe_usd_per_kwh"].mean(), 4
            ),
            "ahorro_anual_estimado_usd": round(
                df_operational["ahorro_diario_usd"].sum() * 365 / len(dates), 0
            ),
            "co2_evitado_anual_toneladas": round(
                df_operational["co2_neto_evitado_kg"].sum() * 365 / len(dates) / 1000, 1
            ),
        },
        "CENTROS EDUCATIVOS POR TAMAÑO": {
            "pequeños": len(
                [s for s in school_params.values() if s["size"] == "small"]
            ),
            "medianos": len(
                [s for s in school_params.values() if s["size"] == "medium"]
            ),
            "grandes": len([s for s in school_params.values() if s["size"] == "large"]),
        },
        "DISTRIBUCIÓN ALTITUDINAL": {
            "rango_altitud_m": f"{min(s['altitude'] for s in school_params.values())} - {max(s['altitude'] for s in school_params.values())}",
            "altitud_promedio_m": round(
                sum(s["altitude"] for s in school_params.values()) / len(school_params),
                0,
            ),
        },
    }

    # Guardar resumen
    with open(
        "datos_simulacion_chimborazo/resumen_estadistico.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Crear reporte en texto
    with open(
        "datos_simulacion_chimborazo/REPORTE_RESUMEN.txt", "w", encoding="utf-8"
    ) as f:
        f.write("=" * 80 + "\n")
        f.write("SIMULACIÓN SISTEMAS HÍBRIDOS SOLAR-EÓLICO\n")
        f.write("CENTROS EDUCATIVOS RURALES - PROVINCIA DE CHIMBORAZO\n")
        f.write("=" * 80 + "\n\n")

        for section, data in summary.items():
            f.write(f"{section}:\n")
            f.write("-" * len(section) + "\n")
            for key, value in data.items():
                f.write(f"  • {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("ARCHIVOS GENERADOS:\n")
        f.write("-" * 20 + "\n")
        for archivo in metadata["archivos_generados"]:
            f.write(f"  📄 {archivo}\n")
        f.write("\n")

        f.write("CENTROS EDUCATIVOS SIMULADOS:\n")
        f.write("-" * 30 + "\n")
        for nombre, params in school_params.items():
            f.write(f"  🏫 {nombre}\n")
            f.write(f"     📍 {params['canton']}, {params['altitude']}m\n")
            f.write(f"     👥 {params['students']} estudiantes\n")
            f.write(f"     📐 {params['area_terreno_m2']}m² terreno\n\n")


# Función para ejecutar la simulación completa
if __name__ == "__main__":
    try:
        save_all_data()

        # Mostrar algunas estadísticas finales
        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS RÁPIDAS:")
        print("=" * 60)

        # Cargar algunos datos para mostrar estadísticas
        df_balance = pd.read_csv("datos_simulacion_chimborazo/balance_energetico.csv")
        df_investment = pd.read_csv("datos_simulacion_chimborazo/inversion_costos.csv")

        print(
            f"💡 Autosuficiencia energética promedio: {df_balance['autosuficiencia_pct'].mean():.1f}%"
        )
        print(
            f"💰 Inversión total requerida: ${df_investment['inversion_total_usd'].sum():,.0f} USD"
        )
        print(
            f"🌱 CO₂ evitado anualmente: {df_balance.groupby('escuela')['co2_neto_evitado_kg'].sum().sum() * 365 / len(dates) / 1000:.1f} toneladas"
        )
        print(
            f"⚡ Generación renovable total: {df_balance['generacion_total_kwh'].sum() / 1000:.1f} MWh"
        )

        print("\n🎯 Los datos están listos para su análisis y modelado!")
        print(
            "📁 Revisa la carpeta 'datos_simulacion_chimborazo' para todos los archivos generados."
        )

    except Exception as e:
        print(f"❌ Error durante la simulación: {str(e)}")
        print("🔍 Revisa los parámetros y dependencias del script.")


# Ejemplo de uso de los datos generados
def ejemplo_analisis_datos():
    """Función ejemplo para mostrar cómo usar los datos generados"""
    print("\n" + "=" * 50)
    print("📈 EJEMPLO DE ANÁLISIS DE DATOS GENERADOS")
    print("=" * 50)

    try:
        # Cargar datos
        df_balance = pd.read_csv("datos_simulacion_chimborazo/balance_energetico.csv")
        df_climate = pd.read_csv("datos_simulacion_chimborazo/datos_climaticos.csv")

        # Convertir fechas
        df_balance["fecha"] = pd.to_datetime(df_balance["fecha"])
        df_climate["fecha"] = pd.to_datetime(df_climate["fecha"])

        # Análisis por escuela
        print("\n🏫 ANÁLISIS POR CENTRO EDUCATIVO:")
        print("-" * 40)

        for escuela in df_balance["escuela"].unique():
            data_escuela = df_balance[df_balance["escuela"] == escuela]

            autosuf_promedio = data_escuela["autosuficiencia_pct"].mean()
            generacion_total = data_escuela["generacion_total_kwh"].sum()
            consumo_total = data_escuela["consumo_diario_kwh"].sum()

            print(f"\n📍 {escuela}:")
            print(f"   ⚖️  Autosuficiencia: {autosuf_promedio:.1f}%")
            print(f"   🔋 Generación total: {generacion_total:.0f} kWh")
            print(f"   ⚡ Consumo total: {consumo_total:.0f} kWh")
            print(f"   🌱 Balance: {generacion_total - consumo_total:.0f} kWh")

        # Análisis estacional
        print("\n📅 ANÁLISIS ESTACIONAL:")
        print("-" * 25)

        df_balance["mes"] = df_balance["fecha"].dt.month
        estacional = (
            df_balance.groupby("mes")
            .agg(
                {
                    "generacion_total_kwh": "mean",
                    "consumo_diario_kwh": "mean",
                    "autosuficiencia_pct": "mean",
                }
            )
            .round(1)
        )

        meses = [
            "Ene",
            "Feb",
            "Mar",
            "Abr",
            "May",
            "Jun",
            "Jul",
            "Ago",
            "Sep",
            "Oct",
            "Nov",
            "Dic",
        ]

        for mes, (idx, row) in zip(meses, estacional.iterrows()):
            print(
                f"{mes}: Gen={row['generacion_total_kwh']:.1f} kWh, "
                f"Cons={row['consumo_diario_kwh']:.1f} kWh, "
                f"Auto={row['autosuficiencia_pct']:.1f}%"
            )

    except FileNotFoundError:
        print("⚠️  Primero ejecuta la simulación principal para generar los datos.")
    except Exception as e:
        print(f"❌ Error en el análisis: {str(e)}")


# Agregar la función de ejemplo al final
print("\n💡 Para ver un ejemplo de análisis de los datos generados, ejecuta:")
print("   ejemplo_analisis_datos()")
