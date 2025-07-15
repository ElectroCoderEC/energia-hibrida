import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment
import warnings

warnings.filterwarnings("ignore")

# Configuración inicial
np.random.seed(42)
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 6, 30)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# Escuelas en zonas rurales de Chimborazo
school_params = {
    "Escuela 21 de Abril": {
        "estudiantes": 75,
        "area_total_m2": 1200,
        "area_techo_disponible_m2": 400,
        "area_terreno_disponible_m2": 800,
        "latitud": -1.649952,
        "longitud": -78.650000,
        "altitud": 3200,
        "terreno": "montañoso",
        "zona": "San Juan",
    },
    "Escuela Sangay": {
        "estudiantes": 150,
        "area_total_m2": 2500,
        "area_techo_disponible_m2": 800,
        "area_terreno_disponible_m2": 1500,
        "latitud": -1.699949,
        "longitud": -78.700004,
        "altitud": 3400,
        "terreno": "valle",
        "zona": "Licto",
    },
    "Politécnica de Chimborazo": {
        "estudiantes": 50,
        "area_total_m2": 800,
        "area_techo_disponible_m2": 300,
        "area_terreno_disponible_m2": 500,
        "latitud": -1.62,
        "longitud": -78.68,
        "altitud": 3600,
        "terreno": "montañoso",
        "zona": "Calpi",
    },
    "Colegio Condorazo": {
        "estudiantes": 250,
        "area_total_m2": 4000,
        "area_techo_disponible_m2": 1200,
        "area_terreno_disponible_m2": 2000,
        "latitud": -1.75,
        "longitud": -78.62,
        "altitud": 2800,
        "terreno": "planicie",
        "zona": "Chambo",
    },
    "Colegio Victor Proaño": {
        "estudiantes": 120,
        "area_total_m2": 2000,
        "area_techo_disponible_m2": 600,
        "area_terreno_disponible_m2": 1000,
        "latitud": -1.68,
        "longitud": -78.67,
        "altitud": 3100,
        "terreno": "valle",
        "zona": "Flores",
    },
}

# Tarifa eléctrica
TARIFA_KWH = 0.10  # USD/kWh
CO2_FACTOR = 0.385  # kg CO2/kWh (factor de emisión Ecuador)


def generate_hourly_climate_data():
    """Genera datos climáticos por hora con patrones realistas para Chimborazo"""
    climate_data = []

    for date in dates:
        day_of_year = date.timetuple().tm_yday

        for school_name, params in school_params.items():
            # Factor estacional (más radiación en época seca: junio-septiembre)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

            # Factor de altitud
            altitude_factor = 1 + (params["altitud"] - 3000) * 0.00005

            for hour in range(24):
                # Radiación solar horaria (0 en la noche, pico al mediodía)
                if 6 <= hour <= 18:
                    hour_angle = (hour - 12) * 15  # grados
                    solar_elevation = 90 - abs(hour_angle)

                    # Radiación base considerando ubicación ecuatorial
                    base_radiation = 5.5 * altitude_factor * seasonal_factor

                    # Modelo de radiación horaria
                    hourly_radiation = (
                        base_radiation * np.sin(np.radians(solar_elevation))
                        if solar_elevation > 0
                        else 0
                    )

                    # Nubosidad variable durante el día
                    cloud_factor = (
                        np.random.beta(2, 5) if hour < 14 else np.random.beta(3, 2)
                    )
                    hourly_radiation *= 1 - cloud_factor * 0.4

                    # Añadir variabilidad
                    hourly_radiation += np.random.normal(0, 0.2)
                    hourly_radiation = max(0, hourly_radiation)
                else:
                    hourly_radiation = 0

                # Temperatura horaria (mínima a las 6am, máxima a las 2pm)
                temp_base = 15 - (params["altitud"] - 2800) * 0.003
                temp_variation = (
                    8 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -2
                )
                temperature = temp_base + temp_variation + np.random.normal(0, 1)

                # Velocidad del viento (mayor en las tardes)
                if params["terreno"] == "montañoso":
                    base_wind = 4.5
                elif params["terreno"] == "valle":
                    base_wind = 3.0
                else:
                    base_wind = 3.8

                wind_factor = (
                    1 + 0.3 * np.sin(np.pi * (hour - 6) / 12)
                    if 6 <= hour <= 18
                    else 0.8
                )
                wind_speed = (
                    base_wind * wind_factor * seasonal_factor + np.random.normal(0, 0.5)
                )
                wind_speed = max(0.5, wind_speed)

                # Humedad relativa (inversa a la temperatura)
                humidity = (
                    65 + 20 * np.sin(np.pi * (18 - hour) / 12)
                    if 6 <= hour <= 18
                    else 80
                )
                humidity += np.random.normal(0, 5)
                humidity = np.clip(humidity, 30, 95)

                climate_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "radiacion_solar_kwh_m2": round(hourly_radiation, 3),
                        "temperatura_c": round(temperature, 1),
                        "velocidad_viento_m_s": round(wind_speed, 2),
                        "humedad_relativa_%": round(humidity, 1),
                        "presion_atmosferica_hpa": round(
                            1013.25
                            * (1 - 0.0065 * params["altitud"] / 288.15) ** 5.255,
                            1,
                        ),
                    }
                )

    return pd.DataFrame(climate_data)


def generate_consumption_data():
    """Genera datos de consumo energético por hora considerando patrones escolares"""
    consumption_data = []

    # Factores de consumo por tipo de espacio
    consumption_factors = {
        "aulas": 0.015,  # kWh/m2/hora durante uso
        "laboratorios": 0.025,  # Mayor consumo por equipos
        "oficinas": 0.012,
        "areas_comunes": 0.008,
        "iluminacion_exterior": 0.005,
    }

    for date in dates:
        is_weekday = date.weekday() < 5
        is_vacation = date.month in [1, 2, 7, 8]  # Vacaciones en Ecuador
        is_holiday = date.month == 12 and date.day > 20  # Fiestas navideñas

        for school_name, params in school_params.items():
            area = params["area_total_m2"]
            students = params["estudiantes"]

            # Consumo base por estudiante
            base_consumption_per_student = 0.08  # kWh/estudiante/hora en horario activo

            for hour in range(24):
                consumption = 0

                # Horario escolar normal (7am - 2pm)
                if is_weekday and not is_vacation and not is_holiday:
                    if 7 <= hour <= 13:
                        # Consumo durante clases
                        occupancy_factor = 0.9 if 8 <= hour <= 12 else 0.7
                        consumption = (
                            area * consumption_factors["aulas"] * 0.6
                            + area * consumption_factors["laboratorios"] * 0.2
                            + area * consumption_factors["oficinas"] * 0.1
                            + area * consumption_factors["areas_comunes"] * 0.1
                        ) * occupancy_factor

                        # Consumo adicional por estudiante
                        consumption += (
                            students * base_consumption_per_student * occupancy_factor
                        )

                    elif 14 <= hour <= 17:
                        # Actividades extracurriculares (20% de probabilidad)
                        if np.random.random() < 0.2:
                            consumption = area * consumption_factors["aulas"] * 0.3
                            consumption += students * base_consumption_per_student * 0.2

                    elif 18 <= hour <= 20:
                        # Tutorías o eventos especiales (10% de probabilidad)
                        if np.random.random() < 0.1:
                            consumption = area * consumption_factors["aulas"] * 0.2
                            consumption += students * base_consumption_per_student * 0.1

                # Fines de semana - actividades ocasionales
                elif not is_weekday and not is_vacation:
                    if 8 <= hour <= 12 and np.random.random() < 0.05:
                        # Eventos comunitarios ocasionales
                        consumption = area * consumption_factors["areas_comunes"] * 0.4

                # Consumo base 24/7 (seguridad, refrigeración, standby)
                base_24_7 = area * 0.001  # kWh/m2/hora
                consumption += base_24_7

                # Iluminación exterior (6pm - 6am)
                if hour >= 18 or hour < 6:
                    consumption += (
                        area * consumption_factors["iluminacion_exterior"] * 0.5
                    )

                # Variabilidad aleatoria (±10%)
                consumption *= 0.9 + 0.2 * np.random.random()

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "consumo_kwh": round(consumption, 3),
                        "tipo_dia": "laboral" if is_weekday else "fin_semana",
                        "periodo": "vacaciones" if is_vacation else "clases",
                    }
                )

    return pd.DataFrame(consumption_data)


def generate_renewable_systems():
    """Genera especificaciones de sistemas renovables disponibles"""
    systems_data = []

    for school_name, params in school_params.items():
        # Dimensionamiento basado en área disponible y número de estudiantes
        area_techo = params["area_techo_disponible_m2"]
        area_terreno = params["area_terreno_disponible_m2"]
        students = params["estudiantes"]

        # Capacidad solar máxima (170 W/m2 para paneles modernos)
        max_solar_capacity = area_techo * 0.17  # kWp

        # Capacidad eólica según terreno
        if params["terreno"] == "montañoso":
            max_wind_capacity = min(area_terreno * 0.01, 20)  # kW
        else:
            max_wind_capacity = min(area_terreno * 0.008, 15)  # kW

        # Sistema dimensionado para la escuela
        solar_capacity = min(
            max_solar_capacity, students * 0.15
        )  # 0.15 kWp por estudiante
        wind_capacity = min(
            max_wind_capacity, students * 0.05
        )  # 0.05 kW por estudiante
        battery_capacity = (solar_capacity + wind_capacity) * 4  # 4 horas de autonomía

        systems_data.append(
            {
                "escuela": school_name,
                "capacidad_solar_instalada_kwp": round(solar_capacity, 2),
                "capacidad_eolica_instalada_kw": round(wind_capacity, 2),
                "capacidad_bateria_kwh": round(battery_capacity, 2),
                "area_paneles_m2": round(solar_capacity / 0.17, 2),
                "numero_aerogeneradores": int(
                    wind_capacity / 2
                ),  # Aerogeneradores de 2kW
                "eficiencia_paneles_%": 17.5,
                "eficiencia_inversor_%": 95,
                "eficiencia_baterias_%": 90,
                "degradacion_anual_solar_%": 0.5,
                "vida_util_paneles_años": 25,
                "vida_util_baterias_años": 15,
                "vida_util_aerogeneradores_años": 20,
            }
        )

    return pd.DataFrame(systems_data)


def generate_equipment_catalog():
    """Genera catálogo de equipos disponibles para sistemas híbridos en Ecuador"""
    equipment = {
        "paneles_solares": [
            {
                "tipo": "Panel Monocristalino 550W",
                "potencia_w": 550,
                "precio_usd": 195,
                "eficiencia_%": 21.2,
                "area_m2": 2.3,
            },
            {
                "tipo": "Panel Monocristalino 450W",
                "potencia_w": 450,
                "precio_usd": 160,
                "eficiencia_%": 20.5,
                "area_m2": 2.1,
            },
            {
                "tipo": "Panel Policristalino 330W",
                "potencia_w": 330,
                "precio_usd": 105,
                "eficiencia_%": 17.8,
                "area_m2": 1.9,
            },
            {
                "tipo": "Panel Bifacial 540W",
                "potencia_w": 540,
                "precio_usd": 210,
                "eficiencia_%": 21.0,
                "area_m2": 2.2,
            },
            {
                "tipo": "Panel Flexible 160W",
                "potencia_w": 160,
                "precio_usd": 110,
                "eficiencia_%": 17.5,
                "area_m2": 1.0,
            },
        ],
        "aerogeneradores": [
            {
                "tipo": "Aerogenerador Horizontal 1.5kW",
                "potencia_w": 1500,
                "precio_usd": 1200,
                "vel_arranque_m/s": 2.8,
                "vel_nominal_m/s": 10,
            },
            {
                "tipo": "Aerogenerador Horizontal 3kW",
                "potencia_w": 3000,
                "precio_usd": 4000,
                "vel_arranque_m/s": 3.0,
                "vel_nominal_m/s": 12,
            },
            {
                "tipo": "Aerogenerador Vertical 600W",
                "potencia_w": 600,
                "precio_usd": 650,
                "vel_arranque_m/s": 2.0,
                "vel_nominal_m/s": 9,
            },
            {
                "tipo": "Aerogenerador Vertical 2kW",
                "potencia_w": 2000,
                "precio_usd": 3200,
                "vel_arranque_m/s": 2.5,
                "vel_nominal_m/s": 11,
            },
        ],
        "baterias": [
            {
                "tipo": "Batería Litio 3.5kWh",
                "capacidad_kwh": 3.5,
                "precio_usd": 1550,
                "ciclos": 6000,
                "dod_%": 90,
            },
            {
                "tipo": "Batería Litio 7kWh",
                "capacidad_kwh": 7.0,
                "precio_usd": 3100,
                "ciclos": 6000,
                "dod_%": 90,
            },
            {
                "tipo": "Batería Plomo-Ácido 2kWh",
                "capacidad_kwh": 2.0,
                "precio_usd": 350,
                "ciclos": 1200,
                "dod_%": 50,
            },
            {
                "tipo": "Batería AGM 4kWh",
                "capacidad_kwh": 4.0,
                "precio_usd": 700,
                "ciclos": 1400,
                "dod_%": 55,
            },
        ],
        "inversores": [
            {
                "tipo": "Inversor Híbrido 3kW",
                "potencia_kw": 3,
                "precio_usd": 850,
                "eficiencia_%": 95,
                "tipo_onda": "senoidal_pura",
            },
            {
                "tipo": "Inversor Híbrido 5kW",
                "potencia_kw": 5,
                "precio_usd": 1250,
                "eficiencia_%": 96,
                "tipo_onda": "senoidal_pura",
            },
            {
                "tipo": "Microinversor 350W",
                "potencia_kw": 0.35,
                "precio_usd": 110,
                "eficiencia_%": 96,
                "tipo_onda": "senoidal_pura",
            },
        ],
        "controladores_carga": [
            {
                "tipo": "Controlador MPPT 40A",
                "corriente_a": 40,
                "precio_usd": 210,
                "eficiencia_%": 98,
                "voltaje_v": 48,
            },
            {
                "tipo": "Controlador MPPT 80A",
                "corriente_a": 80,
                "precio_usd": 420,
                "eficiencia_%": 98,
                "voltaje_v": 48,
            },
            {
                "tipo": "Controlador Híbrido Eólico-Solar 60A",
                "corriente_a": 60,
                "precio_usd": 380,
                "eficiencia_%": 97,
                "voltaje_v": 48,
            },
            {
                "tipo": "Controlador PWM 30A",
                "corriente_a": 30,
                "precio_usd": 75,
                "eficiencia_%": 85,
                "voltaje_v": 24,
            },
        ],
        "kits_solares": [
            {
                "tipo": "Kit Solar 600W + Inversor 3kW + 3.5kWh Litio",
                "precio_usd": 2800,
                "autonomia_dias": 1,
            },
            {
                "tipo": "Kit Solar 1.2kW + Inversor 5kW + 7kWh Litio",
                "precio_usd": 5200,
                "autonomia_dias": 2,
            },
            {
                "tipo": "Kit Solar Económico 300W + Plomo-Ácido 2kWh",
                "precio_usd": 950,
                "autonomia_dias": 0.5,
            },
        ],
        "accesorios": [
            {
                "tipo": "Estructura Techo Aluminio",
                "unidad": "kWp",
                "precio_por_unidad_usd": 160,
            },
            {
                "tipo": "Estructura Suelo Galvanizada",
                "unidad": "kWp",
                "precio_por_unidad_usd": 220,
            },
            {
                "tipo": "Torre Aerogenerador 9m",
                "unidad": "unidad",
                "precio_por_unidad_usd": 2100,
            },
            {
                "tipo": "Cableado Solar PV 4mm2",
                "unidad": "metro",
                "precio_por_unidad_usd": 2.8,
            },
            {
                "tipo": "Gabinete DC+AC",
                "unidad": "unidad",
                "precio_por_unidad_usd": 450,
            },
            {
                "tipo": "Sistema Monitoreo WiFi",
                "unidad": "unidad",
                "precio_por_unidad_usd": 700,
            },
        ],
        "mano_obra": [
            {
                "tipo": "Instalación Kit Solar Pequeño",
                "unidad": "sistema",
                "precio_por_unidad_usd": 400,
            },
            {
                "tipo": "Instalación Sistema Mayor a 5kW",
                "unidad": "sistema",
                "precio_por_unidad_usd": 1200,
            },
            {
                "tipo": "Instalación Aerogenerador",
                "unidad": "unidad",
                "precio_por_unidad_usd": 800,
            },
            {
                "tipo": "Puesta en Marcha Completa",
                "unidad": "sistema",
                "precio_por_unidad_usd": 600,
            },
        ],
    }

    # Convertir a DataFrames
    dfs_equipment = {}
    for category, items in equipment.items():
        dfs_equipment[category] = pd.DataFrame(items)

    return dfs_equipment


def calculate_generation_and_balance(df_climate, df_consumption, df_systems):
    """Calcula generación horaria y balance energético"""
    generation_data = []

    # Agrupar clima por escuela-fecha-hora
    climate_grouped = df_climate.set_index(["escuela", "fecha", "hora"])

    for idx, system in df_systems.iterrows():
        school = system["escuela"]

        # Obtener datos climáticos para esta escuela
        school_climate = climate_grouped.loc[school]

        for (date, hour), climate in school_climate.iterrows():
            # Generación solar
            solar_capacity = system["capacidad_solar_instalada_kwp"]
            radiation = climate["radiacion_solar_kwh_m2"]
            temperature = climate["temperatura_c"]

            # Pérdidas por temperatura (0.4% por grado sobre 25°C)
            temp_loss = max(0, (temperature - 25) * 0.004)
            solar_efficiency = system["eficiencia_inversor_%"] / 100 * (1 - temp_loss)

            # Degradación anual
            years_passed = (date.year - 2021) + date.month / 12
            degradation = 1 - (system["degradacion_anual_solar_%"] / 100 * years_passed)

            solar_generation = (
                solar_capacity * radiation * solar_efficiency * degradation
            )

            # Generación eólica
            wind_capacity = system["capacidad_eolica_instalada_kw"]
            wind_speed = climate["velocidad_viento_m_s"]

            # Curva de potencia simplificada
            if wind_speed < 2.5:
                wind_factor = 0
            elif wind_speed < 10:
                wind_factor = ((wind_speed - 2.5) / 7.5) ** 3
            elif wind_speed < 25:
                wind_factor = 1
            else:
                wind_factor = 0  # Parada por seguridad

            # Ajuste por densidad del aire en altitud
            air_density_factor = (1013.25 / climate["presion_atmosferica_hpa"]) ** 0.5
            wind_generation = (
                wind_capacity
                * wind_factor
                * air_density_factor
                * (system["eficiencia_inversor_%"] / 100)
            )

            generation_data.append(
                {
                    "fecha": date,
                    "hora": hour,
                    "escuela": school,
                    "generacion_solar_kwh": round(solar_generation, 3),
                    "generacion_eolica_kwh": round(wind_generation, 3),
                    "generacion_total_kwh": round(
                        solar_generation + wind_generation, 3
                    ),
                }
            )

    df_generation = pd.DataFrame(generation_data)

    # Calcular balance energético
    df_balance = pd.merge(
        df_generation,
        df_consumption[["fecha", "hora", "escuela", "consumo_kwh"]],
        on=["fecha", "hora", "escuela"],
    )

    # Balance instantáneo
    df_balance["balance_kwh"] = (
        df_balance["generacion_total_kwh"] - df_balance["consumo_kwh"]
    )
    df_balance["excedente_kwh"] = df_balance["balance_kwh"].apply(lambda x: max(0, x))
    df_balance["deficit_kwh"] = df_balance["balance_kwh"].apply(lambda x: max(0, -x))

    # Autosuficiencia
    df_balance["autosuficiencia_%"] = np.minimum(
        100,
        (df_balance["generacion_total_kwh"] / df_balance["consumo_kwh"].replace(0, 1))
        * 100,
    )

    return df_balance


def calculate_economics(df_balance, df_systems, equipment_catalog):
    """Calcula costos, ahorros y métricas económicas con selección aleatoria del catálogo"""
    economics_data = []

    for school in df_systems["escuela"].unique():
        school_data = df_balance[df_balance["escuela"] == school]
        system = df_systems[df_systems["escuela"] == school].iloc[0]

        solar_capacity = system["capacidad_solar_instalada_kwp"]
        wind_capacity = system["capacidad_eolica_instalada_kw"]
        battery_capacity = system["capacidad_bateria_kwh"]

        # -------------------------------
        # Selección aleatoria de equipos
        # -------------------------------
        panel = equipment_catalog["paneles_solares"].sample(1).iloc[0]
        aerogen = equipment_catalog["aerogeneradores"].sample(1).iloc[0]
        bateria = equipment_catalog["baterias"].sample(1).iloc[0]
        inversor = equipment_catalog["inversores"].sample(1).iloc[0]
        controlador = equipment_catalog["controladores_carga"].sample(1).iloc[0]

        # -------------------------------
        # Costos según equipo seleccionado
        # -------------------------------
        # Paneles
        panel_power_w = panel["potencia_w"]
        panel_price = panel["precio_usd"]
        num_panels = (solar_capacity * 1000) / panel_power_w
        solar_cost = num_panels * panel_price

        # Aerogeneradores
        aero_power_w = aerogen["potencia_w"]
        aero_price = aerogen["precio_usd"]
        num_aeros = (wind_capacity * 1000) / aero_power_w
        wind_cost = num_aeros * aero_price

        # Baterías
        bat_kwh = bateria["capacidad_kwh"]
        bat_price = bateria["precio_usd"]
        num_bats = battery_capacity / bat_kwh
        battery_cost = num_bats * bat_price

        # Inversores
        inverter_price_per_kw = inversor["precio_usd"] / inversor["potencia_kw"]
        inverter_cost = max(solar_capacity, wind_capacity) * inverter_price_per_kw

        # Controladores
        controller_price_per_kw = controlador["precio_usd"] / (
            controlador["corriente_a"] * controlador["voltaje_v"] / 1000
        )
        controller_cost = solar_capacity * controller_price_per_kw

        # -------------------------------
        # Costos adicionales
        # -------------------------------
        structure_cost = solar_capacity * 150 + wind_capacity / 2 * 2500
        wiring_cost = (solar_capacity + wind_capacity) * 100
        labor_cost = solar_capacity * 300 + wind_capacity * 500 + 1500

        total_investment = (
            solar_cost
            + wind_cost
            + battery_cost
            + inverter_cost
            + controller_cost
            + structure_cost
            + wiring_cost
            + labor_cost
        )

        # -------------------------------
        # Métricas energéticas
        # -------------------------------
        total_generation = school_data["generacion_total_kwh"].sum()
        total_consumption = school_data["consumo_kwh"].sum()
        total_deficit = school_data["deficit_kwh"].sum()

        energy_from_grid = total_deficit
        energy_saved = total_consumption - energy_from_grid
        cost_saved = energy_saved * TARIFA_KWH
        co2_avoided = energy_saved * CO2_FACTOR

        annual_days = 365
        years_analyzed = len(school_data) / (24 * annual_days)
        annual_savings = cost_saved / years_analyzed
        simple_payback = (
            total_investment / annual_savings if annual_savings > 0 else 999
        )

        # -------------------------------
        # Registrar resultados
        # -------------------------------
        economics_data.append(
            {
                "escuela": school,
                "inversion_total_usd": round(total_investment, 2),
                "inversion_solar_usd": round(solar_cost, 2),
                "inversion_eolica_usd": round(wind_cost, 2),
                "inversion_baterias_usd": round(battery_cost, 2),
                "inversion_bos_usd": round(
                    inverter_cost
                    + controller_cost
                    + structure_cost
                    + wiring_cost
                    + labor_cost,
                    2,
                ),
                "generacion_total_kwh": round(total_generation, 2),
                "consumo_total_kwh": round(total_consumption, 2),
                "energia_red_kwh": round(energy_from_grid, 2),
                "energia_ahorrada_kwh": round(energy_saved, 2),
                "ahorro_total_usd": round(cost_saved, 2),
                "ahorro_anual_usd": round(annual_savings, 2),
                "co2_evitado_total_kg": round(co2_avoided, 2),
                "co2_evitado_anual_kg": round(co2_avoided / years_analyzed, 2),
                "periodo_retorno_años": round(simple_payback, 2),
                "autosuficiencia_promedio_%": round(
                    school_data["autosuficiencia_%"].mean(), 2
                ),
                "equipo_panel": panel["tipo"],
                "equipo_aerogenerador": aerogen["tipo"],
                "equipo_bateria": bateria["tipo"],
                "equipo_inversor": inversor["tipo"],
                "equipo_controlador": controlador["tipo"],
            }
        )

    return pd.DataFrame(economics_data)


def generate_ml_dataset(df_balance, df_systems, df_economics):
    """Genera dataset para entrenamiento de ML"""
    ml_data = []

    # Agregar datos por escuela y mes
    df_balance["año_mes"] = pd.to_datetime(df_balance["fecha"]).dt.to_period("M")

    monthly_grouped = (
        df_balance.groupby(["escuela", "año_mes"])
        .agg(
            {
                "generacion_solar_kwh": "sum",
                "generacion_eolica_kwh": "sum",
                "consumo_kwh": "sum",
                "deficit_kwh": "sum",
                "autosuficiencia_%": "mean",
            }
        )
        .reset_index()
    )

    for idx, row in monthly_grouped.iterrows():
        school = row["escuela"]
        school_data = school_params[school]
        system = df_systems[df_systems["escuela"] == school].iloc[0]
        economics = df_economics[df_economics["escuela"] == school].iloc[0]

        ml_data.append(
            {
                # Features de entrada
                "latitud": school_data["latitud"],
                "longitud": school_data["longitud"],
                "altitud_m": school_data["altitud"],
                "area_disponible_techo_m2": school_data["area_techo_disponible_m2"],
                "area_disponible_terreno_m2": school_data["area_terreno_disponible_m2"],
                "numero_estudiantes": school_data["estudiantes"],
                "consumo_mensual_kwh": row["consumo_kwh"],
                "mes": row["año_mes"].month,
                "año": row["año_mes"].year,
                # Features del sistema instalado
                "capacidad_solar_kwp": system["capacidad_solar_instalada_kwp"],
                "capacidad_eolica_kw": system["capacidad_eolica_instalada_kw"],
                "capacidad_bateria_kwh": system["capacidad_bateria_kwh"],
                # Variables objetivo (para predicción)
                "generacion_solar_mensual_kwh": row["generacion_solar_kwh"],
                "generacion_eolica_mensual_kwh": row["generacion_eolica_kwh"],
                "autosuficiencia_promedio_%": row["autosuficiencia_%"],
                "deficit_mensual_kwh": row["deficit_kwh"],
                "ahorro_mensual_usd": row["consumo_kwh"] * TARIFA_KWH
                - row["deficit_kwh"] * TARIFA_KWH,
                "co2_evitado_mensual_kg": (row["consumo_kwh"] - row["deficit_kwh"])
                * CO2_FACTOR,
            }
        )

    return pd.DataFrame(ml_data)


def save_to_excel(filename, dataframes_dict):
    """Guarda múltiples DataFrames en un archivo Excel con formato"""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Obtener la hoja y aplicar formato
            worksheet = writer.sheets[sheet_name]

            # Formato de encabezados
            for cell in worksheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(
                    start_color="366092", end_color="366092", fill_type="solid"
                )
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Ajustar ancho de columnas
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)
                worksheet.column_dimensions[column_letter].width = adjusted_width


def main():
    """Función principal para generar todos los datos"""
    print("Generando datos sintéticos para sistemas híbridos de energía renovable...")
    print(f"Período: {start_date.date()} a {end_date.date()}")
    print(f"Escuelas: {len(school_params)}")

    # Generar datos
    print("\n1. Generando datos climáticos horarios...")
    df_climate = generate_hourly_climate_data()
    print(f"   - {len(df_climate)} registros generados")

    print("\n2. Generando datos de consumo energético...")
    df_consumption = generate_consumption_data()
    print(f"   - {len(df_consumption)} registros generados")

    print("\n3. Generando especificaciones de sistemas...")
    df_systems = generate_renewable_systems()
    print(f"   - {len(df_systems)} sistemas configurados")

    print("\n4. Generando catálogo de equipos...")
    equipment_catalog = generate_equipment_catalog()
    print(f"   - {len(equipment_catalog)} categorías de equipos")

    print("\n5. Calculando generación y balance energético...")
    df_balance = calculate_generation_and_balance(
        df_climate, df_consumption, df_systems
    )
    print(f"   - {len(df_balance)} registros de balance")

    print("\n6. Calculando análisis económico...")
    df_economics = calculate_economics(df_balance, df_systems, equipment_catalog)
    print(f"   - Análisis económico completado")

    print("\n7. Generando dataset para ML...")
    df_ml = generate_ml_dataset(df_balance, df_systems, df_economics)
    print(f"   - {len(df_ml)} registros para entrenamiento")

    # Crear resúmenes adicionales
    print("\n8. Creando resúmenes...")

    # Resumen diario
    df_daily = (
        df_balance.groupby(["fecha", "escuela"])
        .agg(
            {
                "generacion_solar_kwh": "sum",
                "generacion_eolica_kwh": "sum",
                "generacion_total_kwh": "sum",
                "consumo_kwh": "sum",
                "deficit_kwh": "sum",
                "excedente_kwh": "sum",
                "autosuficiencia_%": "mean",
            }
        )
        .reset_index()
    )

    # Resumen mensual
    df_balance["año_mes"] = pd.to_datetime(df_balance["fecha"]).dt.to_period("M")
    df_monthly = (
        df_balance.groupby(["año_mes", "escuela"])
        .agg(
            {
                "generacion_solar_kwh": "sum",
                "generacion_eolica_kwh": "sum",
                "generacion_total_kwh": "sum",
                "consumo_kwh": "sum",
                "deficit_kwh": "sum",
                "excedente_kwh": "sum",
                "autosuficiencia_%": "mean",
            }
        )
        .reset_index()
    )

    # Preparar datos para Excel
    excel_data = {
        "Resumen_Economico": df_economics,
        "Sistemas_Instalados": df_systems,
        "Balance_Diario": df_daily,
        "Balance_Mensual": df_monthly,
        "Datos_Climaticos_Muestra": df_climate.head(1000),  # Muestra
        "Consumo_Energetico_Muestra": df_consumption.head(1000),  # Muestra
        "Catalogo_Paneles": equipment_catalog["paneles_solares"],
        "Catalogo_Aerogeneradores": equipment_catalog["aerogeneradores"],
        "Catalogo_Baterias": equipment_catalog["baterias"],
        "Catalogo_Inversores": equipment_catalog["inversores"],
        "Catalogo_Controladores": equipment_catalog["controladores_carga"],
    }

    # Guardar archivos
    print("\n9. Guardando archivos...")

    # Crear directorio si no existe
    if not os.path.exists("datos_energia_renovable"):
        os.makedirs("datos_energia_renovable")

    # Guardar Excel completo
    excel_filename = (
        "datos_energia_renovable/analisis_sistemas_hibridos_chimborazo.xlsx"
    )
    save_to_excel(excel_filename, excel_data)
    print(f"   - Excel guardado: {excel_filename}")

    # Guardar CSV para ML
    csv_filename = "datos_energia_renovable/dataset_ml_energia_renovable.csv"
    df_ml.to_csv(csv_filename, index=False)
    print(f"   - CSV para ML guardado: {csv_filename}")

    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)

    for idx, row in df_economics.iterrows():
        print(f"\n{row['escuela']}:")
        print(f"  - Inversión total: ${row['inversion_total_usd']:,.2f}")
        print(f"  - Ahorro anual: ${row['ahorro_anual_usd']:,.2f}")
        print(f"  - Período de retorno: {row['periodo_retorno_años']:.1f} años")
        print(f"  - CO2 evitado anual: {row['co2_evitado_anual_kg']:,.0f} kg")
        print(f"  - Autosuficiencia promedio: {row['autosuficiencia_promedio_%']:.1f}%")

    print("\n" + "=" * 60)
    print("RECOMENDACIÓN DE MODELO ML")
    print("=" * 60)
    print("\nPara este tipo de análisis, se recomienda usar:")
    print("\n1. **Random Forest Regressor** o **XGBoost**:")
    print("   - Excelentes para capturar relaciones no lineales")
    print("   - Manejan bien las interacciones entre variables")
    print("   - Robustos ante outliers")
    print("\n2. **Red Neuronal (MLP)**:")
    print("   - Para patrones más complejos")
    print("   - Si se tiene suficiente data")
    print("\n3. **Ensemble de modelos**:")
    print("   - Combinar predicciones de múltiples modelos")
    print("   - Mayor precisión y robustez")

    print("\nVariables de entrada para el modelo:")
    print("- Ubicación (latitud, longitud, altitud)")
    print("- Área disponible (techo y terreno)")
    print("- Consumo energético promedio")
    print("- Presupuesto disponible")
    print("- Mes del año (estacionalidad)")

    print("\nVariables de salida del modelo:")
    print("- Capacidad óptima solar (kWp)")
    print("- Capacidad óptima eólica (kW)")
    print("- Capacidad de baterías (kWh)")
    print("- Generación esperada (kWh)")
    print("- Ahorro anual (USD)")
    print("- ROI y período de retorno")
    print("- CO2 evitado")
    print("- Lista de equipos recomendados")

    print("\n¡Generación de datos completada exitosamente!")


if __name__ == "__main__":
    main()
