def generate_equipment_catalog():
    """Generar catálogo completo de equipos disponibles"""
    equipment_data = []

    # 1. PANELES SOLARES
    solar_panels = [
        {
            "categoria": "Panel Solar",
            "modelo": "Monocristalino 450W",
            "potencia_w": 450,
            "eficiencia": 0.21,
            "precio_usd": 180,
            "dimensiones": "2.1x1.1x0.04m",
            "garantia_años": 25,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Monocristalino 550W",
            "potencia_w": 550,
            "eficiencia": 0.22,
            "precio_usd": 220,
            "dimensiones": "2.3x1.1x0.04m",
            "garantia_años": 25,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Policristalino 400W",
            "potencia_w": 400,
            "eficiencia": 0.19,
            "precio_usd": 150,
            "dimensiones": "2.0x1.0x0.04m",
            "garantia_años": 20,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Policristalino 330W",
            "potencia_w": 330,
            "eficiencia": 0.17,
            "precio_usd": 125,
            "dimensiones": "1.96x0.99x0.04m",
            "garantia_años": 20,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Bifacial 500W",
            "potencia_w": 500,
            "eficiencia": 0.21,
            "precio_usd": 250,
            "dimensiones": "2.1x1.1x0.04m",
            "garantia_años": 30,
        },
    ]

    # 2. AEROGENERADORES
    wind_turbines = [
        {
            "categoria": "Aerogenerador",
            "modelo": "Horizontal 1kW",
            "potencia_w": 1000,
            "velocidad_arranque": 2.0,
            "precio_usd": 2800,
            "dimensiones": "Rotor 2.5m",
            "garantia_años": 5,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Horizontal 2kW",
            "potencia_w": 2000,
            "velocidad_arranque": 2.5,
            "precio_usd": 4500,
            "dimensiones": "Rotor 3.2m",
            "garantia_años": 5,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Horizontal 5kW",
            "potencia_w": 5000,
            "velocidad_arranque": 3.0,
            "precio_usd": 8500,
            "dimensiones": "Rotor 5.2m",
            "garantia_años": 5,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Vertical 1.5kW",
            "potencia_w": 1500,
            "velocidad_arranque": 2.0,
            "precio_usd": 3800,
            "dimensiones": "H:4m D:1.5m",
            "garantia_años": 3,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Vertical 3kW",
            "potencia_w": 3000,
            "velocidad_arranque": 2.5,
            "precio_usd": 6200,
            "dimensiones": "H:5m D:2m",
            "garantia_años": 3,
        },
    ]

    # 3. BATERÍAS
    batteries = [
        {
            "categoria": "Batería",
            "modelo": "LiFePO4 100Ah 12V",
            "capacidad_kwh": 1.2,
            "ciclos_vida": 4000,
            "precio_usd": 380,
            "dimensiones": "32x17x22cm",
            "garantia_años": 5,
        },
        {
            "categoria": "Batería",
            "modelo": "LiFePO4 200Ah 12V",
            "capacidad_kwh": 2.4,
            "ciclos_vida": 4000,
            "precio_usd": 720,
            "dimensiones": "52x24x22cm",
            "garantia_años": 5,
        },
        {
            "categoria": "Batería",
            "modelo": "LiFePO4 200Ah 24V",
            "capacidad_kwh": 4.8,
            "ciclos_vida": 4500,
            "precio_usd": 1250,
            "dimensiones": "52x24x22cm",
            "garantia_años": 8,
        },
        {
            "categoria": "Batería",
            "modelo": "Plomo-Ácido 200Ah 12V",
            "capacidad_kwh": 2.4,
            "ciclos_vida": 1200,
            "precio_usd": 420,
            "dimensiones": "52x24x22cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Batería",
            "modelo": "Gel 150Ah 12V",
            "capacidad_kwh": 1.8,
            "ciclos_vida": 1800,
            "precio_usd": 350,
            "dimensiones": "48x17x22cm",
            "garantia_años": 3,
        },
    ]

    # 4. INVERSORES
    inverters = [
        {
            "categoria": "Inversor",
            "modelo": "Híbrido 3kW MPPT",
            "potencia_w": 3000,
            "eficiencia": 0.95,
            "precio_usd": 650,
            "dimensiones": "40x30x15cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Inversor",
            "modelo": "Híbrido 5kW MPPT",
            "potencia_w": 5000,
            "eficiencia": 0.96,
            "precio_usd": 980,
            "dimensiones": "45x35x18cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Inversor",
            "modelo": "Híbrido 10kW MPPT",
            "potencia_w": 10000,
            "eficiencia": 0.97,
            "precio_usd": 1850,
            "dimensiones": "60x40x20cm",
            "garantia_años": 3,
        },
        {
            "categoria": "Inversor",
            "modelo": "Grid-Tie 8kW",
            "potencia_w": 8000,
            "eficiencia": 0.98,
            "precio_usd": 1200,
            "dimensiones": "50x35x15cm",
            "garantia_años": 5,
        },
        {
            "categoria": "Inversor",
            "modelo": "Off-Grid 2kW",
            "potencia_w": 2000,
            "eficiencia": 0.93,
            "precio_usd": 450,
            "dimensiones": "35x25x12cm",
            "garantia_años": 2,
        },
    ]

    # 5. CONTROLADORES DE CARGA
    charge_controllers = [
        {
            "categoria": "Controlador",
            "modelo": "MPPT 40A 12/24V",
            "corriente_max": 40,
            "voltaje": "12/24V",
            "precio_usd": 180,
            "dimensiones": "20x15x8cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Controlador",
            "modelo": "MPPT 60A 12/24/48V",
            "corriente_max": 60,
            "voltaje": "12/24/48V",
            "precio_usd": 280,
            "dimensiones": "25x18x10cm",
            "garantia_años": 3,
        },
        {
            "categoria": "Controlador",
            "modelo": "PWM 30A 12/24V",
            "corriente_max": 30,
            "voltaje": "12/24V",
            "precio_usd": 85,
            "dimensiones": "18x12x6cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Controlador",
            "modelo": "MPPT 80A 12/24/48V",
            "corriente_max": 80,
            "voltaje": "12/24/48V",
            "precio_usd": 420,
            "dimensiones": "30x20x12cm",
            "garantia_años": 3,
        },
        {
            "categoria": "Controlador",
            "modelo": "Eólico 1kW 12/24V",
            "corriente_max": 50,
            "voltaje": "12/24V",
            "precio_usd": 320,
            "dimensiones": "25x15x10cm",
            "garantia_años": 2,
        },
    ]

    # 6. ESTRUCTURAS Y MONTAJE
    structures = [
        {
            "categoria": "Estructura",
            "modelo": "Soporte Techo 4 Paneles",
            "capacidad": "4 paneles",
            "material": "Aluminio",
            "precio_usd": 280,
            "dimensiones": "4.2x2.2m",
            "garantia_años": 10,
        },
        {
            "categoria": "Estructura",
            "modelo": "Soporte Suelo 8 Paneles",
            "capacidad": "8 paneles",
            "material": "Acero galvanizado",
            "precio_usd": 450,
            "dimensiones": "8.4x2.2m",
            "garantia_años": 15,
        },
        {
            "categoria": "Estructura",
            "modelo": "Torre Eólica 12m",
            "capacidad": "Hasta 5kW",
            "material": "Acero galvanizado",
            "precio_usd": 2800,
            "dimensiones": "H:12m",
            "garantia_años": 20,
        },
        {
            "categoria": "Estructura",
            "modelo": "Torre Eólica 18m",
            "capacidad": "Hasta 10kW",
            "material": "Acero galvanizado",
            "precio_usd": 4200,
            "dimensiones": "H:18m",
            "garantia_años": 20,
        },
        {
            "categoria": "Estructura",
            "modelo": "Rieles Aluminio",
            "capacidad": "Por metro",
            "material": "Aluminio",
            "precio_usd": 15,
            "dimensiones": "1m",
            "garantia_años": 25,
        },
    ]

    # 7. CABLEADO Y PROTECCIONES
    wiring = [
        {
            "categoria": "Cableado",
            "modelo": "Cable Solar 4mm² Rojo",
            "especificacion": "4mm²",
            "tipo": "DC Solar",
            "precio_usd": 2.5,
            "unidad": "metro",
            "garantia_años": 20,
        },
        {
            "categoria": "Cableado",
            "modelo": "Cable Solar 6mm² Negro",
            "especificacion": "6mm²",
            "tipo": "DC Solar",
            "precio_usd": 3.8,
            "unidad": "metro",
            "garantia_años": 20,
        },
        {
            "categoria": "Protección",
            "modelo": "Fusible DC 20A",
            "especificacion": "20A",
            "tipo": "Fusible",
            "precio_usd": 8,
            "unidad": "pieza",
            "garantia_años": 5,
        },
        {
            "categoria": "Protección",
            "modelo": "Breaker AC 40A",
            "especificacion": "40A",
            "tipo": "Interruptor",
            "precio_usd": 45,
            "unidad": "pieza",
            "garantia_años": 10,
        },
        {
            "categoria": "Protección",
            "modelo": "Supresor Sobretensión",
            "especificacion": "1000V",
            "tipo": "SPD",
            "precio_usd": 180,
            "unidad": "pieza",
            "garantia_años": 5,
        },
    ]

    # 8. GABINETES Y ACCESORIOS
    accessories = [
        {
            "categoria": "Gabinete",
            "modelo": "Gabinete Metálico IP65",
            "especificacion": "60x40x20cm",
            "tipo": "Protección",
            "precio_usd": 120,
            "unidad": "pieza",
            "garantia_años": 10,
        },
        {
            "categoria": "Monitoreo",
            "modelo": "Sistema Monitoreo WiFi",
            "especificacion": "Inalámbrico",
            "tipo": "Monitoreo",
            "precio_usd": 320,
            "unidad": "kit",
            "garantia_años": 3,
        },
        {
            "categoria": "Herramientas",
            "modelo": "Kit Herramientas Instalación",
            "especificacion": "Completo",
            "tipo": "Herramientas",
            "precio_usd": 450,
            "unidad": "kit",
            "garantia_años": 5,
        },
        {
            "categoria": "Medición",
            "modelo": "Multímetro Digital",
            "especificacion": "600V",
            "tipo": "Medición",
            "precio_usd": 85,
            "unidad": "pieza",
            "garantia_años": 2,
        },
        {
            "categoria": "Conectores",
            "modelo": "Conectores MC4 Par",
            "especificacion": "MC4",
            "tipo": "Conector",
            "precio_usd": 12,
            "unidad": "par",
            "garantia_años": 25,
        },
    ]

    # 9. MANO DE OBRA Y SERVICIOS
    labor = [
        {
            "categoria": "Mano de Obra",
            "modelo": "Instalación Sistema Solar",
            "especificacion": "Por kW",
            "tipo": "Instalación",
            "precio_usd": 150,
            "unidad": "kW",
            "garantia_años": 1,
        },
        {
            "categoria": "Mano de Obra",
            "modelo": "Instalación Sistema Eólico",
            "especificacion": "Por kW",
            "tipo": "Instalación",
            "precio_usd": 250,
            "unidad": "kW",
            "garantia_años": 1,
        },
        {
            "categoria": "Mano de Obra",
            "modelo": "Configuración Sistema",
            "especificacion": "Puesta en marcha",
            "tipo": "Configuración",
            "precio_usd": 300,
            "unidad": "sistema",
            "garantia_años": 1,
        },
        {
            "categoria": "Servicio",
            "modelo": "Mantenimiento Anual",
            "especificacion": "Preventivo",
            "tipo": "Mantenimiento",
            "precio_usd": 200,
            "unidad": "año",
            "garantia_años": 1,
        },
        {
            "categoria": "Servicio",
            "modelo": "Capacitación Operadores",
            "especificacion": "8 horas",
            "tipo": "Capacitación",
            "precio_usd": 400,
            "unidad": "curso",
            "garantia_años": 0,
        },
    ]

    # Combinar todos los equipos
    all_equipment = (
        solar_panels
        + wind_turbines
        + batteries
        + inverters
        + charge_controllers
        + structures
        + wiring
        + accessories
        + labor
    )

    for eq in all_equipment:
        equipment_data.append(eq)

    return pd.DataFrame(equipment_data)


"""
Generador de Datos Sintéticos para Sistemas de Energía Renovable
Instituciones Educativas Rurales - Provincia de Chimborazo, Ecuador

Basado en investigación de condiciones climáticas reales, costos de mercado ecuatoriano,
y patrones de consumo educativo rural específicos de la región andina.

Período: 2021-2025
Ubicación: Cantón Riobamba, Provincia de Chimborazo
Altitud: 2800-3500m
"""

import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime, timedelta
import random

# Configuración inicial
np.random.seed(42)
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 6, 30)
dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]

# Horas de radiación solar (6:00 AM - 6:00 PM aproximadamente)
SOLAR_HOURS_START = 6
SOLAR_HOURS_END = 18

# Configuración de escuelas con datos realistas para Chimborazo
school_params = {
    "Escuela 21 de Abril": {
        "size": "small",
        "students": 75,
        "altitude": 2950,
        "terrain": "valle",
        "area_disponible_m2": 180,
        "coordenadas": (-1.6342, -78.6342),
        "zona_viento": "media",
    },
    "Escuela Sangay": {
        "size": "medium",
        "students": 145,
        "altitude": 3150,
        "terrain": "montañoso",
        "area_disponible_m2": 320,
        "coordenadas": (-1.6789, -78.6123),
        "zona_viento": "alta",
    },
    "Politécnica de Chimborazo": {
        "size": "small",
        "students": 65,
        "altitude": 3400,
        "terrain": "montañoso",
        "area_disponible_m2": 150,
        "coordenadas": (-1.6456, -78.5987),
        "zona_viento": "muy_alta",
    },
    "Colegio Condorazo": {
        "size": "large",
        "students": 280,
        "altitude": 2800,
        "terrain": "planicie",
        "area_disponible_m2": 450,
        "coordenadas": (-1.6123, -78.6456),
        "zona_viento": "baja",
    },
    "Colegio Victor Proaño": {
        "size": "medium",
        "students": 165,
        "altitude": 3050,
        "terrain": "valle",
        "area_disponible_m2": 285,
        "coordenadas": (-1.6567, -78.6234),
        "zona_viento": "media",
    },
}


def create_directories():
    """Crear directorios necesarios"""
    if not os.path.exists("datos_sintéticos_chimborazo"):
        os.makedirs("datos_sintéticos_chimborazo")


def generate_climate_data():
    """Generar datos climáticos realistas para Chimborazo por hora"""
    climate_data = []

    for school_name, params in school_params.items():
        # Radiación solar base según altitud (mejora con altitud)
        base_radiation = 4.3 + (params["altitude"] - 2800) * 0.0004

        # Velocidad de viento según zona y altitud
        wind_zones = {
            "baja": (2.5, 4.0),
            "media": (3.5, 5.5),
            "alta": (4.5, 6.5),
            "muy_alta": (5.5, 7.5),
        }
        wind_base_min, wind_base_max = wind_zones[params["zona_viento"]]

        for day, date in enumerate(dates):
            # Variación estacional (Ecuador: más radiación dic-feb, menos jun-ago)
            seasonal_factor = np.sin(2 * np.pi * (day - 80) / 365) * 0.25 + 1

            # Radiación solar diaria base (kWh/m²/día)
            daily_radiation_base = base_radiation * seasonal_factor + np.random.normal(
                0, 0.4
            )
            daily_radiation_base = max(daily_radiation_base, 2.8)
            daily_radiation_base = min(daily_radiation_base, 6.2)

            # Velocidad de viento diaria base
            wind_seasonal = 1 + 0.15 * np.sin(2 * np.pi * (day - 180) / 365)
            daily_wind_base = (
                np.random.uniform(wind_base_min, wind_base_max) * wind_seasonal
            )
            daily_wind_base = max(daily_wind_base, 1.2)

            # Temperatura base según altitud y estación
            temp_base = 14.5 - (params["altitude"] - 2800) * 0.006
            seasonal_temp = 3 * np.sin(2 * np.pi * (day - 30) / 365)
            daily_temp_base = temp_base + seasonal_temp + np.random.normal(0, 2.5)

            # Humedad relativa base
            humidity_base = (
                65 + 20 * np.sin(2 * np.pi * (day + 60) / 365) + np.random.normal(0, 8)
            )
            humidity_base = max(30, min(95, humidity_base))

            # Nubosidad base
            cloud_cover_base = (
                np.random.beta(2.5, 4)
                if daily_radiation_base > 4.5
                else np.random.beta(4, 2.5)
            )

            # Precipitación diaria
            is_wet_season = date.month in [1, 2, 3, 4, 10, 11, 12]
            rain_prob = 0.35 if is_wet_season else 0.15
            daily_precipitation = (
                np.random.exponential(8) if np.random.random() < rain_prob else 0
            )

            # Generar datos por hora
            for hour in range(24):
                # Radiación solar horaria (solo durante horas de luz)
                if SOLAR_HOURS_START <= hour <= SOLAR_HOURS_END:
                    # Curva solar tipo campana
                    solar_peak_hour = 12  # Mediodía
                    hour_factor = np.cos((hour - solar_peak_hour) * np.pi / 12) ** 2
                    # Variación por nubosidad
                    cloud_factor = 1 - cloud_cover_base * np.random.uniform(0.2, 0.6)
                    hourly_radiation = (
                        (daily_radiation_base / 8) * hour_factor * cloud_factor
                    )
                    hourly_radiation = max(0, hourly_radiation)
                else:
                    hourly_radiation = 0

                # Viento horario (con variación)
                wind_variation = (
                    1 + 0.3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.2)
                )
                hourly_wind = daily_wind_base * wind_variation
                hourly_wind = max(0.5, hourly_wind)

                # Temperatura horaria (ciclo diurno)
                temp_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
                hourly_temp = daily_temp_base + temp_variation + np.random.normal(0, 1)

                # Humedad horaria (inversa a temperatura)
                humidity_variation = -10 * np.sin(2 * np.pi * (hour - 6) / 24)
                hourly_humidity = (
                    humidity_base + humidity_variation + np.random.normal(0, 5)
                )
                hourly_humidity = max(20, min(100, hourly_humidity))

                # Nubosidad horaria
                hourly_clouds = cloud_cover_base + np.random.normal(0, 0.1)
                hourly_clouds = max(0, min(1, hourly_clouds))

                climate_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "radiacion_solar_kwh_m2": round(hourly_radiation, 3),
                        "velocidad_viento_m_s": round(hourly_wind, 2),
                        "temperatura_c": round(hourly_temp, 1),
                        "humedad_relativa": round(hourly_humidity, 1),
                        "nubosidad": round(hourly_clouds, 2),
                        "precipitacion_mm": round(
                            daily_precipitation / 24, 2
                        ),  # Distribuir en 24h
                        "altitud_m": params["altitude"],
                        "latitud": params["coordenadas"][0],
                        "longitud": params["coordenadas"][1],
                    }
                )

    return pd.DataFrame(climate_data)


def generate_consumption_data():
    """Generar datos de consumo energético realistas para escuelas rurales por hora"""
    consumption_data = []

    for school_name, params in school_params.items():
        # Consumo base por estudiante (kWh/mes) según tamaño
        consumption_per_student = {
            "small": np.random.uniform(18, 25),
            "medium": np.random.uniform(22, 32),
            "large": np.random.uniform(28, 40),
        }

        base_monthly = consumption_per_student[params["size"]] * params["students"]
        base_hourly_avg = base_monthly / (30 * 24)  # Promedio por hora

        # Patrones horarios escolares
        hourly_patterns = np.zeros(24)

        # Horario matutino (7:00-12:00) - clases principales
        for h in range(7, 13):
            hourly_patterns[h] = np.random.uniform(0.08, 0.15)

        # Horario vespertino (13:00-18:00) - clases complementarias
        for h in range(13, 19):
            hourly_patterns[h] = np.random.uniform(0.05, 0.10)

        # Horarios nocturnos (19:00-22:00) - actividades extracurriculares ocasionales
        for h in range(19, 23):
            hourly_patterns[h] = np.random.uniform(0.01, 0.03)

        # Horas de madrugada (23:00-06:00) - consumo mínimo (seguridad, sistemas básicos)
        for h in list(range(23, 24)) + list(range(0, 7)):
            hourly_patterns[h] = np.random.uniform(0.005, 0.015)

        for date in dates:
            # Factor vacaciones (calendario Sierra: vacaciones jul-ago, dic-ene)
            is_vacation = date.month in [7, 8] or (
                date.month in [12, 1] and date.day > 15
            )
            vacation_factor = 0.25 if is_vacation else 1.0

            # Factor fin de semana
            is_weekend = date.weekday() >= 5
            weekend_factor = 0.20 if is_weekend else 1.0

            # Factor actividades extracurriculares (5% de probabilidad cualquier día)
            extra_activities = np.random.random() < 0.05

            for hour in range(24):
                # Consumo base según patrón horario
                hourly_consumption = base_hourly_avg * hourly_patterns[hour]

                # Aplicar factores estacionales
                hourly_consumption *= vacation_factor * weekend_factor

                # Determinar si es hora de clases
                is_class_time = (
                    not is_vacation
                    and not is_weekend
                    and ((7 <= hour <= 12) or (13 <= hour <= 18))
                )

                # Actividades extracurriculares (tutorías, exposiciones, eventos)
                if extra_activities and 19 <= hour <= 22:
                    hourly_consumption *= np.random.uniform(
                        2, 4
                    )  # Incremento por actividades extra
                    is_class_time = False  # No es clase regular

                # Pequeño porcentaje de equipos que quedan encendidos fines de semana
                if is_weekend and np.random.random() < 0.15:  # 15% probabilidad
                    hourly_consumption *= np.random.uniform(1.5, 2.5)

                # Variabilidad aleatoria
                hourly_consumption *= np.random.uniform(0.85, 1.15)

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hour,
                        "escuela": school_name,
                        "consumo_kwh": round(max(0.001, hourly_consumption), 4),
                        "es_vacacion": is_vacation,
                        "es_fin_semana": is_weekend,
                        "es_hora_clases": is_class_time,
                        "actividades_extra": extra_activities and (19 <= hour <= 22),
                    }
                )

    return pd.DataFrame(consumption_data)


def generate_renewable_systems_data():
    """Generar configuraciones de sistemas renovables por escuela"""
    systems_data = []

    # Costos actualizados para Ecuador 2024-2025
    cost_ranges = {
        "solar_usd_kwp": (1200, 2200),
        "wind_usd_kw": (2000, 5000),
        "battery_lithium_usd_kwh": (250, 450),
        "battery_lead_usd_kwh": (150, 280),
    }

    for school_name, params in school_params.items():
        # Dimensionamiento según tamaño y consumo estimado
        system_sizing = {
            "small": {"solar_kw": (5, 8), "wind_kw": (1, 3), "battery_kwh": (15, 25)},
            "medium": {
                "solar_kw": (10, 15),
                "wind_kw": (2, 5),
                "battery_kwh": (25, 40),
            },
            "large": {"solar_kw": (15, 25), "wind_kw": (4, 8), "battery_kwh": (40, 65)},
        }

        sizing = system_sizing[params["size"]]

        # Capacidades instaladas
        solar_capacity = np.random.uniform(*sizing["solar_kw"])
        wind_capacity = np.random.uniform(*sizing["wind_kw"])
        battery_capacity = np.random.uniform(*sizing["battery_kwh"])

        # Verificar limitación por área disponible
        solar_area_needed = solar_capacity * 6  # ~6 m²/kW para paneles
        if (
            solar_area_needed > params["area_disponible_m2"] * 0.8
        ):  # 80% del área utilizable
            solar_capacity = (params["area_disponible_m2"] * 0.8) / 6

        # Eficiencias según tecnología
        solar_efficiency = np.random.uniform(0.18, 0.22)  # Paneles modernos
        wind_efficiency = np.random.uniform(0.25, 0.40)
        battery_efficiency = np.random.uniform(0.90, 0.95)  # Litio

        # Degradación anual
        system_age = np.random.randint(0, 3)  # Sistemas relativamente nuevos
        solar_degradation = 0.005 * system_age  # 0.5% anual
        battery_degradation = 0.02 * system_age  # 2% anual

        # Costos
        solar_cost = solar_capacity * np.random.uniform(*cost_ranges["solar_usd_kwp"])
        wind_cost = wind_capacity * np.random.uniform(*cost_ranges["wind_usd_kw"])
        battery_cost = battery_capacity * np.random.uniform(
            *cost_ranges["battery_lithium_usd_kwh"]
        )
        installation_cost = (
            solar_cost + wind_cost + battery_cost
        ) * 0.12  # 12% instalación

        # Factor de corrección por altitud
        altitude_factor = 1 - (params["altitude"] - 2800) * 0.00005  # Densidad aire

        systems_data.append(
            {
                "escuela": school_name,
                "students": params["students"],  # Agregar número de estudiantes
                "size": params["size"],  # Agregar tamaño
                "altitude": params["altitude"],  # Agregar altitud
                "terrain": params["terrain"],  # Agregar terreno
                "zona_viento": params["zona_viento"],  # Agregar zona de viento
                "capacidad_solar_kw": round(solar_capacity, 2),
                "capacidad_eolica_kw": round(wind_capacity, 2),
                "capacidad_bateria_kwh": round(battery_capacity, 2),
                "eficiencia_solar": round(solar_efficiency, 3),
                "eficiencia_eolica": round(wind_efficiency, 3),
                "eficiencia_bateria": round(battery_efficiency, 3),
                "edad_sistema_años": system_age,
                "degradacion_solar": round(solar_degradation, 3),
                "degradacion_bateria": round(battery_degradation, 3),
                "costo_solar_usd": round(solar_cost, 2),
                "costo_eolico_usd": round(wind_cost, 2),
                "costo_baterias_usd": round(battery_cost, 2),
                "costo_instalacion_usd": round(installation_cost, 2),
                "costo_total_usd": round(
                    solar_cost + wind_cost + battery_cost + installation_cost, 2
                ),
                "factor_altitud": round(altitude_factor, 3),
                "area_solar_ocupada_m2": round(solar_capacity * 6, 1),
                "area_disponible_m2": params["area_disponible_m2"],
                "tipo_panel": np.random.choice(["monocristalino", "policristalino"]),
                "tipo_bateria": "litio",
                "tipo_aerogenerador": np.random.choice(
                    ["eje_horizontal", "eje_vertical"]
                ),
            }
        )

    return pd.DataFrame(systems_data)


def generate_power_generation_data(df_climate, df_systems):
    """Generar datos de producción de energía por hora"""
    generation_data = []

    for _, climate_row in df_climate.iterrows():
        school_name = climate_row["escuela"]
        date = climate_row["fecha"]
        hour = climate_row["hora"]

        # Obtener parámetros del sistema
        system = df_systems[df_systems["escuela"] == school_name].iloc[0]

        # === GENERACIÓN SOLAR ===
        radiation = climate_row["radiacion_solar_kwh_m2"]
        temperature = climate_row["temperatura_c"]
        cloud_cover = climate_row["nubosidad"]

        # Solo generar energía solar durante horas de luz
        if SOLAR_HOURS_START <= hour <= SOLAR_HOURS_END and radiation > 0:
            # Factor de temperatura (disminuye 0.4% por grado sobre 25°C)
            temp_factor = 1 - max(0, (temperature - 25) * 0.004)

            # Factor de nubosidad
            cloud_factor = 1 - cloud_cover * 0.4

            # Factor de degradación y eficiencia
            efficiency_factor = (
                system["eficiencia_solar"]
                * (1 - system["degradacion_solar"])
                * temp_factor
                * cloud_factor
            )

            # Generación solar horaria
            solar_generation = (
                system["capacidad_solar_kw"]
                * radiation
                * efficiency_factor
                * system["factor_altitud"]
            )
        else:
            solar_generation = 0

        # === GENERACIÓN EÓLICA ===
        wind_speed = climate_row["velocidad_viento_m_s"]

        # Curva de potencia simplificada para aerogeneradores
        cut_in = 2.5
        rated_speed = 12.0
        cut_out = 25.0

        if wind_speed < cut_in or wind_speed > cut_out:
            wind_factor = 0
        elif wind_speed < rated_speed:
            wind_factor = ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            wind_factor = 1.0

        # Generación eólica horaria
        wind_generation = (
            system["capacidad_eolica_kw"]
            * wind_factor
            * system["eficiencia_eolica"]
            * system["factor_altitud"]
        )

        # Factores de corrección adicionales
        # Efecto de turbulencia en terreno montañoso
        if system["terrain"] == "montañoso":
            wind_generation *= np.random.uniform(0.85, 1.05)

        generation_data.append(
            {
                "fecha": date,
                "hora": hour,
                "escuela": school_name,
                "generacion_solar_kwh": round(max(0, solar_generation), 4),
                "generacion_eolica_kwh": round(max(0, wind_generation), 4),
                "generacion_total_kwh": round(
                    max(0, solar_generation + wind_generation), 4
                ),
                "factor_capacidad_solar": (
                    round(solar_generation / system["capacidad_solar_kw"], 4)
                    if system["capacidad_solar_kw"] > 0
                    else 0
                ),
                "factor_capacidad_eolica": (
                    round(wind_generation / system["capacidad_eolica_kw"], 4)
                    if system["capacidad_eolica_kw"] > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(generation_data)


def generate_energy_balance(df_consumption, df_generation):
    """Calcular balance energético horario y diario"""
    # Verificar que ambos DataFrames tienen las mismas columnas clave
    print(f"   🔍 Generación: {len(df_generation):,} registros")
    print(f"   🔍 Consumo: {len(df_consumption):,} registros")

    # Unir consumo y generación por fecha, hora y escuela
    df_balance = pd.merge(
        df_generation, df_consumption, on=["fecha", "hora", "escuela"], how="inner"
    )
    print(f"   ✅ Balance combinado: {len(df_balance):,} registros")

    # Calcular balance horario
    df_balance["balance_energetico_kwh"] = (
        df_balance["generacion_total_kwh"] - df_balance["consumo_kwh"]
    )

    df_balance["excedente_kwh"] = np.where(
        df_balance["balance_energetico_kwh"] > 0,
        df_balance["balance_energetico_kwh"],
        0,
    )

    df_balance["deficit_kwh"] = np.where(
        df_balance["balance_energetico_kwh"] < 0,
        -df_balance["balance_energetico_kwh"],
        0,
    )

    # Calcular métricas adicionales
    df_balance["autosuficiencia_horaria_pct"] = np.where(
        df_balance["consumo_kwh"] > 0,
        np.minimum(
            100, (df_balance["generacion_total_kwh"] / df_balance["consumo_kwh"]) * 100
        ),
        100,
    )

    return df_balance


def generate_economics_data(df_systems, df_balance):
    """Generar datos económicos y de emisiones"""
    # Parámetros económicos Ecuador
    electricity_cost = 0.10  # USD/kWh
    co2_factor = 0.385  # kg CO2/kWh (factor de emisión Ecuador)

    economics_data = []

    print(f"   📊 Calculando datos económicos para {len(df_balance):,} registros...")

    for _, balance_row in df_balance.iterrows():
        school_name = balance_row["escuela"]

        # Obtener datos del sistema para esta escuela
        system = df_systems[df_systems["escuela"] == school_name].iloc[0]

        # Costos operativos horarios
        grid_energy = balance_row["deficit_kwh"]
        hourly_grid_cost = grid_energy * electricity_cost

        # Ahorros por energía renovable
        renewable_energy = balance_row["generacion_total_kwh"]
        hourly_savings = renewable_energy * electricity_cost

        # Emisiones
        co2_avoided = renewable_energy * co2_factor
        co2_emitted = grid_energy * co2_factor

        # ROI simple anual (evitar división por cero)
        annual_savings = hourly_savings * 24 * 365
        roi_years = (
            system["costo_total_usd"] / annual_savings if annual_savings > 100 else 50
        )

        economics_data.append(
            {
                "fecha": balance_row["fecha"],
                "hora": balance_row["hora"],
                "escuela": school_name,
                "costo_energia_red_usd": round(hourly_grid_cost, 4),
                "ahorro_horario_usd": round(hourly_savings, 4),
                "co2_evitado_kg": round(co2_avoided, 4),
                "co2_emitido_kg": round(co2_emitted, 4),
                "roi_simple_años": round(min(roi_years, 50), 1),  # Cap a 50 años
            }
        )

    return pd.DataFrame(economics_data)


def generate_equipment_catalog():
    """Generar catálogo completo de equipos disponibles"""
    equipment_data = []

    # 1. PANELES SOLARES
    solar_panels = [
        {
            "categoria": "Panel Solar",
            "modelo": "Monocristalino 450W",
            "potencia_w": 450,
            "eficiencia": 0.21,
            "precio_usd": 180,
            "dimensiones": "2.1x1.1x0.04m",
            "garantia_años": 25,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Monocristalino 550W",
            "potencia_w": 550,
            "eficiencia": 0.22,
            "precio_usd": 220,
            "dimensiones": "2.3x1.1x0.04m",
            "garantia_años": 25,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Policristalino 400W",
            "potencia_w": 400,
            "eficiencia": 0.19,
            "precio_usd": 150,
            "dimensiones": "2.0x1.0x0.04m",
            "garantia_años": 20,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Policristalino 330W",
            "potencia_w": 330,
            "eficiencia": 0.17,
            "precio_usd": 125,
            "dimensiones": "1.96x0.99x0.04m",
            "garantia_años": 20,
        },
        {
            "categoria": "Panel Solar",
            "modelo": "Bifacial 500W",
            "potencia_w": 500,
            "eficiencia": 0.21,
            "precio_usd": 250,
            "dimensiones": "2.1x1.1x0.04m",
            "garantia_años": 30,
        },
    ]

    # 2. AEROGENERADORES
    wind_turbines = [
        {
            "categoria": "Aerogenerador",
            "modelo": "Horizontal 1kW",
            "potencia_w": 1000,
            "velocidad_arranque": 2.0,
            "precio_usd": 2800,
            "dimensiones": "Rotor 2.5m",
            "garantia_años": 5,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Horizontal 2kW",
            "potencia_w": 2000,
            "velocidad_arranque": 2.5,
            "precio_usd": 4500,
            "dimensiones": "Rotor 3.2m",
            "garantia_años": 5,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Horizontal 5kW",
            "potencia_w": 5000,
            "velocidad_arranque": 3.0,
            "precio_usd": 8500,
            "dimensiones": "Rotor 5.2m",
            "garantia_años": 5,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Vertical 1.5kW",
            "potencia_w": 1500,
            "velocidad_arranque": 2.0,
            "precio_usd": 3800,
            "dimensiones": "H:4m D:1.5m",
            "garantia_años": 3,
        },
        {
            "categoria": "Aerogenerador",
            "modelo": "Vertical 3kW",
            "potencia_w": 3000,
            "velocidad_arranque": 2.5,
            "precio_usd": 6200,
            "dimensiones": "H:5m D:2m",
            "garantia_años": 3,
        },
    ]

    # 3. BATERÍAS
    batteries = [
        {
            "categoria": "Batería",
            "modelo": "LiFePO4 100Ah 12V",
            "capacidad_kwh": 1.2,
            "ciclos_vida": 4000,
            "precio_usd": 380,
            "dimensiones": "32x17x22cm",
            "garantia_años": 5,
        },
        {
            "categoria": "Batería",
            "modelo": "LiFePO4 200Ah 12V",
            "capacidad_kwh": 2.4,
            "ciclos_vida": 4000,
            "precio_usd": 720,
            "dimensiones": "52x24x22cm",
            "garantia_años": 5,
        },
        {
            "categoria": "Batería",
            "modelo": "LiFePO4 200Ah 24V",
            "capacidad_kwh": 4.8,
            "ciclos_vida": 4500,
            "precio_usd": 1250,
            "dimensiones": "52x24x22cm",
            "garantia_años": 8,
        },
        {
            "categoria": "Batería",
            "modelo": "Plomo-Ácido 200Ah 12V",
            "capacidad_kwh": 2.4,
            "ciclos_vida": 1200,
            "precio_usd": 420,
            "dimensiones": "52x24x22cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Batería",
            "modelo": "Gel 150Ah 12V",
            "capacidad_kwh": 1.8,
            "ciclos_vida": 1800,
            "precio_usd": 350,
            "dimensiones": "48x17x22cm",
            "garantia_años": 3,
        },
    ]

    # 4. INVERSORES
    inverters = [
        {
            "categoria": "Inversor",
            "modelo": "Híbrido 3kW MPPT",
            "potencia_w": 3000,
            "eficiencia": 0.95,
            "precio_usd": 650,
            "dimensiones": "40x30x15cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Inversor",
            "modelo": "Híbrido 5kW MPPT",
            "potencia_w": 5000,
            "eficiencia": 0.96,
            "precio_usd": 980,
            "dimensiones": "45x35x18cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Inversor",
            "modelo": "Híbrido 10kW MPPT",
            "potencia_w": 10000,
            "eficiencia": 0.97,
            "precio_usd": 1850,
            "dimensiones": "60x40x20cm",
            "garantia_años": 3,
        },
        {
            "categoria": "Inversor",
            "modelo": "Grid-Tie 8kW",
            "potencia_w": 8000,
            "eficiencia": 0.98,
            "precio_usd": 1200,
            "dimensiones": "50x35x15cm",
            "garantia_años": 5,
        },
        {
            "categoria": "Inversor",
            "modelo": "Off-Grid 2kW",
            "potencia_w": 2000,
            "eficiencia": 0.93,
            "precio_usd": 450,
            "dimensiones": "35x25x12cm",
            "garantia_años": 2,
        },
    ]

    # 5. CONTROLADORES DE CARGA
    charge_controllers = [
        {
            "categoria": "Controlador",
            "modelo": "MPPT 40A 12/24V",
            "corriente_max": 40,
            "voltaje": "12/24V",
            "precio_usd": 180,
            "dimensiones": "20x15x8cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Controlador",
            "modelo": "MPPT 60A 12/24/48V",
            "corriente_max": 60,
            "voltaje": "12/24/48V",
            "precio_usd": 280,
            "dimensiones": "25x18x10cm",
            "garantia_años": 3,
        },
        {
            "categoria": "Controlador",
            "modelo": "PWM 30A 12/24V",
            "corriente_max": 30,
            "voltaje": "12/24V",
            "precio_usd": 85,
            "dimensiones": "18x12x6cm",
            "garantia_años": 2,
        },
        {
            "categoria": "Controlador",
            "modelo": "MPPT 80A 12/24/48V",
            "corriente_max": 80,
            "voltaje": "12/24/48V",
            "precio_usd": 420,
            "dimensiones": "30x20x12cm",
            "garantia_años": 3,
        },
        {
            "categoria": "Controlador",
            "modelo": "Eólico 1kW 12/24V",
            "corriente_max": 50,
            "voltaje": "12/24V",
            "precio_usd": 320,
            "dimensiones": "25x15x10cm",
            "garantia_años": 2,
        },
    ]

    # 6. ESTRUCTURAS Y MONTAJE
    structures = [
        {
            "categoria": "Estructura",
            "modelo": "Soporte Techo 4 Paneles",
            "capacidad": "4 paneles",
            "material": "Aluminio",
            "precio_usd": 280,
            "dimensiones": "4.2x2.2m",
            "garantia_años": 10,
        },
        {
            "categoria": "Estructura",
            "modelo": "Soporte Suelo 8 Paneles",
            "capacidad": "8 paneles",
            "material": "Acero galvanizado",
            "precio_usd": 450,
            "dimensiones": "8.4x2.2m",
            "garantia_años": 15,
        },
        {
            "categoria": "Estructura",
            "modelo": "Torre Eólica 12m",
            "capacidad": "Hasta 5kW",
            "material": "Acero galvanizado",
            "precio_usd": 2800,
            "dimensiones": "H:12m",
            "garantia_años": 20,
        },
        {
            "categoria": "Estructura",
            "modelo": "Torre Eólica 18m",
            "capacidad": "Hasta 10kW",
            "material": "Acero galvanizado",
            "precio_usd": 4200,
            "dimensiones": "H:18m",
            "garantia_años": 20,
        },
        {
            "categoria": "Estructura",
            "modelo": "Rieles Aluminio",
            "capacidad": "Por metro",
            "material": "Aluminio",
            "precio_usd": 15,
            "dimensiones": "1m",
            "garantia_años": 25,
        },
    ]

    # 7. CABLEADO Y PROTECCIONES
    wiring = [
        {
            "categoria": "Cableado",
            "modelo": "Cable Solar 4mm² Rojo",
            "especificacion": "4mm²",
            "tipo": "DC Solar",
            "precio_usd": 2.5,
            "unidad": "metro",
            "garantia_años": 20,
        },
        {
            "categoria": "Cableado",
            "modelo": "Cable Solar 6mm² Negro",
            "especificacion": "6mm²",
            "tipo": "DC Solar",
            "precio_usd": 3.8,
            "unidad": "metro",
            "garantia_años": 20,
        },
        {
            "categoria": "Protección",
            "modelo": "Fusible DC 20A",
            "especificacion": "20A",
            "tipo": "Fusible",
            "precio_usd": 8,
            "unidad": "pieza",
            "garantia_años": 5,
        },
        {
            "categoria": "Protección",
            "modelo": "Breaker AC 40A",
            "especificacion": "40A",
            "tipo": "Interruptor",
            "precio_usd": 45,
            "unidad": "pieza",
            "garantia_años": 10,
        },
        {
            "categoria": "Protección",
            "modelo": "Supresor Sobretensión",
            "especificacion": "1000V",
            "tipo": "SPD",
            "precio_usd": 180,
            "unidad": "pieza",
            "garantia_años": 5,
        },
    ]

    # 8. GABINETES Y ACCESORIOS
    accessories = [
        {
            "categoria": "Gabinete",
            "modelo": "Gabinete Metálico IP65",
            "especificacion": "60x40x20cm",
            "tipo": "Protección",
            "precio_usd": 120,
            "unidad": "pieza",
            "garantia_años": 10,
        },
        {
            "categoria": "Monitoreo",
            "modelo": "Sistema Monitoreo WiFi",
            "especificacion": "Inalámbrico",
            "tipo": "Monitoreo",
            "precio_usd": 320,
            "unidad": "kit",
            "garantia_años": 3,
        },
        {
            "categoria": "Herramientas",
            "modelo": "Kit Herramientas Instalación",
            "especificacion": "Completo",
            "tipo": "Herramientas",
            "precio_usd": 450,
            "unidad": "kit",
            "garantia_años": 5,
        },
        {
            "categoria": "Medición",
            "modelo": "Multímetro Digital",
            "especificacion": "600V",
            "tipo": "Medición",
            "precio_usd": 85,
            "unidad": "pieza",
            "garantia_años": 2,
        },
        {
            "categoria": "Conectores",
            "modelo": "Conectores MC4 Par",
            "especificacion": "MC4",
            "tipo": "Conector",
            "precio_usd": 12,
            "unidad": "par",
            "garantia_años": 25,
        },
    ]

    # 9. MANO DE OBRA Y SERVICIOS
    labor = [
        {
            "categoria": "Mano de Obra",
            "modelo": "Instalación Sistema Solar",
            "especificacion": "Por kW",
            "tipo": "Instalación",
            "precio_usd": 150,
            "unidad": "kW",
            "garantia_años": 1,
        },
        {
            "categoria": "Mano de Obra",
            "modelo": "Instalación Sistema Eólico",
            "especificacion": "Por kW",
            "tipo": "Instalación",
            "precio_usd": 250,
            "unidad": "kW",
            "garantia_años": 1,
        },
        {
            "categoria": "Mano de Obra",
            "modelo": "Configuración Sistema",
            "especificacion": "Puesta en marcha",
            "tipo": "Configuración",
            "precio_usd": 300,
            "unidad": "sistema",
            "garantia_años": 1,
        },
        {
            "categoria": "Servicio",
            "modelo": "Mantenimiento Anual",
            "especificacion": "Preventivo",
            "tipo": "Mantenimiento",
            "precio_usd": 200,
            "unidad": "año",
            "garantia_años": 1,
        },
        {
            "categoria": "Servicio",
            "modelo": "Capacitación Operadores",
            "especificacion": "8 horas",
            "tipo": "Capacitación",
            "precio_usd": 400,
            "unidad": "curso",
            "garantia_años": 0,
        },
    ]

    # Combinar todos los equipos
    all_equipment = (
        solar_panels
        + wind_turbines
        + batteries
        + inverters
        + charge_controllers
        + structures
        + wiring
        + accessories
        + labor
    )

    for eq in all_equipment:
        equipment_data.append(eq)

    return pd.DataFrame(equipment_data)


def create_ml_dataset(df_climate, df_systems, df_balance, df_economics):
    """Crear dataset para machine learning"""
    print("   🔧 Creando dataset ML...")

    # Verificar datos de entrada
    print(f"      - Clima: {len(df_climate):,} registros")
    print(f"      - Sistemas: {len(df_systems)} escuelas")
    print(f"      - Balance: {len(df_balance):,} registros")
    print(f"      - Economía: {len(df_economics):,} registros")

    # Combinar tablas paso a paso
    # 1. Balance ya incluye clima, consumo y generación
    ml_data = df_balance.copy()

    print(ml_data.columns)

    # 2. Agregar datos de sistemas
    ml_data = ml_data.merge(df_systems, on="escuela", how="left")

    # 3. Agregar datos económicos
    ml_data = ml_data.merge(df_economics, on=["fecha", "hora", "escuela"], how="left")

    print(f"      ✅ Dataset combinado: {len(ml_data):,} registros")

    # Features temporales adicionales
    ml_data["mes"] = ml_data["fecha"].dt.month
    ml_data["dia_año"] = ml_data["fecha"].dt.dayofyear
    ml_data["año"] = ml_data["fecha"].dt.year
    ml_data["dia_semana"] = ml_data["fecha"].dt.dayofweek
    ml_data["es_verano"] = ml_data["mes"].isin([12, 1, 2, 3]).astype(int)
    ml_data["es_invierno"] = ml_data["mes"].isin([6, 7, 8]).astype(int)
    ml_data["es_fin_semana"] = ml_data["dia_semana"].isin([5, 6]).astype(int)

    # Variables objetivo para dimensionamiento
    ml_data["ratio_solar_consumo"] = np.where(
        ml_data["consumo_kwh"] > 0,
        ml_data["capacidad_solar_kw"]
        / (ml_data["consumo_kwh"] * 24),  # kW por kWh diario
        0,
    )

    ml_data["ratio_bateria_consumo"] = np.where(
        ml_data["consumo_kwh"] > 0,
        ml_data["capacidad_bateria_kwh"]
        / (ml_data["consumo_kwh"] * 24),  # kWh batería por kWh diario
        0,
    )

    # Verificar que la columna students existe
    if "students" in ml_data.columns and ml_data["students"].max() > 0:
        ml_data["costo_por_estudiante"] = (
            ml_data["costo_total_usd"] / ml_data["students"]
        )
        ml_data["consumo_per_capita"] = ml_data["consumo_kwh"] / (
            ml_data["students"] / 24
        )  # Por hora
    else:
        print("      ⚠️ Advertencia: Columna 'students' no válida")
        ml_data["costo_por_estudiante"] = ml_data["costo_total_usd"] / 100
        ml_data["consumo_per_capita"] = ml_data["consumo_kwh"] / 5

    # Features derivados técnicos
    ml_data["densidad_potencia"] = (
        ml_data["capacidad_solar_kw"] + ml_data["capacidad_eolica_kw"]
    ) / ml_data["area_disponible_m2"]
    ml_data["factor_altitud_normalizado"] = (
        ml_data["altitude"] - 2800
    ) / 1000  # Normalizar altitud
    ml_data["irradiancia_ajustada"] = ml_data["generacion_solar_kwh"] * (
        1 + ml_data["factor_altitud_normalizado"] * 0.1
    )

    # Limpiar valores infinitos y NaN
    ml_data = ml_data.replace([np.inf, -np.inf], np.nan)

    # Rellenar valores NaN de forma inteligente
    numeric_columns = ml_data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column != "fecha":
            ml_data[column] = ml_data[column].fillna(ml_data[column].median())

    # Para columnas categóricas
    categorical_columns = ml_data.select_dtypes(include=[object]).columns
    for column in categorical_columns:
        if column not in ["fecha", "escuela"]:
            mode_value = ml_data[column].mode()
            if len(mode_value) > 0:
                ml_data[column] = ml_data[column].fillna(mode_value.iloc[0])
            else:
                ml_data[column] = ml_data[column].fillna("unknown")

    print(f"      📊 Features finales: {len(ml_data.columns)} columnas")
    print(f"      🔢 Registros válidos: {len(ml_data.dropna()):,}")

    return ml_data


def save_to_excel(dfs_dict, filename):
    """Guardar múltiples DataFrames en un archivo Excel con formato"""
    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for sheet_name, df in dfs_dict.items():
                # Limpiar nombres de hojas (máximo 31 caracteres para Excel)
                clean_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

                # Formatear hojas
                worksheet = writer.sheets[clean_sheet_name]

                # Encabezados en negrita
                try:
                    from openpyxl.styles import Font, PatternFill

                    for cell in worksheet[1]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill(
                            start_color="366092", end_color="366092", fill_type="solid"
                        )

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
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = (
                            adjusted_width
                        )

                except ImportError:
                    print("   ⚠️ openpyxl no disponible para formateo avanzado")

        print(f"   ✅ Excel guardado exitosamente: {filename}")

    except Exception as e:
        print(f"   ❌ Error guardando Excel: {e}")
        # Intentar guardar sin formato
        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                for sheet_name, df in dfs_dict.items():
                    clean_sheet_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
            print(f"   ✅ Excel guardado (sin formato): {filename}")
        except Exception as e2:
            print(f"   ❌ Error crítico guardando Excel: {e2}")
            raise


def main():
    """Función principal"""
    print("Generando datos sintéticos para sistemas de energía renovable")
    print("Instituciones Educativas - Chimborazo, Ecuador")
    print("=" * 60)
    print(
        f"📅 Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"📊 Total días: {len(dates):,}")
    print(f"🏫 Escuelas: {len(school_params)}")
    print(
        f"⏰ Registros esperados: {len(dates) * 24 * len(school_params):,} (horarios)"
    )

    create_directories()

    # Generar datos paso a paso
    print("\n1. Generando datos climáticos horarios...")
    df_climate = generate_climate_data()
    print(f"   ✅ {len(df_climate):,} registros climáticos generados")

    print("\n2. Generando patrones de consumo horarios...")
    df_consumption = generate_consumption_data()
    print(f"   ✅ {len(df_consumption):,} registros de consumo generados")

    print("\n3. Configurando sistemas renovables...")
    df_systems = generate_renewable_systems_data()
    print(f"   ✅ {len(df_systems)} sistemas configurados")

    print("\n4. Calculando generación de energía horaria...")
    df_generation = generate_power_generation_data(df_climate, df_systems)
    print(f"   ✅ {len(df_generation):,} registros de generación calculados")

    print("\n5. Computando balance energético...")
    df_balance = generate_energy_balance(df_consumption, df_generation)
    print(f"   ✅ {len(df_balance):,} registros de balance calculados")

    print("\n6. Analizando aspectos económicos...")
    df_economics = generate_economics_data(df_systems, df_balance)
    print(f"   ✅ {len(df_economics):,} registros económicos generados")


def validate_generated_data(
    df_climate, df_consumption, df_generation, df_balance, df_economics
):
    """Validar coherencia de los datos generados"""
    print("\n🔍 VALIDANDO COHERENCIA DE DATOS...")
    print("-" * 50)

    validation_errors = []
    validation_warnings = []

    # 1. Verificar rangos de datos climáticos
    if df_climate["radiacion_solar_kwh_m2"].max() > 8:
        validation_warnings.append("Radiación solar muy alta (>8 kWh/m²)")

    if df_climate["radiacion_solar_kwh_m2"].min() < 0:
        validation_errors.append("Radiación solar negativa detectada")

    if df_climate["velocidad_viento_m_s"].max() > 30:
        validation_warnings.append("Velocidad de viento muy alta (>30 m/s)")

    if (
        df_climate["temperatura_c"].min() < -10
        or df_climate["temperatura_c"].max() > 40
    ):
        validation_warnings.append("Temperaturas fuera del rango típico de Chimborazo")

    # 2. Verificar coherencia energética
    negative_consumption = (df_consumption["consumo_kwh"] < 0).sum()
    if negative_consumption > 0:
        validation_errors.append(
            f"Consumo negativo en {negative_consumption} registros"
        )

    negative_generation = (df_generation["generacion_total_kwh"] < 0).sum()
    if negative_generation > 0:
        validation_errors.append(
            f"Generación negativa en {negative_generation} registros"
        )

    # 3. Verificar radiación nocturna
    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    night_radiation = df_climate[df_climate["hora"].isin(night_hours)][
        "radiacion_solar_kwh_m2"
    ]
    if (night_radiation > 0.01).sum() > 0:
        validation_warnings.append("Radiación solar detectada en horas nocturnas")

    # 4. Verificar balance energético
    extreme_autosuficiencia = (df_balance["autosuficiencia_horaria_pct"] > 500).sum()
    if extreme_autosuficiencia > 0:
        validation_warnings.append(
            f"Autosuficiencia >500% en {extreme_autosuficiencia} registros"
        )

    # 5. Verificar costos
    zero_costs = (df_economics["costo_energia_red_usd"] == 0).sum()
    total_records = len(df_economics)
    if zero_costs / total_records > 0.8:
        validation_warnings.append(
            "Demasiados registros con costo cero (posible sobredimensionamiento)"
        )

    # 6. Verificar distribución temporal
    unique_dates = df_climate["fecha"].nunique()
    expected_dates = len(dates)
    if unique_dates != expected_dates:
        validation_errors.append(
            f"Fechas faltantes: esperadas {expected_dates}, encontradas {unique_dates}"
        )

    # 7. Verificar integridad por escuela
    for school in school_params.keys():
        school_records = len(df_climate[df_climate["escuela"] == school])
        expected_records = len(dates) * 24  # Horario completo
        if school_records != expected_records:
            validation_errors.append(
                f"{school}: {school_records} registros, esperados {expected_records}"
            )

    # Mostrar resultados
    if validation_errors:
        print("❌ ERRORES CRÍTICOS:")
        for error in validation_errors:
            print(f"   • {error}")
    else:
        print("✅ Sin errores críticos detectados")

    if validation_warnings:
        print("\n⚠️ ADVERTENCIAS:")
        for warning in validation_warnings:
            print(f"   • {warning}")
    else:
        print("✅ Sin advertencias")

    # Estadísticas de calidad
    print(f"\n📊 ESTADÍSTICAS DE CALIDAD:")
    print(f"   • Completitud temporal: {unique_dates/expected_dates*100:.1f}%")
    print(
        f"   • Registros con radiación diurna: {((df_climate['hora'].between(6, 18)) & (df_climate['radiacion_solar_kwh_m2'] > 0)).sum():,}"
    )
    print(
        f"   • Registros con radiación nocturna: {((~df_climate['hora'].between(6, 18)) & (df_climate['radiacion_solar_kwh_m2'] > 0)).sum():,}"
    )
    print(
        f"   • Balance energético promedio: {df_balance['balance_energetico_kwh'].mean():.3f} kWh"
    )
    print(
        f"   • Autosuficiencia promedio: {df_balance['autosuficiencia_horaria_pct'].mean():.1f}%"
    )

    return len(validation_errors) == 0  # True si no hay errores críticos


def main():
    """Función principal"""
    print("Generando datos sintéticos para sistemas de energía renovable")
    print("Instituciones Educativas - Chimborazo, Ecuador")
    print("=" * 60)
    print(
        f"📅 Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"📊 Total días: {len(dates):,}")
    print(f"🏫 Escuelas: {len(school_params)}")
    print(
        f"⏰ Registros esperados: {len(dates) * 24 * len(school_params):,} (horarios)"
    )

    create_directories()

    # Generar datos paso a paso
    print("\n1. Generando datos climáticos horarios...")
    df_climate = generate_climate_data()
    print(f"   ✅ {len(df_climate):,} registros climáticos generados")

    print("\n2. Generando patrones de consumo horarios...")
    df_consumption = generate_consumption_data()
    print(f"   ✅ {len(df_consumption):,} registros de consumo generados")

    print("\n3. Configurando sistemas renovables...")
    df_systems = generate_renewable_systems_data()
    print(f"   ✅ {len(df_systems)} sistemas configurados")

    print("\n4. Calculando generación de energía horaria...")
    df_generation = generate_power_generation_data(df_climate, df_systems)
    print(f"   ✅ {len(df_generation):,} registros de generación calculados")

    print("\n5. Computando balance energético...")
    df_balance = generate_energy_balance(df_consumption, df_generation)
    print(f"   ✅ {len(df_balance):,} registros de balance calculados")

    print("\n6. Analizando aspectos económicos...")
    df_economics = generate_economics_data(df_systems, df_balance)
    print(f"   ✅ {len(df_economics):,} registros económicos generados")

    # Validar datos antes de continuar
    print("\n6.5. Validando coherencia de datos...")
    validation_passed = validate_generated_data(
        df_climate, df_consumption, df_generation, df_balance, df_economics
    )

    if not validation_passed:
        print("⚠️ Se detectaron errores en la validación. Continuando con precaución...")

    print("\n7. Creando catálogo de equipos...")
    df_equipment = generate_equipment_catalog()
    print(f"   ✅ {len(df_equipment)} equipos en catálogo")

    print("\n8. Preparando dataset para ML...")
    try:
        df_ml = create_ml_dataset(df_climate, df_systems, df_balance, df_economics)
        print(f"   ✅ {len(df_ml):,} registros para ML preparados")
        print(f"   📊 Features: {len(df_ml.columns)} columnas")
    except Exception as e:
        print(f"   ❌ Error creando dataset ML: {e}")
        traceback.print_exc()
        return

    # Guardar archivos
    print("\n9. Guardando archivos...")

    # Excel completo con muestras
    sample_size = min(5000, len(df_balance) // 10)  # Muestra más pequeña para Excel
    excel_data = {
        "Resumen_Sistemas": df_systems,
        "Muestra_Clima": df_climate.sample(n=sample_size).sort_values(
            ["escuela", "fecha", "hora"]
        ),
        "Muestra_Consumo": df_consumption.sample(n=sample_size).sort_values(
            ["escuela", "fecha", "hora"]
        ),
        "Muestra_Generacion": df_generation.sample(n=sample_size).sort_values(
            ["escuela", "fecha", "hora"]
        ),
        "Muestra_Balance": df_balance.sample(n=sample_size).sort_values(
            ["escuela", "fecha", "hora"]
        ),
        "Muestra_Economia": df_economics.sample(n=sample_size).sort_values(
            ["escuela", "fecha", "hora"]
        ),
        "Catalogo_Equipos": df_equipment,
    }

    excel_filename = (
        "datos_sintéticos_chimborazo/sistemas_renovables_chimborazo_completo.xlsx"
    )

    try:
        save_to_excel(excel_data, excel_filename)
        print(f"   ✅ Excel guardado: {excel_filename}")
    except Exception as e:
        print(f"   ⚠️ Error guardando Excel: {e}")

    # CSV completo para ML
    csv_filename = (
        "datos_sintéticos_chimborazo/dataset_ml_dimensionamiento_renovables.csv"
    )
    try:
        df_ml.to_csv(csv_filename, index=False)
        print(f"   ✅ CSV ML guardado: {csv_filename}")
    except Exception as e:
        print(f"   ❌ Error guardando CSV: {e}")
        return

    # Guardar datasets individuales como backup
    backup_files = {
        "clima_horario.csv": df_climate,
        "consumo_horario.csv": df_consumption,
        "generacion_horaria.csv": df_generation,
        "balance_energetico.csv": df_balance,
        "datos_economicos.csv": df_economics,
        "sistemas_configurados.csv": df_systems,
    }

    print(f"\n   💾 Guardando archivos de respaldo...")
    for filename, dataframe in backup_files.items():
        try:
            backup_path = f"datos_sintéticos_chimborazo/{filename}"
            dataframe.to_csv(backup_path, index=False)
            print(f"      ✅ {filename}")
        except Exception as e:
            print(f"      ❌ Error en {filename}: {e}")

    # Estadísticas detalladas
    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DETALLADAS")
    print("=" * 60)
    print(
        f"Período completo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"Total días simulados: {len(dates):,}")
    print(f"Registros por tabla:")
    print(f"  • Clima (horario): {len(df_climate):,}")
    print(f"  • Consumo (horario): {len(df_consumption):,}")
    print(f"  • Generación (horario): {len(df_generation):,}")
    print(f"  • Balance (horario): {len(df_balance):,}")
    print(f"  • Economía (horario): {len(df_economics):,}")
    print(f"  • Dataset ML final: {len(df_ml):,}")
    print(f"  • Equipos en catálogo: {len(df_equipment)}")

    # Verificación de datos por escuela
    print(f"\n🏫 VERIFICACIÓN POR ESCUELA:")
    print("-" * 50)
    for school, params in school_params.items():
        school_data = df_systems[df_systems["escuela"] == school].iloc[0]
        school_ml_count = len(df_ml[df_ml["escuela"] == school])

        print(f"\n📍 {school}")
        print(f"   Estudiantes: {params['students']}")
        print(f"   Altitud: {params['altitude']}m")
        print(f"   Área: {params['area_disponible_m2']}m²")
        print(f"   Solar: {school_data['capacidad_solar_kw']:.1f} kW")
        print(f"   Eólico: {school_data['capacidad_eolica_kw']:.1f} kW")
        print(f"   Baterías: {school_data['capacidad_bateria_kwh']:.1f} kWh")
        print(f"   Inversión: ${school_data['costo_total_usd']:,.0f}")
        print(f"   Registros ML: {school_ml_count:,}")

    # Verificar integridad del dataset ML
    print(f"\n🔍 VERIFICACIÓN INTEGRIDAD DATASET ML:")
    print("-" * 50)

    # Verificar valores nulos
    null_counts = df_ml.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    if len(columns_with_nulls) == 0:
        print("   ✅ Sin valores nulos")
    else:
        print("   ⚠️ Columnas con valores nulos:")
        for col, count in columns_with_nulls.items():
            print(f"      - {col}: {count:,} ({count/len(df_ml)*100:.1f}%)")

    # Verificar rangos de valores clave
    key_columns = [
        "radiacion_solar_kwh_m2",
        "velocidad_viento_m_s",
        "temperatura_c",
        "generacion_total_kwh",
        "consumo_kwh",
        "costo_total_usd",
    ]

    print(f"\n   📊 Rangos de valores clave:")
    for col in key_columns:
        if col in df_ml.columns:
            min_val = df_ml[col].min()
            max_val = df_ml[col].max()
            mean_val = df_ml[col].mean()
            print(f"      {col}: {min_val:.3f} - {max_val:.3f} (μ={mean_val:.3f})")

    # Verificar distribución temporal
    date_range = df_ml["fecha"].dt.date.unique()
    print(f"\n   📅 Rango temporal: {len(date_range):,} días únicos")
    print(f"      Desde: {date_range.min()}")
    print(f"      Hasta: {date_range.max()}")

    # Estadísticas por hora
    hourly_counts = df_ml["hora"].value_counts().sort_index()
    print(
        f"\n   ⏰ Distribución horaria: {hourly_counts.min():,} - {hourly_counts.max():,} registros/hora"
    )

    print(f"\n✅ ARCHIVOS GENERADOS EXITOSAMENTE:")
    print(f"   📊 Excel: {excel_filename}")
    print(f"   🤖 CSV ML: {csv_filename}")
    print(f"   📁 Directorio: datos_sintéticos_chimborazo/")

    print(f"\n🔬 MODELO ML RECOMENDADO:")
    print("-" * 40)
    print("Para dimensionamiento óptimo usar:")
    print("1. 🏆 Random Forest Regressor (principal)")
    print("2. 🥈 Gradient Boosting Regressor (ensemble)")
    print("3. 🥉 XGBoost Regressor (alternativa)")

    print(f"\nVariables objetivo principales:")
    print("- capacidad_solar_kw")
    print("- capacidad_eolica_kw")
    print("- capacidad_bateria_kwh")
    print("- costo_total_usd")

    print(f"\nFeatures más relevantes:")
    print("- radiacion_solar_kwh_m2, velocidad_viento_m_s")
    print("- consumo_kwh, students, altitude")
    print("- area_disponible_m2, es_hora_clases")

    print(f"\n🎯 SIGUIENTES PASOS:")
    print("1. Ejecutar visualizaciones: python visualization_dashboard.py")
    print("2. Entrenar modelo ML: python modelo_ml_dimensionamiento.py")
    print("3. Usar utilidades: python utilidades_sistema_renovables.py")

    print(f"\n🎉 ¡GENERACIÓN COMPLETADA CON ÉXITO!")
    print(f"Dataset final: {len(df_ml):,} registros × {len(df_ml.columns)} variables")

    # Estadísticas detalladas
    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DETALLADAS")
    print("=" * 60)
    print(
        f"Período completo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"Total días simulados: {len(dates):,}")
    print(f"Registros por tabla:")
    print(f"  • Clima (horario): {len(df_climate):,}")
    print(f"  • Consumo (horario): {len(df_consumption):,}")
    print(f"  • Generación (horario): {len(df_generation):,}")
    print(f"  • Balance (horario): {len(df_balance):,}")
    print(f"  • Economía (horario): {len(df_economics):,}")
    print(f"  • Dataset ML final: {len(df_ml):,}")
    print(f"  • Equipos en catálogo: {len(df_equipment)}")

    # Verificación de datos por escuela
    print(f"\n🏫 VERIFICACIÓN POR ESCUELA:")
    print("-" * 50)
    for school, params in school_params.items():
        school_data = df_systems[df_systems["escuela"] == school].iloc[0]
        school_ml_count = len(df_ml[df_ml["escuela"] == school])

        print(f"\n📍 {school}")
        print(f"   Estudiantes: {params['students']}")
        print(f"   Altitud: {params['altitude']}m")
        print(f"   Área: {params['area_disponible_m2']}m²")
        print(f"   Solar: {school_data['capacidad_solar_kw']:.1f} kW")
        print(f"   Eólico: {school_data['capacidad_eolica_kw']:.1f} kW")
        print(f"   Baterías: {school_data['capacidad_bateria_kwh']:.1f} kWh")
        print(f"   Inversión: ${school_data['costo_total_usd']:,.0f}")
        print(f"   Registros ML: {school_ml_count:,}")

    # Verificar integridad del dataset ML
    print(f"\n🔍 VERIFICACIÓN INTEGRIDAD DATASET ML:")
    print("-" * 50)

    # Verificar valores nulos
    null_counts = df_ml.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    if len(columns_with_nulls) == 0:
        print("   ✅ Sin valores nulos")
    else:
        print("   ⚠️ Columnas con valores nulos:")
        for col, count in columns_with_nulls.items():
            print(f"      - {col}: {count:,} ({count/len(df_ml)*100:.1f}%)")

    # Verificar rangos de valores clave
    key_columns = [
        "radiacion_solar_kwh_m2",
        "velocidad_viento_m_s",
        "temperatura_c",
        "generacion_total_kwh",
        "consumo_kwh",
        "costo_total_usd",
    ]

    print(f"\n   📊 Rangos de valores clave:")
    for col in key_columns:
        if col in df_ml.columns:
            min_val = df_ml[col].min()
            max_val = df_ml[col].max()
            mean_val = df_ml[col].mean()
            print(f"      {col}: {min_val:.3f} - {max_val:.3f} (μ={mean_val:.3f})")

    # Verificar distribución temporal
    date_range = df_ml["fecha"].dt.date.unique()
    print(f"\n   📅 Rango temporal: {len(date_range):,} días únicos")
    print(f"      Desde: {date_range.min()}")
    print(f"      Hasta: {date_range.max()}")

    # Estadísticas por hora
    hourly_counts = df_ml["hora"].value_counts().sort_index()
    print(
        f"\n   ⏰ Distribución horaria: {hourly_counts.min():,} - {hourly_counts.max():,} registros/hora"
    )

    print(f"\n✅ ARCHIVOS GENERADOS EXITOSAMENTE:")
    print(f"   📊 Excel: {excel_filename}")
    print(f"   🤖 CSV ML: {csv_filename}")
    print(f"   📁 Directorio: datos_sintéticos_chimborazo/")

    print(f"\n🔬 MODELO ML RECOMENDADO:")
    print("-" * 40)
    print("Para dimensionamiento óptimo usar:")
    print("1. 🏆 Random Forest Regressor (principal)")
    print("2. 🥈 Gradient Boosting Regressor (ensemble)")
    print("3. 🥉 XGBoost Regressor (alternativa)")

    print(f"\nVariables objetivo principales:")
    print("- capacidad_solar_kw")
    print("- capacidad_eolica_kw")
    print("- capacidad_bateria_kwh")
    print("- costo_total_usd")

    print(f"\nFeatures más relevantes:")
    print("- radiacion_solar_kwh_m2, velocidad_viento_m_s")
    print("- consumo_kwh, students, altitude")
    print("- area_disponible_m2, es_hora_clases")

    print(f"\n🎯 SIGUIENTES PASOS:")
    print("1. Ejecutar visualizaciones: python visualization_dashboard.py")
    print("2. Entrenar modelo ML: python modelo_ml_dimensionamiento.py")
    print("3. Usar utilidades: python utilidades_sistema_renovables.py")


def validate_generated_data(
    df_climate, df_consumption, df_generation, df_balance, df_economics
):
    """Validar coherencia de los datos generados"""
    print("\n🔍 VALIDANDO COHERENCIA DE DATOS...")
    print("-" * 50)

    validation_errors = []
    validation_warnings = []

    # 1. Verificar rangos de datos climáticos
    if df_climate["radiacion_solar_kwh_m2"].max() > 8:
        validation_warnings.append("Radiación solar muy alta (>8 kWh/m²)")

    if df_climate["radiacion_solar_kwh_m2"].min() < 0:
        validation_errors.append("Radiación solar negativa detectada")

    if df_climate["velocidad_viento_m_s"].max() > 30:
        validation_warnings.append("Velocidad de viento muy alta (>30 m/s)")

    if (
        df_climate["temperatura_c"].min() < -10
        or df_climate["temperatura_c"].max() > 40
    ):
        validation_warnings.append("Temperaturas fuera del rango típico de Chimborazo")

    # 2. Verificar coherencia energética
    negative_consumption = (df_consumption["consumo_kwh"] < 0).sum()
    if negative_consumption > 0:
        validation_errors.append(
            f"Consumo negativo en {negative_consumption} registros"
        )

    negative_generation = (df_generation["generacion_total_kwh"] < 0).sum()
    if negative_generation > 0:
        validation_errors.append(
            f"Generación negativa en {negative_generation} registros"
        )

    # 3. Verificar radiación nocturna
    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    night_radiation = df_climate[df_climate["hora"].isin(night_hours)][
        "radiacion_solar_kwh_m2"
    ]
    if (night_radiation > 0.01).sum() > 0:
        validation_warnings.append("Radiación solar detectada en horas nocturnas")

    # 4. Verificar balance energético
    extreme_autosuficiencia = (df_balance["autosuficiencia_horaria_pct"] > 500).sum()
    if extreme_autosuficiencia > 0:
        validation_warnings.append(
            f"Autosuficiencia >500% en {extreme_autosuficiencia} registros"
        )

    # 5. Verificar costos
    zero_costs = (df_economics["costo_energia_red_usd"] == 0).sum()
    total_records = len(df_economics)
    if zero_costs / total_records > 0.8:
        validation_warnings.append(
            "Demasiados registros con costo cero (posible sobredimensionamiento)"
        )

    # 6. Verificar distribución temporal
    unique_dates = df_climate["fecha"].nunique()
    expected_dates = len(dates)
    if unique_dates != expected_dates:
        validation_errors.append(
            f"Fechas faltantes: esperadas {expected_dates}, encontradas {unique_dates}"
        )

    # 7. Verificar integridad por escuela
    for school in school_params.keys():
        school_records = len(df_climate[df_climate["escuela"] == school])
        expected_records = len(dates) * 24  # Horario completo
        if school_records != expected_records:
            validation_errors.append(
                f"{school}: {school_records} registros, esperados {expected_records}"
            )

    # Mostrar resultados
    if validation_errors:
        print("❌ ERRORES CRÍTICOS:")
        for error in validation_errors:
            print(f"   • {error}")
    else:
        print("✅ Sin errores críticos detectados")

    if validation_warnings:
        print("\n⚠️ ADVERTENCIAS:")
        for warning in validation_warnings:
            print(f"   • {warning}")
    else:
        print("✅ Sin advertencias")

    # Estadísticas de calidad
    print(f"\n📊 ESTADÍSTICAS DE CALIDAD:")
    print(f"   • Completitud temporal: {unique_dates/expected_dates*100:.1f}%")
    print(
        f"   • Registros con radiación diurna: {((df_climate['hora'].between(6, 18)) & (df_climate['radiacion_solar_kwh_m2'] > 0)).sum():,}"
    )
    print(
        f"   • Registros con radiación nocturna: {((~df_climate['hora'].between(6, 18)) & (df_climate['radiacion_solar_kwh_m2'] > 0)).sum():,}"
    )
    print(
        f"   • Balance energético promedio: {df_balance['balance_energetico_kwh'].mean():.3f} kWh"
    )
    print(
        f"   • Autosuficiencia promedio: {df_balance['autosuficiencia_horaria_pct'].mean():.1f}%"
    )

    return (
        len(validation_errors) == 0
    )  # True si no hay errores críticos. Creando catálogo de equipos...")


if __name__ == "__main__":
    main()
