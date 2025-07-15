import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Configuración de la simulación
np.random.seed(42)  # Para reproducibilidad
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 6, 30)
dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]

# Parámetros específicos para centros educativos rurales de Chimborazo
# Basado en datos reales de la provincia
school_params = {
    "Escuela 21 de Abril": {
        "size": "small",
        "students": 85,
        "altitude": 2650,  # Zona rural baja de Chimborazo
        "terrain": "valle",
        "canton": "Riobamba",
        "area_terreno_m2": 1200,
        "area_construida_m2": 350,
        "niveles_educativos": ["inicial", "basica"],
        "jornada": "matutina",
        "tipo_construccion": "mixta",
    },
    "Escuela Sangay": {
        "size": "medium",
        "students": 150,
        "altitude": 3100,  # Zona media de Chimborazo
        "terrain": "montañoso",
        "canton": "Penipe",
        "area_terreno_m2": 2800,
        "area_construida_m2": 650,
        "niveles_educativos": ["inicial", "basica", "bachillerato"],
        "jornada": "completa",
        "tipo_construccion": "hormigon",
    },
    "Politécnica de Chimborazo": {
        "size": "large",
        "students": 280,
        "altitude": 2754,  # Riobamba urbano-marginal
        "terrain": "planicie",
        "canton": "Riobamba",
        "area_terreno_m2": 4500,
        "area_construida_m2": 1200,
        "niveles_educativos": ["bachillerato_tecnico", "superior"],
        "jornada": "completa",
        "tipo_construccion": "hormigon",
    },
    "Colegio Condorazo": {
        "size": "medium",
        "students": 190,
        "altitude": 3350,  # Zona alta de Chimborazo
        "terrain": "montañoso",
        "canton": "Guano",
        "area_terreno_m2": 3200,
        "area_construida_m2": 780,
        "niveles_educativos": ["basica", "bachillerato"],
        "jornada": "matutina",
        "tipo_construccion": "mixta",
    },
    "Colegio Victor Proaño": {
        "size": "small",
        "students": 120,
        "altitude": 2890,  # Zona intermedia
        "terrain": "valle",
        "canton": "Chambo",
        "area_terreno_m2": 1800,
        "area_construida_m2": 520,
        "niveles_educativos": ["basica", "bachillerato"],
        "jornada": "vespertina",
        "tipo_construccion": "mixta",
    },
}

# Crear directorio para guardar datos si no existe
if not os.path.exists("datos_simulacion_chimborazo"):
    os.makedirs("datos_simulacion_chimborazo")


def generate_climate_data():
    """
    Genera datos climáticos realistas para la provincia de Chimborazo
    Considera la variabilidad climática de la región andina ecuatoriana
    """
    climate_data = []

    for school_name, params in school_params.items():
        # Parámetros climáticos específicos de Chimborazo
        # Radiación solar base según altitud (mayor en altitudes elevadas)
        base_radiation = 4.2 + (params["altitude"] - 2500) * 0.0004

        # Velocidad del viento según terreno y altitud
        if params["terrain"] == "montañoso":
            base_wind = 3.8 + (params["altitude"] - 2500) * 0.0008
        elif params["terrain"] == "valle":
            base_wind = 2.5 + (params["altitude"] - 2500) * 0.0005
        else:  # planicie
            base_wind = 3.2 + (params["altitude"] - 2500) * 0.0006

        # Temperatura base según altitud (gradiente térmico andino)
        temp_base = 23 - (params["altitude"] - 2500) * 0.0065  # -6.5°C/km

        for day, date in enumerate(dates):
            # Modelar estacionalidad ecuatorial (dos estaciones: seca y lluviosa)
            # Estación seca: junio-septiembre, diciembre-febrero
            # Estación lluviosa: marzo-mayo, octubre-noviembre

            day_of_year = date.timetuple().tm_yday

            # Factor estacional para radiación (más alta en estación seca)
            if (day_of_year >= 152 and day_of_year <= 273) or (
                day_of_year >= 335 or day_of_year <= 59
            ):
                # Estación seca
                seasonal_radiation_factor = 1.15
                rain_probability = 0.15
            else:
                # Estación lluviosa
                seasonal_radiation_factor = 0.85
                rain_probability = 0.45

            # Radiación solar diaria
            daily_radiation = base_radiation * seasonal_radiation_factor
            daily_radiation += np.random.normal(0, 0.6)  # Variabilidad diaria
            daily_radiation = max(daily_radiation, 0.8)  # Valor mínimo razonable

            # Velocidad del viento (mayor variabilidad en zona andina)
            wind_seasonal_factor = 1.2 if seasonal_radiation_factor > 1 else 0.9
            daily_wind = base_wind * wind_seasonal_factor
            daily_wind += np.random.normal(0, 1.2)
            daily_wind = max(daily_wind, 0.5)

            # Temperatura diaria
            temp_seasonal_variation = 3 * np.sin(2 * np.pi * day_of_year / 365)
            daily_temp = temp_base + temp_seasonal_variation
            daily_temp += np.random.normal(0, 2.5)  # Variabilidad diaria

            # Humedad relativa (alta en estación lluviosa)
            base_humidity = 0.65 if seasonal_radiation_factor < 1 else 0.45
            daily_humidity = base_humidity + np.random.normal(0, 0.1)
            daily_humidity = np.clip(daily_humidity, 0.2, 0.95)

            # Nubosidad correlacionada con humedad
            cloud_cover = 0.3 + 0.5 * daily_humidity + np.random.normal(0, 0.15)
            cloud_cover = np.clip(cloud_cover, 0, 1)

            # Precipitación
            if np.random.random() < rain_probability:
                if seasonal_radiation_factor < 1:  # Estación lluviosa
                    precipitation = np.random.gamma(2, 8)  # Lluvias más intensas
                else:  # Estación seca
                    precipitation = np.random.gamma(1, 3)  # Lluvias ligeras
            else:
                precipitation = 0

            # Presión atmosférica (varía con altitud)
            pressure_base = (
                1013.25 * (1 - 0.0065 * params["altitude"] / 288.15) ** 5.255
            )
            daily_pressure = pressure_base + np.random.normal(0, 5)

            climate_data.append(
                {
                    "fecha": date,
                    "escuela": school_name,
                    "radiacion_solar_kwh_m2": round(daily_radiation, 2),
                    "velocidad_viento_m_s": round(daily_wind, 2),
                    "temperatura_c": round(daily_temp, 1),
                    "humedad_relativa": round(daily_humidity, 2),
                    "nubosidad": round(cloud_cover, 2),
                    "precipitacion_mm": round(precipitation, 1),
                    "presion_atmosferica_hpa": round(daily_pressure, 1),
                    "altitud_m": params["altitude"],
                    "canton": params["canton"],
                }
            )

    return pd.DataFrame(climate_data)


def generate_consumption_data():
    """
    Genera patrones de consumo realistas para centros educativos rurales
    """
    consumption_data = []

    # Parámetros de consumo por tipo de actividad (kW)
    consumption_profiles = {
        "aulas": 0.8,  # Iluminación LED + ventiladores
        "laboratorio": 2.5,  # Equipos de cómputo + instrumentos
        "biblioteca": 1.2,  # Iluminación + equipos audiovisuales
        "cocina": 4.0,  # Equipos de cocina
        "administracion": 1.5,  # Oficinas + equipos de cómputo
        "auditorio": 3.0,  # Sistema de sonido + iluminación
        "seguridad": 0.5,  # Sistemas de seguridad (24h)
    }

    for school_name, params in school_params.items():
        # Calcular espacios según el tamaño de la escuela
        num_aulas = max(4, params["students"] // 25)
        tiene_laboratorio = params["size"] in ["medium", "large"]
        tiene_biblioteca = True
        tiene_cocina = params["jornada"] in ["completa", "matutina"]
        tiene_auditorio = params["size"] == "large"

        # Horarios según jornada escolar
        if params["jornada"] == "matutina":
            horario_activo = list(range(7, 13))
        elif params["jornada"] == "vespertina":
            horario_activo = list(range(13, 19))
        else:  # completa
            horario_activo = list(range(7, 17))

        for date in dates:
            # Verificar si es día laborable
            es_laborable = date.weekday() < 5  # Lunes a viernes

            # Verificar vacaciones escolares (Ecuador)
            mes = date.month
            es_vacaciones = (
                mes in [7, 8]
                or (mes >= 12 and date.day > 15)
                or (mes == 1 and date.day < 15)
            )

            # Factor de actividad
            if not es_laborable or es_vacaciones:
                factor_actividad = 0.1  # Solo seguridad y mantenimiento
            else:
                factor_actividad = 1.0

            for hora in range(24):
                consumo_total = 0

                # Consumo base (seguridad) - 24 horas
                consumo_total += consumption_profiles["seguridad"]

                if factor_actividad > 0.1:  # Día laborable no vacacional
                    if hora in horario_activo:
                        # Aulas activas
                        aulas_activas = int(num_aulas * 0.8)  # 80% de aulas en uso
                        consumo_total += aulas_activas * consumption_profiles["aulas"]

                        # Laboratorio
                        if tiene_laboratorio and hora in range(9, 16):
                            consumo_total += consumption_profiles["laboratorio"] * 0.6

                        # Biblioteca
                        if tiene_biblioteca:
                            consumo_total += consumption_profiles["biblioteca"] * 0.4

                        # Cocina (horarios de comida)
                        if tiene_cocina and hora in [10, 11, 12, 13]:
                            consumo_total += consumption_profiles["cocina"] * 0.8

                        # Administración
                        if hora in range(8, 16):
                            consumo_total += consumption_profiles["administracion"]

                        # Auditorio (ocasional)
                        if tiene_auditorio and np.random.random() < 0.1:
                            consumo_total += consumption_profiles["auditorio"] * 0.5

                    elif hora in range(18, 21):  # Actividades nocturnas ocasionales
                        if np.random.random() < 0.2:  # 20% probabilidad
                            consumo_total += (
                                num_aulas * 0.3 * consumption_profiles["aulas"]
                            )

                # Agregar variabilidad aleatoria (±15%)
                consumo_total *= 0.85 + 0.3 * np.random.random()

                # Escalar según número de estudiantes
                factor_escala = params["students"] / 150  # Base 150 estudiantes
                consumo_total *= factor_escala

                consumption_data.append(
                    {
                        "fecha": date,
                        "hora": hora,
                        "escuela": school_name,
                        "consumo_kwh": round(consumo_total, 3),
                        "factor_actividad": factor_actividad,
                        "es_laborable": es_laborable,
                        "es_vacaciones": es_vacaciones,
                    }
                )

    return pd.DataFrame(consumption_data)


def generate_renewable_system_data():
    """
    Genera especificaciones de sistemas renovables dimensionados para cada escuela
    """
    systems_data = []

    for school_name, params in school_params.items():
        # Dimensionamiento según demanda estimada y área disponible
        demanda_estimada_diaria = (
            params["students"] * 0.15
        )  # kWh/estudiante/día (rural)

        # Área disponible para paneles (30% del área de terreno libre)
        area_libre = params["area_terreno_m2"] - params["area_construida_m2"]
        area_paneles_disponible = area_libre * 0.3

        # Capacidad solar (considerando irradiación de Chimborazo ~4.5 kWh/m²/día)
        irradiacion_promedio = 4.5
        # Área requerida: ~7 m²/kWp para paneles modernos
        capacidad_solar_maxima = area_paneles_disponible / 7

        # Dimensionar para cubrir 80-120% de la demanda
        factor_sobredimensionamiento = np.random.uniform(0.8, 1.2)
        capacidad_solar = min(
            capacidad_solar_maxima,
            demanda_estimada_diaria
            / irradiacion_promedio
            * factor_sobredimensionamiento,
        )

        # Capacidad eólica (complementaria, 20-40% de la solar)
        capacidad_eolica = capacidad_solar * np.random.uniform(0.2, 0.4)

        # Capacidad de baterías (1-3 días de autonomía)
        dias_autonomia = np.random.uniform(1, 3)
        capacidad_bateria = demanda_estimada_diaria * dias_autonomia

        # Características técnicas actualizadas
        eficiencia_solar = np.random.uniform(0.18, 0.22)  # Paneles modernos
        eficiencia_eolica = np.random.uniform(0.30, 0.45)
        eficiencia_bateria = np.random.uniform(0.90, 0.95)  # Baterías de litio

        # Edad del sistema
        fecha_instalacion = np.random.choice([2020, 2021, 2022, 2023])
        system_age = 2025 - fecha_instalacion

        # Degradación acumulada
        degradacion_solar = 0.005 * system_age  # 0.5% anual
        degradacion_bateria = 0.02 * system_age  # 2% anual

        # Especificaciones técnicas
        panel_type = np.random.choice(
            ["monocristalino", "policristalino"], p=[0.7, 0.3]
        )
        battery_type = np.random.choice(["litio", "plomo-acido"], p=[0.6, 0.4])
        wind_type = np.random.choice(["eje_horizontal", "eje_vertical"], p=[0.8, 0.2])

        # Inversor
        capacidad_inversor = capacidad_solar * 1.2  # 20% sobredimensionamiento
        eficiencia_inversor = np.random.uniform(0.94, 0.97)

        # Estructura de montaje
        tipo_montaje = np.random.choice(["tierra", "techo"], p=[0.7, 0.3])

        systems_data.append(
            {
                "escuela": school_name,
                "capacidad_solar_kWp": round(capacidad_solar, 2),
                "capacidad_eolica_kW": round(capacidad_eolica, 2),
                "capacidad_bateria_kWh": round(capacidad_bateria, 2),
                "capacidad_inversor_kW": round(capacidad_inversor, 2),
                "eficiencia_solar": round(eficiencia_solar, 3),
                "eficiencia_eolica": round(eficiencia_eolica, 3),
                "eficiencia_bateria": round(eficiencia_bateria, 3),
                "eficiencia_inversor": round(eficiencia_inversor, 3),
                "edad_sistema_años": system_age,
                "fecha_instalacion": fecha_instalacion,
                "degradacion_solar": round(degradacion_solar, 3),
                "degradacion_bateria": round(degradacion_bateria, 3),
                "tipo_panel": panel_type,
                "tipo_bateria": battery_type,
                "tipo_aerogenerador": wind_type,
                "tipo_montaje": tipo_montaje,
                "area_paneles_m2": round(capacidad_solar * 7, 1),
                "area_disponible_m2": round(area_paneles_disponible, 1),
                "demanda_estimada_kwh_dia": round(demanda_estimada_diaria, 2),
                "altitud_m": params["altitude"],
                "terreno": params["terrain"],
                "canton": params["canton"],
            }
        )

    return pd.DataFrame(systems_data)


def generate_power_generation_data(df_climate, df_systems):
    """
    Calcula generación de energía basada en condiciones climáticas y especificaciones del sistema
    """
    generation_data = []

    # Agrupar datos climáticos
    climate_grouped = df_climate.groupby(["escuela", "fecha"])

    for (school_name, date), climate_group in climate_grouped:
        system_params = df_systems[df_systems["escuela"] == school_name].iloc[0]
        climate_day = climate_group.iloc[0]

        # === GENERACIÓN SOLAR ===
        capacidad_solar = system_params["capacidad_solar_kWp"]
        eficiencia_solar = system_params["eficiencia_solar"] * (
            1 - system_params["degradacion_solar"]
        )
        eficiencia_inversor = system_params["eficiencia_inversor"]

        radiacion = climate_day["radiacion_solar_kwh_m2"]
        temperatura = climate_day["temperatura_c"]
        nubosidad = climate_day["nubosidad"]

        # Factor de temperatura (coeficiente -0.4%/°C típico)
        temp_factor = 1 - max(0, (temperatura - 25) * 0.004)

        # Factor de nubosidad
        cloud_factor = 1 - nubosidad * 0.4

        # Factor de limpieza (afectado por lluvia)
        lluvia_ayer = 1 if climate_day["precipitacion_mm"] > 2 else 0
        factor_limpieza = 0.95 + 0.05 * lluvia_ayer  # Mejora con lluvia

        # Factor de altitud (mejor radiación en altitud)
        altitude_factor = 1 + (system_params["altitud_m"] - 2500) * 0.00005

        # Generación solar
        generacion_solar = (
            capacidad_solar
            * radiacion
            * eficiencia_solar
            * temp_factor
            * cloud_factor
            * factor_limpieza
            * altitude_factor
            * eficiencia_inversor
        )

        # === GENERACIÓN EÓLICA ===
        capacidad_eolica = system_params["capacidad_eolica_kW"]
        eficiencia_eolica = system_params["eficiencia_eolica"]
        velocidad_viento = climate_day["velocidad_viento_m_s"]
        presion = climate_day["presion_atmosferica_hpa"]

        # Curva de potencia del aerogenerador
        cut_in_speed = 2.5  # m/s
        rated_speed = 12.0  # m/s
        cut_out_speed = 25.0  # m/s

        if velocidad_viento < cut_in_speed or velocidad_viento > cut_out_speed:
            wind_factor = 0
        elif velocidad_viento < rated_speed:
            wind_factor = (
                (velocidad_viento - cut_in_speed) / (rated_speed - cut_in_speed)
            ) ** 3
        else:
            wind_factor = 1.0

        # Factor de densidad del aire (altitud)
        densidad_factor = presion / 1013.25

        # Horas equivalentes de viento
        horas_viento = (
            np.random.uniform(6, 12) if velocidad_viento > cut_in_speed else 3
        )

        # Generación eólica
        generacion_eolica = (
            capacidad_eolica
            * wind_factor
            * eficiencia_eolica
            * densidad_factor
            * horas_viento
            / 24
        )

        # Total de generación
        generacion_total = generacion_solar + generacion_eolica

        generation_data.append(
            {
                "fecha": date,
                "escuela": school_name,
                "generacion_solar_kwh": round(max(0, generacion_solar), 2),
                "generacion_eolica_kwh": round(max(0, generacion_eolica), 2),
                "generacion_total_kwh": round(max(0, generacion_total), 2),
                "factor_temperatura": round(temp_factor, 3),
                "factor_nubosidad": round(cloud_factor, 3),
                "factor_viento": round(wind_factor, 3),
                "horas_sol_equivalentes": round(radiacion, 2),
                "horas_viento_equivalentes": round(horas_viento, 2),
            }
        )

    return pd.DataFrame(generation_data)


def generate_energy_balance(df_consumption, df_generation, df_systems):
    """
    Calcula el balance energético incluyendo almacenamiento en baterías
    """
    # Agregar consumo diario
    daily_consumption = (
        df_consumption.groupby(["fecha", "escuela"])["consumo_kwh"].sum().reset_index()
    )
    daily_consumption.rename(
        columns={"consumo_kwh": "consumo_diario_kwh"}, inplace=True
    )

    # Unir con generación
    df_balance = pd.merge(df_generation, daily_consumption, on=["fecha", "escuela"])

    # Agregar capacidad de baterías
    df_balance = pd.merge(
        df_balance,
        df_systems[["escuela", "capacidad_bateria_kWh", "eficiencia_bateria"]],
        on="escuela",
    )

    # Calcular balance energético con almacenamiento
    balance_data = []

    # Procesar por escuela para simular almacenamiento temporal
    for school in df_balance["escuela"].unique():
        school_data = df_balance[df_balance["escuela"] == school].sort_values("fecha")
        capacidad_bateria = school_data["capacidad_bateria_kWh"].iloc[0]
        eficiencia_bateria = school_data["eficiencia_bateria"].iloc[0]

        # Estado inicial de batería (50% de capacidad)
        estado_bateria = capacidad_bateria * 0.5

        for _, row in school_data.iterrows():
            generacion = row["generacion_total_kwh"]
            consumo = row["consumo_diario_kwh"]

            # Balance inicial
            balance_inicial = generacion - consumo

            if balance_inicial > 0:  # Exceso de generación
                # Cargar batería
                energia_a_cargar = min(
                    balance_inicial * eficiencia_bateria,
                    capacidad_bateria - estado_bateria,
                )
                estado_bateria += energia_a_cargar
                excedente = balance_inicial - energia_a_cargar / eficiencia_bateria
                deficit = 0
                energia_red = 0
                energia_bateria_usada = 0

            else:  # Déficit de generación
                deficit_inicial = -balance_inicial
                # Usar energía de batería
                energia_disponible_bateria = estado_bateria * eficiencia_bateria
                energia_bateria_usada = min(deficit_inicial, energia_disponible_bateria)
                estado_bateria -= energia_bateria_usada / eficiencia_bateria

                # Energía restante de la red
                energia_red = max(0, deficit_inicial - energia_bateria_usada)
                deficit = deficit_inicial
                excedente = 0

            # Autodescarga de batería (0.1% diario)
            estado_bateria *= 0.999

            # Calcular métricas
            autosuficiencia = min(
                100, (generacion + energia_bateria_usada) / consumo * 100
            )

            balance_data.append(
                {
                    "fecha": row["fecha"],
                    "escuela": school,
                    "generacion_total_kwh": generacion,
                    "consumo_diario_kwh": consumo,
                    "excedente_energetico_kwh": round(excedente, 2),
                    "deficit_energetico_kwh": round(deficit, 2),
                    "energia_red_kwh": round(energia_red, 2),
                    "energia_bateria_usada_kwh": round(energia_bateria_usada, 2),
                    "estado_bateria_kwh": round(estado_bateria, 2),
                    "porcentaje_autosuficiencia": round(autosuficiencia, 2),
                    "soc_bateria": round(estado_bateria / capacidad_bateria * 100, 1),
                }
            )

    return pd.DataFrame(balance_data)


def generate_economics_data(df_systems, df_balance):
    """
    Genera análisis económico y de emisiones
    """
    # Parámetros económicos para Ecuador
    tarifa_electrica = 0.10  # USD/kWh (actualizado)
    factor_co2_red = 0.385  # kg CO2/kWh (Ecuador)
    tasa_descuento = 0.08  # 8% anual

    # Costos de inversión actualizados (USD)
    system_costs = []

    for _, system in df_systems.iterrows():
        # Costos unitarios actualizados 2025
        costo_solar_unitario = np.random.uniform(800, 1200)  # USD/kWp
        costo_eolico_unitario = np.random.uniform(1500, 2200)  # USD/kW

        if system["tipo_bateria"] == "litio":
            costo_bateria_unitario = np.random.uniform(600, 900)  # USD/kWh
        else:
            costo_bateria_unitario = np.random.uniform(300, 500)  # USD/kWh

        costo_inversor_unitario = np.random.uniform(200, 400)  # USD/kW

        # Inversiones
        inversion_solar = system["capacidad_solar_kWp"] * costo_solar_unitario
        inversion_eolica = system["capacidad_eolica_kW"] * costo_eolico_unitario
        inversion_baterias = system["capacidad_bateria_kWh"] * costo_bateria_unitario
        inversion_inversor = system["capacidad_inversor_kW"] * costo_inversor_unitario

        # Costos adicionales
        costo_instalacion = (
            inversion_solar + inversion_eolica + inversion_baterias + inversion_inversor
        ) * 0.25
        costo_ingenieria = (
            inversion_solar + inversion_eolica + inversion_baterias + inversion_inversor
        ) * 0.10

        inversion_total = (
            inversion_solar
            + inversion_eolica
            + inversion_baterias
            + inversion_inversor
            + costo_instalacion
            + costo_ingenieria
        )

        # Costos O&M anuales
        costo_om_solar = inversion_solar * 0.015  # 1.5% anual
        costo_om_eolico = inversion_eolica * 0.035  # 3.5% anual
        costo_om_baterias = inversion_baterias * 0.02  # 2% anual
        costo_om_total = costo_om_solar + costo_om_eolico + costo_om_baterias

        # Vidas útiles
        vida_util_solar = 25
        vida_util_eolica = 20
        vida_util_baterias = 10 if system["tipo_bateria"] == "litio" else 7
        vida_util_inversor = 15

        system_costs.append(
            {
                "escuela": system["escuela"],
                "inversion_solar_usd": round(inversion_solar, 2),
                "inversion_eolica_usd": round(inversion_eolica, 2),
                "inversion_baterias_usd": round(inversion_baterias, 2),
                "inversion_inversor_usd": round(inversion_inversor, 2),
                "costo_instalacion_usd": round(costo_instalacion, 2),
                "costo_ingenieria_usd": round(costo_ingenieria, 2),
                "inversion_total_usd": round(inversion_total, 2),
                "costo_om_anual_usd": round(costo_om_total, 2),
                "vida_util_solar_años": vida_util_solar,
                "vida_util_eolica_años": vida_util_eolica,
                "vida_util_baterias_años": vida_util_baterias,
                "vida_util_inversor_años": vida_util_inversor,
                "lcoe_estimado_usd_kwh": round(
                    (inversion_total + costo_om_total * vida_util_solar)
                    / (system["demanda_estimada_kwh_dia"] * 365 * vida_util_solar),
                    3,
                ),
            }
        )

    df_costs = pd.DataFrame(system_costs)

    # Calcular impactos económicos y ambientales diarios
    operational_data = []

    for _, balance in df_balance.iterrows():
        energia_red = balance["energia_red_kwh"]
        generacion_renovable = balance["generacion_total_kwh"]

        # Costos diarios
        costo_energia_red = energia_red * tarifa_electrica
        ahorro_energia = generacion_renovable * tarifa_electrica

        # Emisiones CO2
        co2_evitado = generacion_renovable * factor_co2_red
        co2_emitido = energia_red * factor_co2_red

        operational_data.append(
            {
                "fecha": balance["fecha"],
                "escuela": balance["escuela"],
                "costo_energia_red_usd": round(costo_energia_red, 3),
                "ahorro_energia_usd": round(ahorro_energia, 3),
                "co2_evitado_kg": round(co2_evitado, 2),
                "co2_emitido_kg": round(co2_emitido, 2),
                "co2_neto_evitado_kg": round(co2_evitado - co2_emitido, 2),
            }
        )

    df_operational = pd.DataFrame(operational_data)

    return df_costs, df_operational


def generate_performance_metrics(df_balance, df_generation, df_systems):
    """
    Genera métricas de rendimiento del sistema
    """
    performance_data = []

    for school in df_balance["escuela"].unique():
        school_balance = df_balance[df_balance["escuela"] == school]
        school_generation = df_generation[df_generation["escuela"] == school]
        system_specs = df_systems[df_systems["escuela"] == school].iloc[0]

        # Métricas mensuales
        for year in range(2021, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 6:  # Solo hasta junio 2025
                    break

                monthly_data = school_balance[
                    (school_balance["fecha"].dt.year == year)
                    & (school_balance["fecha"].dt.month == month)
                ]

                if len(monthly_data) == 0:
                    continue

                monthly_generation = school_generation[
                    (school_generation["fecha"].dt.year == year)
                    & (school_generation["fecha"].dt.month == month)
                ]

                # Calcular métricas
                generacion_total = monthly_data["generacion_total_kwh"].sum()
                consumo_total = monthly_data["consumo_diario_kwh"].sum()
                energia_red_total = monthly_data["energia_red_kwh"].sum()

                # Factor de capacidad
                dias_mes = len(monthly_data)
                capacidad_instalada = (
                    system_specs["capacidad_solar_kWp"]
                    + system_specs["capacidad_eolica_kW"]
                )
                factor_capacidad = (
                    generacion_total / (capacidad_instalada * 24 * dias_mes) * 100
                )

                # Productividad específica
                productividad_solar = (
                    monthly_generation["generacion_solar_kwh"].sum()
                    / system_specs["capacidad_solar_kWp"]
                )
                productividad_eolica = (
                    monthly_generation["generacion_eolica_kwh"].sum()
                    / system_specs["capacidad_eolica_kW"]
                )

                # Disponibilidad del sistema (simulada)
                disponibilidad = np.random.uniform(92, 98)  # 92-98%

                # Eficiencia del sistema
                eficiencia_sistema = (
                    generacion_total
                    / (generacion_total + monthly_data["energia_red_kwh"].sum())
                    * 100
                )

                performance_data.append(
                    {
                        "año": year,
                        "mes": month,
                        "escuela": school,
                        "generacion_total_kwh": round(generacion_total, 2),
                        "consumo_total_kwh": round(consumo_total, 2),
                        "autosuficiencia_promedio": round(
                            monthly_data["porcentaje_autosuficiencia"].mean(), 2
                        ),
                        "factor_capacidad_pct": round(factor_capacidad, 2),
                        "productividad_solar_kwh_kWp": round(productividad_solar, 2),
                        "productividad_eolica_kwh_kW": round(productividad_eolica, 2),
                        "disponibilidad_pct": round(disponibilidad, 2),
                        "eficiencia_sistema_pct": round(eficiencia_sistema, 2),
                        "energia_red_kwh": round(energia_red_total, 2),
                    }
                )

    return pd.DataFrame(performance_data)


def save_all_data():
    """
    Función principal que genera todos los datasets y los guarda
    """
    print("Generando datos sintéticos para sistemas híbridos en Chimborazo...")
    print("=" * 60)

    # 1. Generar datos climáticos
    print("1. Generando datos climáticos...")
    df_climate = generate_climate_data()
    df_climate.to_csv("datos_simulacion_chimborazo/datos_climaticos.csv", index=False)
    print(f"   ✓ Generados {len(df_climate)} registros climáticos")

    # 2. Generar datos de consumo
    print("2. Generando patrones de consumo...")
    df_consumption = generate_consumption_data()
    df_consumption.to_csv("datos_simulacion_chimborazo/datos_consumo.csv", index=False)
    print(f"   ✓ Generados {len(df_consumption)} registros de consumo")

    # 3. Generar especificaciones de sistemas
    print("3. Generando especificaciones de sistemas renovables...")
    df_systems = generate_renewable_system_data()
    df_systems.to_csv(
        "datos_simulacion_chimborazo/especificaciones_sistemas.csv", index=False
    )
    print(f"   ✓ Generadas especificaciones para {len(df_systems)} sistemas")

    # 4. Generar datos de generación
    print("4. Calculando generación de energía...")
    df_generation = generate_power_generation_data(df_climate, df_systems)
    df_generation.to_csv(
        "datos_simulacion_chimborazo/datos_generacion.csv", index=False
    )
    print(f"   ✓ Calculados {len(df_generation)} registros de generación")

    # 5. Calcular balance energético
    print("5. Calculando balance energético...")
    df_balance = generate_energy_balance(df_consumption, df_generation, df_systems)
    df_balance.to_csv("datos_simulacion_chimborazo/balance_energetico.csv", index=False)
    print(f"   ✓ Calculados {len(df_balance)} registros de balance")

    # 6. Generar análisis económico
    print("6. Generando análisis económico...")
    df_costs, df_operational = generate_economics_data(df_systems, df_balance)
    df_costs.to_csv("datos_simulacion_chimborazo/costos_inversion.csv", index=False)
    df_operational.to_csv(
        "datos_simulacion_chimborazo/impactos_economicos.csv", index=False
    )
    print(f"   ✓ Generado análisis económico para {len(df_costs)} sistemas")

    # 7. Generar métricas de rendimiento
    print("7. Calculando métricas de rendimiento...")
    df_performance = generate_performance_metrics(df_balance, df_generation, df_systems)
    df_performance.to_csv(
        "datos_simulacion_chimborazo/metricas_rendimiento.csv", index=False
    )
    print(f"   ✓ Calculadas {len(df_performance)} métricas mensuales")

    # 8. Generar resumen ejecutivo
    print("8. Generando resumen ejecutivo...")
    generate_executive_summary(df_systems, df_balance, df_costs, df_operational)

    print("\n" + "=" * 60)
    print("✅ Generación de datos completada exitosamente!")
    print(f"📁 Archivos guardados en: datos_simulacion_chimborazo/")
    print("\nArchivos generados:")
    archivos = [
        "datos_climaticos.csv",
        "datos_consumo.csv",
        "especificaciones_sistemas.csv",
        "datos_generacion.csv",
        "balance_energetico.csv",
        "costos_inversion.csv",
        "impactos_economicos.csv",
        "metricas_rendimiento.csv",
        "resumen_ejecutivo.csv",
    ]
    for archivo in archivos:
        print(f"   • {archivo}")


def generate_executive_summary(df_systems, df_balance, df_costs, df_operational):
    """
    Genera un resumen ejecutivo con estadísticas clave
    """
    summary_data = []

    for school in df_systems["escuela"].unique():
        system = df_systems[df_systems["escuela"] == school].iloc[0]
        balance = df_balance[df_balance["escuela"] == school]
        costs = df_costs[df_costs["escuela"] == school].iloc[0]
        operational = df_operational[df_operational["escuela"] == school]

        # Estadísticas anuales (promedio)
        generacion_anual = (
            balance["generacion_total_kwh"].sum() / 4.5 * 365
        )  # Escalar a año completo
        consumo_anual = balance["consumo_diario_kwh"].sum() / 4.5 * 365
        autosuficiencia_promedio = balance["porcentaje_autosuficiencia"].mean()
        energia_red_anual = balance["energia_red_kwh"].sum() / 4.5 * 365

        # Impactos económicos anuales
        ahorro_anual = operational["ahorro_energia_usd"].sum() / 4.5 * 365
        costo_red_anual = operational["costo_energia_red_usd"].sum() / 4.5 * 365

        # Impactos ambientales anuales
        co2_evitado_anual = operational["co2_evitado_kg"].sum() / 4.5 * 365

        # Ratios de rendimiento
        factor_capacidad_solar = (
            (balance["generacion_total_kwh"].sum() * 0.7)
            / (system["capacidad_solar_kWp"] * 24 * len(balance))
            * 100
        )

        # Payback simple
        payback_años = (
            costs["inversion_total_usd"] / ahorro_anual if ahorro_anual > 0 else 0
        )

        summary_data.append(
            {
                "escuela": school,
                "canton": system["canton"],
                "estudiantes": school_params[school]["students"],
                "altitud_m": system["altitud_m"],
                "capacidad_solar_kWp": system["capacidad_solar_kWp"],
                "capacidad_eolica_kW": system["capacidad_eolica_kW"],
                "capacidad_bateria_kWh": system["capacidad_bateria_kWh"],
                "area_paneles_m2": system["area_paneles_m2"],
                "inversion_total_usd": costs["inversion_total_usd"],
                "lcoe_usd_kwh": costs["lcoe_estimado_usd_kwh"],
                "generacion_anual_kwh": round(generacion_anual, 0),
                "consumo_anual_kwh": round(consumo_anual, 0),
                "autosuficiencia_promedio_pct": round(autosuficiencia_promedio, 1),
                "energia_red_anual_kwh": round(energia_red_anual, 0),
                "factor_capacidad_pct": round(factor_capacidad_solar, 1),
                "ahorro_anual_usd": round(ahorro_anual, 0),
                "costo_red_anual_usd": round(costo_red_anual, 0),
                "co2_evitado_anual_kg": round(co2_evitado_anual, 0),
                "payback_simple_años": round(payback_años, 1),
                "roi_anual_pct": (
                    round(ahorro_anual / costs["inversion_total_usd"] * 100, 1)
                    if costs["inversion_total_usd"] > 0
                    else 0
                ),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("datos_simulacion_chimborazo/resumen_ejecutivo.csv", index=False)

    return df_summary


def create_sample_analysis():
    """
    Crea gráficos de muestra para validar los datos generados
    """
    print("\n9. Creando análisis de muestra...")

    try:
        # Cargar datos
        df_climate = pd.read_csv("datos_simulacion_chimborazo/datos_climaticos.csv")
        df_balance = pd.read_csv("datos_simulacion_chimborazo/balance_energetico.csv")
        df_summary = pd.read_csv("datos_simulacion_chimborazo/resumen_ejecutivo.csv")

        # Convertir fechas
        df_climate["fecha"] = pd.to_datetime(df_climate["fecha"])
        df_balance["fecha"] = pd.to_datetime(df_balance["fecha"])

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Análisis de Datos Sintéticos - Sistemas Híbridos Chimborazo", fontsize=16
        )

        # 1. Radiación solar por escuela
        df_climate_monthly = (
            df_climate.groupby([df_climate["fecha"].dt.to_period("M"), "escuela"])[
                "radiacion_solar_kwh_m2"
            ]
            .mean()
            .reset_index()
        )
        df_climate_monthly["fecha"] = df_climate_monthly["fecha"].dt.to_timestamp()

        for school in df_climate["escuela"].unique():
            school_data = df_climate_monthly[df_climate_monthly["escuela"] == school]
            axes[0, 0].plot(
                school_data["fecha"],
                school_data["radiacion_solar_kwh_m2"],
                label=school.replace("Escuela ", "").replace("Colegio ", ""),
                linewidth=2,
            )

        axes[0, 0].set_title("Radiación Solar Promedio Mensual")
        axes[0, 0].set_ylabel("kWh/m²/día")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Autosuficiencia energética
        df_balance_monthly = (
            df_balance.groupby(
                [pd.to_datetime(df_balance["fecha"]).dt.to_period("M"), "escuela"]
            )["porcentaje_autosuficiencia"]
            .mean()
            .reset_index()
        )
        df_balance_monthly["fecha"] = df_balance_monthly["fecha"].dt.to_timestamp()

        for school in df_balance["escuela"].unique():
            school_data = df_balance_monthly[df_balance_monthly["escuela"] == school]
            axes[0, 1].plot(
                school_data["fecha"],
                school_data["porcentaje_autosuficiencia"],
                label=school.replace("Escuela ", "").replace("Colegio ", ""),
                linewidth=2,
            )

        axes[0, 1].set_title("Autosuficiencia Energética Promedio")
        axes[0, 1].set_ylabel("Porcentaje (%)")
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Inversión vs Capacidad
        axes[1, 0].scatter(
            df_summary["capacidad_solar_kWp"] + df_summary["capacidad_eolica_kW"],
            df_summary["inversion_total_usd"],
            s=df_summary["estudiantes"],
            alpha=0.7,
            c=df_summary["altitud_m"],
            cmap="viridis",
        )

        for i, row in df_summary.iterrows():
            axes[1, 0].annotate(
                row["escuela"].replace("Escuela ", "").replace("Colegio ", ""),
                (
                    row["capacidad_solar_kWp"] + row["capacidad_eolica_kW"],
                    row["inversion_total_usd"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        axes[1, 0].set_title("Inversión vs Capacidad Instalada")
        axes[1, 0].set_xlabel("Capacidad Total (kW)")
        axes[1, 0].set_ylabel("Inversión Total (USD)")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Retorno de inversión
        axes[1, 1].bar(
            range(len(df_summary)),
            df_summary["payback_simple_años"],
            color=plt.cm.RdYlGn_r(
                df_summary["payback_simple_años"]
                / df_summary["payback_simple_años"].max()
            ),
        )

        axes[1, 1].set_title("Período de Recuperación de Inversión")
        axes[1, 1].set_ylabel("Años")
        axes[1, 1].set_xticks(range(len(df_summary)))
        axes[1, 1].set_xticklabels(
            [
                name.replace("Escuela ", "").replace("Colegio ", "")
                for name in df_summary["escuela"]
            ],
            rotation=45,
        )
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "datos_simulacion_chimborazo/analisis_muestra.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("   ✓ Gráficos de análisis guardados en: analisis_muestra.png")

    except Exception as e:
        print(f"   ⚠ Error al crear gráficos: {e}")


# Ejecutar la generación completa de datos
if __name__ == "__main__":
    save_all_data()
    create_sample_analysis()

    print("\n📊 Estadísticas de los datos generados:")
    print("-" * 40)

    # Mostrar estadísticas básicas
    try:
        df_summary = pd.read_csv("datos_simulacion_chimborazo/resumen_ejecutivo.csv")

        print(f"Número de escuelas simuladas: {len(df_summary)}")
        print(f"Total de estudiantes: {df_summary['estudiantes'].sum()}")
        print(
            f"Capacidad solar total: {df_summary['capacidad_solar_kWp'].sum():.1f} kWp"
        )
        print(
            f"Capacidad eólica total: {df_summary['capacidad_eolica_kW'].sum():.1f} kW"
        )
        print(f"Inversión total: ${df_summary['inversion_total_usd'].sum():,.0f}")
        print(f"Ahorro anual promedio: ${df_summary['ahorro_anual_usd'].mean():,.0f}")
        print(
            f"Autosuficiencia promedio: {df_summary['autosuficiencia_promedio_pct'].mean():.1f}%"
        )
        print(f"CO2 evitado anual: {df_summary['co2_evitado_anual_kg'].sum():,.0f} kg")
        print(f"Payback promedio: {df_summary['payback_simple_años'].mean():.1f} años")

    except Exception as e:
        print(f"Error al mostrar estadísticas: {e}")

    print("\n🎯 Datos listos para simulación y análisis de optimización!")
    print("Los archivos CSV generados pueden ser utilizados directamente")
    print("en algoritmos de machine learning y modelos de optimización.")
