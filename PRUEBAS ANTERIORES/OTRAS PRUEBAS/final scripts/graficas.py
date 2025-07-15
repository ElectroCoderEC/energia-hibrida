import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")

# Configuración de estilo compatible con diferentes versiones
try:
    plt.style.use("seaborn-darkgrid")
except:
    try:
        plt.style.use("seaborn-darkgrid")
    except:
        plt.style.use("ggplot")

sns.set_palette("husl")

# Escuelas en zonas rurales de Chimborazo
school_params = {
    "Escuela 21 de Abril": {
        "estudiantes": 75,
        "area_total_m2": 1200,
        "area_techo_disponible_m2": 400,
        "area_terreno_disponible_m2": 800,
        "latitud": -1.65,
        "longitud": -78.65,
        "altitud": 3200,
        "terreno": "montañoso",
        "zona": "San Juan",
    },
    "Escuela Sangay": {
        "estudiantes": 150,
        "area_total_m2": 2500,
        "area_techo_disponible_m2": 800,
        "area_terreno_disponible_m2": 1500,
        "latitud": -1.70,
        "longitud": -78.70,
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


def load_data():
    """Carga los datos generados del archivo Excel"""
    file_path = "datos_energia_renovable/analisis_sistemas_hibridos_chimborazo.xlsx"

    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo {file_path}")
        print("Por favor, ejecute primero el script de generación de datos.")
        return None

    # Cargar hojas relevantes
    dfs = {
        "economics": pd.read_excel(file_path, sheet_name="Resumen_Economico"),
        "systems": pd.read_excel(file_path, sheet_name="Sistemas_Instalados"),
        "daily": pd.read_excel(file_path, sheet_name="Balance_Diario"),
        "monthly": pd.read_excel(file_path, sheet_name="Balance_Mensual"),
        "climate_sample": pd.read_excel(
            file_path, sheet_name="Datos_Climaticos_Muestra"
        ),
        "consumption_sample": pd.read_excel(
            file_path, sheet_name="Consumo_Energetico_Muestra"
        ),
    }

    # Convertir fechas
    dfs["daily"]["fecha"] = pd.to_datetime(dfs["daily"]["fecha"])
    dfs["monthly"]["año_mes"] = pd.to_datetime(dfs["monthly"]["año_mes"])
    dfs["climate_sample"]["fecha"] = pd.to_datetime(dfs["climate_sample"]["fecha"])
    dfs["consumption_sample"]["fecha"] = pd.to_datetime(
        dfs["consumption_sample"]["fecha"]
    )

    return dfs


def plot_economic_summary(df_economics):
    """Visualiza resumen económico por escuela"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Análisis Económico de Sistemas Híbridos por Escuela",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Inversión vs Ahorro Anual
    ax1 = axes[0, 0]
    schools = (
        df_economics["escuela"]
        .str.replace("Escuela ", "")
        .str.replace("Colegio ", "")
        .str.replace("Politécnica de ", "")
    )
    x = np.arange(len(schools))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        df_economics["inversion_total_usd"] / 10,
        width,
        label="Inversión Total",
        color="#e74c3c",
    )
    bars2 = ax1.bar(
        x + width / 2,
        df_economics["ahorro_anual_usd"] / 5,
        width,
        label="Ahorro Anual",
        color="#27ae60",
    )

    ax1.set_xlabel("Institución Educativa")
    ax1.set_ylabel("USD (Inversión en miles)")
    ax1.set_title("Inversión vs Ahorro Anual")
    ax1.set_xticks(x)
    ax1.set_xticklabels(schools, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${height:.1f}k",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Período de Retorno
    ax2 = axes[0, 1]
    colors = [
        "#27ae60" if x < 10 else "#f39c12" if x < 15 else "#e74c3c"
        for x in (df_economics["periodo_retorno_años"]) / 2
    ]
    bars = ax2.bar(schools, df_economics["periodo_retorno_años"], color=colors)
    ax2.set_xlabel("Institución Educativa")
    ax2.set_ylabel("Años")
    ax2.set_title("Período de Retorno de la Inversión")
    ax2.set_xticklabels(schools, rotation=45, ha="right")
    ax2.axhline(y=5, color="orange", linestyle="--", alpha=0.7, label="10 años")
    ax2.grid(True, alpha=0.3)

    # Añadir valores
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. CO2 Evitado Anual
    ax3 = axes[1, 0]
    co2_data = df_economics["co2_evitado_anual_kg"] / 1000  # Convertir a toneladas
    bars = ax3.bar(schools, co2_data, color="#3498db")
    ax3.set_xlabel("Institución Educativa")
    ax3.set_ylabel("Toneladas CO2/año")
    ax3.set_title("CO2 Evitado Anualmente")
    ax3.set_xticklabels(schools, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3)

    # Añadir valores
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}t",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Autosuficiencia Energética
    ax4 = axes[1, 1]
    autosuf = df_economics["autosuficiencia_promedio_%"]
    colors = [
        "#e74c3c" if x < 50 else "#f39c12" if x < 70 else "#27ae60" for x in autosuf
    ]
    bars = ax4.bar(schools, autosuf, color=colors)
    ax4.set_xlabel("Institución Educativa")
    ax4.set_ylabel("Porcentaje (%)")
    ax4.set_title("Autosuficiencia Energética Promedio")
    ax4.set_xticklabels(schools, rotation=45, ha="right")
    ax4.axhline(y=70, color="green", linestyle="--", alpha=0.7, label="Meta 70%")
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)

    # Añadir valores
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_generation_patterns(df_daily):
    """Visualiza patrones de generación solar y eólica"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3)

    fig.suptitle(
        "Patrones de Generación de Energía Renovable", fontsize=16, fontweight="bold"
    )

    # Seleccionar una escuela para análisis detallado
    school = "Colegio Condorazo"  # La más grande
    school_data = df_daily[df_daily["escuela"] == school].copy()

    # 1. Serie temporal de generación
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(
        school_data["fecha"].to_numpy(),
        school_data["generacion_solar_kwh"].to_numpy(),
        label="Solar",
        color="#f39c12",
        alpha=0.7,
        linewidth=1,
    )
    ax1.plot(
        school_data["fecha"].to_numpy(),
        school_data["generacion_eolica_kwh"].to_numpy(),
        label="Eólica",
        color="#3498db",
        alpha=0.7,
        linewidth=1,
    )
    ax1.fill_between(
        school_data["fecha"].to_numpy(),
        school_data["generacion_solar_kwh"].to_numpy(),
        alpha=0.3,
        color="#f39c12",
    )
    ax1.fill_between(
        school_data["fecha"].to_numpy(),
        school_data["generacion_eolica_kwh"].to_numpy(),
        alpha=0.3,
        color="#3498db",
    )
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Generación (kWh/día)")
    ax1.set_title(f"Serie Temporal de Generación Diaria - {school}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Patrón estacional - Solar
    ax2 = plt.subplot(gs[1, 0])
    monthly_solar = (
        df_daily.groupby([df_daily["fecha"].dt.month, "escuela"])[
            "generacion_solar_kwh"
        ]
        .mean()
        .reset_index()
    )
    for escuela in df_daily["escuela"].unique():
        school_monthly = monthly_solar[monthly_solar["escuela"] == escuela]
        ax2.plot(
            school_monthly["fecha"].to_numpy(),
            school_monthly["generacion_solar_kwh"].to_numpy(),
            marker="o",
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax2.set_xlabel("Mes")
    ax2.set_ylabel("Generación Solar Promedio (kWh/día)")
    ax2.set_title("Patrón Estacional - Generación Solar")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(
        [
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
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="best")

    # 3. Patrón estacional - Eólica
    ax3 = plt.subplot(gs[1, 1])
    monthly_wind = (
        df_daily.groupby([df_daily["fecha"].dt.month, "escuela"])[
            "generacion_eolica_kwh"
        ]
        .mean()
        .reset_index()
    )
    for escuela in df_daily["escuela"].unique():
        school_monthly = monthly_wind[monthly_wind["escuela"] == escuela]
        ax3.plot(
            school_monthly["fecha"].to_numpy(),
            school_monthly["generacion_eolica_kwh"].to_numpy(),
            marker="o",
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax3.set_xlabel("Mes")
    ax3.set_ylabel("Generación Eólica Promedio (kWh/día)")
    ax3.set_title("Patrón Estacional - Generación Eólica")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(
        [
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
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc="best")

    # 4. Correlación Solar vs Eólica
    ax4 = plt.subplot(gs[2, 0])
    for escuela in df_daily["escuela"].unique():
        school_data = df_daily[df_daily["escuela"] == escuela]
        ax4.scatter(
            school_data["generacion_solar_kwh"],
            school_data["generacion_eolica_kwh"],
            alpha=0.5,
            s=10,
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax4.set_xlabel("Generación Solar (kWh/día)")
    ax4.set_ylabel("Generación Eólica (kWh/día)")
    ax4.set_title("Correlación entre Generación Solar y Eólica")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # 5. Complementariedad horaria (muestra de un día típico)
    ax5 = plt.subplot(gs[2, 1])
    # Simular un patrón diario típico
    hours = np.arange(24)
    solar_pattern = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0.1,
            0.5,
            1.5,
            3.0,
            4.5,
            5.5,
            6.0,
            5.8,
            5.2,
            4.0,
            2.5,
            1.0,
            0.2,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    wind_pattern = np.array(
        [
            2.0,
            1.8,
            1.6,
            1.5,
            1.4,
            1.3,
            1.2,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.2,
            4.5,
            4.3,
            4.0,
            3.5,
            3.0,
            2.8,
            2.5,
            2.3,
            2.2,
            2.1,
        ]
    )

    ax5.fill_between(hours, solar_pattern, alpha=0.5, color="#f39c12", label="Solar")
    ax5.fill_between(hours, wind_pattern, alpha=0.5, color="#3498db", label="Eólica")
    ax5.plot(hours, solar_pattern + wind_pattern, "k-", linewidth=2, label="Total")
    ax5.set_xlabel("Hora del día")
    ax5.set_ylabel("Generación (kW)")
    ax5.set_title("Complementariedad Horaria Solar-Eólica (Día Típico)")
    ax5.set_xticks(range(0, 24, 3))
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    plt.tight_layout()
    return fig


def plot_consumption_analysis(df_consumption):
    """Analiza patrones de consumo energético"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Análisis de Patrones de Consumo Energético", fontsize=16, fontweight="bold"
    )

    # 1. Perfil de consumo diario promedio por escuela
    ax1 = axes[0, 0]
    hourly_avg = (
        df_consumption.groupby(["hora", "escuela"])["consumo_kwh"].mean().reset_index()
    )
    for escuela in df_consumption["escuela"].unique():
        school_hourly = hourly_avg[hourly_avg["escuela"] == escuela]
        ax1.plot(
            school_hourly["hora"].to_numpy(),
            school_hourly["consumo_kwh"].to_numpy(),
            marker="o",
            markersize=4,
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax1.set_xlabel("Hora del día")
    ax1.set_ylabel("Consumo Promedio (kWh)")
    ax1.set_title("Perfil de Consumo Diario por Institución")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))

    # 2. Comparación días laborales vs fines de semana
    ax2 = axes[0, 1]
    weekday_avg = (
        df_consumption[df_consumption["tipo_dia"] == "laboral"]
        .groupby("hora")["consumo_kwh"]
        .mean()
    )
    weekend_avg = (
        df_consumption[df_consumption["tipo_dia"] == "fin_semana"]
        .groupby("hora")["consumo_kwh"]
        .mean()
    )

    ax2.fill_between(
        weekday_avg.index.values,
        weekday_avg.values,
        alpha=0.5,
        color="#e74c3c",
        label="Días Laborales",
    )
    ax2.fill_between(
        weekend_avg.index.values,
        weekend_avg.values / 5,
        alpha=0.5,
        color="#3498db",
        label="Fines de Semana",
    )
    ax2.set_xlabel("Hora del día")
    ax2.set_ylabel("Consumo Promedio (kWh)")
    ax2.set_title("Consumo: Días Laborales vs Fines de Semana")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))

    # 3. Consumo por período (clases vs vacaciones)
    ax3 = axes[1, 0]
    period_consumption = (
        df_consumption.groupby(["periodo", "escuela"])["consumo_kwh"]
        .sum()
        .reset_index()
    )
    schools = period_consumption["escuela"].unique()
    x = np.arange(len(schools))
    width = 0.35

    clases_data = period_consumption[
        period_consumption["periodo"] == "vacaciones"
    ].set_index("escuela")["consumo_kwh"]
    vacaciones_data = period_consumption[
        period_consumption["periodo"] == "clases"
    ].set_index("escuela")["consumo_kwh"]

    print("vacaciones: ", vacaciones_data)

    bars1 = ax3.bar(
        x - width / 2,
        [clases_data.get(s, 0) for s in schools],
        width,
        label="Período Clases",
        color="#27ae60",
    )
    bars2 = ax3.bar(
        x + width / 2,
        [vacaciones_data.get(s, 0) for s in schools],
        width,
        label="Período Vacaciones",
        color="#f39c12",
    )

    ax3.set_xlabel("Institución Educativa")
    ax3.set_ylabel("Consumo Total (kWh)")
    ax3.set_title("Consumo Total por Período")
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [
            s.replace("Escuela ", "")
            .replace("Colegio ", "")
            .replace("Politécnica de ", "")
            for s in schools
        ],
        rotation=45,
        ha="right",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap de consumo (hora vs día de la semana)
    ax4 = axes[1, 1]
    # Crear matriz de consumo promedio
    df_consumption["dia_semana"] = pd.to_datetime(df_consumption["fecha"]).dt.dayofweek
    heatmap_data = (
        df_consumption.groupby(["dia_semana", "hora"])["consumo_kwh"].mean().unstack()
    )

    im = ax4.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    ax4.set_xlabel("Hora del día")
    ax4.set_ylabel("Día de la semana")
    ax4.set_title("Mapa de Calor - Consumo Promedio")
    ax4.set_xticks(range(0, 24, 2))
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])

    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("Consumo (kWh)")

    plt.tight_layout()
    return fig


def plot_balance_analysis(df_daily, df_monthly):
    """Analiza el balance energético del sistema"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Análisis del Balance Energético", fontsize=16, fontweight="bold")

    # 1. Balance diario para una escuela
    ax1 = axes[0, 0]
    school = "Colegio Condorazo"
    school_data = df_daily[df_daily["escuela"] == school].copy()
    school_data = school_data.sort_values("fecha")

    ax1.fill_between(
        school_data["fecha"].values,
        0,
        school_data["excedente_kwh"].values,
        alpha=0.5,
        color="#27ae60",
        label="Excedente",
    )
    ax1.fill_between(
        school_data["fecha"].values,
        0,
        -school_data["deficit_kwh"].values,
        alpha=0.5,
        color="#e74c3c",
        label="Déficit",
    )
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Energía (kWh)")
    ax1.set_title(f"Balance Energético Diario - {school}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Autosuficiencia mensual por escuela
    ax2 = axes[0, 1]
    for escuela in df_monthly["escuela"].unique():
        school_monthly = df_monthly[df_monthly["escuela"] == escuela].sort_values(
            "año_mes"
        )
        ax2.plot(
            school_monthly["año_mes"].values,
            school_monthly["autosuficiencia_%"].values,
            marker="o",
            markersize=4,
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax2.axhline(y=70, color="green", linestyle="--", alpha=0.7, label="Meta 70%")
    ax2.set_xlabel("Mes")
    ax2.set_ylabel("Autosuficiencia (%)")
    ax2.set_title("Evolución de la Autosuficiencia Mensual")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Comparación generación vs consumo mensual
    ax3 = axes[1, 0]
    monthly_totals = (
        df_monthly.groupby("año_mes")
        .agg({"generacion_total_kwh": "sum", "consumo_kwh": "sum"})
        .reset_index()
    )

    width = 15  # días
    ax3.bar(
        monthly_totals["año_mes"] - pd.Timedelta(days=width / 2),
        monthly_totals["generacion_total_kwh"],
        width=width,
        label="Generación Total",
        color="#3498db",
        alpha=0.7,
    )
    ax3.bar(
        monthly_totals["año_mes"] + pd.Timedelta(days=width / 2),
        monthly_totals["consumo_kwh"],
        width=width,
        label="Consumo Total",
        color="#e74c3c",
        alpha=0.7,
    )
    ax3.set_xlabel("Mes")
    ax3.set_ylabel("Energía (kWh)")
    ax3.set_title("Generación vs Consumo Mensual Total")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 4. Distribución de excedentes y déficits
    ax4 = axes[1, 1]
    excedente_total = df_daily.groupby("escuela")["excedente_kwh"].sum()
    deficit_total = df_daily.groupby("escuela")["deficit_kwh"].sum()

    schools = excedente_total.index
    x = np.arange(len(schools))
    width = 0.35

    bars1 = ax4.bar(
        x - width / 2,
        excedente_total.values,
        width,
        label="Excedente Total",
        color="#27ae60",
    )
    bars2 = ax4.bar(
        x + width / 2,
        deficit_total.values,
        width,
        label="Déficit Total",
        color="#e74c3c",
    )

    ax4.set_xlabel("Institución Educativa")
    ax4.set_ylabel("Energía Total (kWh)")
    ax4.set_title("Excedentes vs Déficits Totales por Institución")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [
            s.replace("Escuela ", "")
            .replace("Colegio ", "")
            .replace("Politécnica de ", "")
            for s in schools
        ],
        rotation=45,
        ha="right",
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_climate_correlation(df_climate, df_daily):
    """Analiza correlación entre clima y generación"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Correlación entre Variables Climáticas y Generación",
        fontsize=16,
        fontweight="bold",
    )

    # Preparar datos agregados diarios
    climate_daily = (
        df_climate.groupby(["fecha", "escuela"])
        .agg(
            {
                "radiacion_solar_kwh_m2": "sum",
                "velocidad_viento_m_s": "mean",
                "temperatura_c": "mean",
                "humedad_relativa_%": "mean",
            }
        )
        .reset_index()
    )

    # Unir con datos de generación
    merged_data = pd.merge(
        climate_daily,
        df_daily[["fecha", "escuela", "generacion_solar_kwh", "generacion_eolica_kwh"]],
        on=["fecha", "escuela"],
    )

    # 1. Radiación vs Generación Solar
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(merged_data["escuela"].unique())))
    for i, escuela in enumerate(merged_data["escuela"].unique()):
        school_data = merged_data[merged_data["escuela"] == escuela]
        ax1.scatter(
            school_data["radiacion_solar_kwh_m2"].to_numpy(),  # Convertir a numpy
            school_data["generacion_solar_kwh"].to_numpy(),  # Convertir a numpy
            alpha=0.5,
            s=20,
            color=colors[i],
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax1.set_xlabel("Radiación Solar (kWh/m²/día)")
    ax1.set_ylabel("Generación Solar (kWh/día)")
    ax1.set_title("Correlación: Radiación vs Generación Solar")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Velocidad del viento vs Generación Eólica
    ax2 = axes[0, 1]
    for i, escuela in enumerate(merged_data["escuela"].unique()):
        school_data = merged_data[merged_data["escuela"] == escuela]
        ax2.scatter(
            school_data["velocidad_viento_m_s"].to_numpy(),  # Convertir a numpy
            school_data["generacion_eolica_kwh"].to_numpy(),  # Convertir a numpy
            alpha=0.5,
            s=20,
            color=colors[i],
            label=escuela.replace("Escuela ", "").replace("Colegio ", ""),
        )
    ax2.set_xlabel("Velocidad del Viento (m/s)")
    ax2.set_ylabel("Generación Eólica (kWh/día)")
    ax2.set_title("Correlación: Viento vs Generación Eólica")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # 3. Efecto de la temperatura en generación solar
    ax3 = axes[1, 0]
    # Agrupar por rangos de temperatura
    temp_bins = pd.cut(merged_data["temperatura_c"], bins=10)
    temp_effect = (
        merged_data.groupby(temp_bins)
        .agg({"generacion_solar_kwh": "mean", "radiacion_solar_kwh_m2": "mean"})
        .reset_index()
    )
    temp_effect["eficiencia_relativa"] = (
        temp_effect["generacion_solar_kwh"] / temp_effect["radiacion_solar_kwh_m2"]
    )

    temp_centers = (
        temp_effect["temperatura_c"].apply(lambda x: x.mid).to_numpy()
    )  # Convertir a numpy
    ax3.plot(
        temp_centers,
        temp_effect["eficiencia_relativa"].to_numpy(),  # Convertir a numpy
        "o-",
        color="#e74c3c",
        markersize=8,
        linewidth=2,
    )
    ax3.set_xlabel("Temperatura (°C)")
    ax3.set_ylabel("Eficiencia Relativa")
    ax3.set_title("Efecto de la Temperatura en la Eficiencia Solar")
    ax3.grid(True, alpha=0.3)

    # 4. Rosa de vientos con generación
    ax4 = axes[1, 1]
    wind_speeds = merged_data["velocidad_viento_m_s"].to_numpy()  # Convertir a numpy
    wind_generation = merged_data[
        "generacion_eolica_kwh"
    ].to_numpy()  # Convertir a numpy

    ax4.scatter(wind_speeds, wind_generation, alpha=0.3, s=10, color="#3498db")

    # Ajustar curva de potencia
    from scipy.optimize import curve_fit

    def power_curve(v, a, b, c):
        return np.where(v < 2.5, 0, np.where(v > 25, 0, a * (v - 2.5) ** b + c))

    v_sorted = np.sort(wind_speeds)
    popt, _ = curve_fit(power_curve, wind_speeds, wind_generation, p0=[1, 3, 0])
    ax4.plot(
        v_sorted,
        power_curve(v_sorted, *popt),
        "r-",
        linewidth=2,
        label="Curva de potencia",
    )

    ax4.set_xlabel("Velocidad del Viento (m/s)")
    ax4.set_ylabel("Generación Eólica (kWh)")
    ax4.set_title("Curva de Potencia Eólica")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_systems_comparison(df_systems):

    global school_params
    """Compara las especificaciones de los sistemas instalados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Comparación de Sistemas Instalados", fontsize=16, fontweight="bold")

    schools = (
        df_systems["escuela"]
        .str.replace("Escuela ", "")
        .str.replace("Colegio ", "")
        .str.replace("Politécnica de ", "")
    )

    # 1. Capacidades instaladas
    ax1 = axes[0, 0]
    x = np.arange(len(schools))
    width = 0.25

    bars1 = ax1.bar(
        x - width,
        df_systems["capacidad_solar_instalada_kwp"],
        width,
        label="Solar (kWp)",
        color="#f39c12",
    )
    bars2 = ax1.bar(
        x,
        df_systems["capacidad_eolica_instalada_kw"],
        width,
        label="Eólica (kW)",
        color="#3498db",
    )
    """
    bars3 = ax1.bar(
        x + width,
        df_systems["capacidad_bateria_kwh"],
        width,
        label="Batería (kWh)",
        color="#27ae60",
    )
    """

    ax1.set_xlabel("Institución Educativa")
    ax1.set_ylabel("Capacidad")
    ax1.set_title("Capacidades Instaladas por Sistema")
    ax1.set_xticks(x)
    ax1.set_xticklabels(schools, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Relación capacidad vs estudiantes
    ax2 = axes[0, 1]
    students = [
        school_params[school]["estudiantes"] for school in df_systems["escuela"]
    ]

    ax2.scatter(
        students,
        df_systems["capacidad_solar_instalada_kwp"],
        s=100,
        alpha=0.7,
        label="Solar",
        color="#f39c12",
    )
    ax2.scatter(
        students,
        df_systems["capacidad_eolica_instalada_kw"],
        s=100,
        alpha=0.7,
        label="Eólica",
        color="#3498db",
    )

    # Líneas de tendencia
    z1 = np.polyfit(students, df_systems["capacidad_solar_instalada_kwp"], 1)
    z2 = np.polyfit(students, df_systems["capacidad_eolica_instalada_kw"], 1)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)

    ax2.plot(students, p1(students), "--", color="#f39c12", alpha=0.8)
    ax2.plot(students, p2(students), "--", color="#3498db", alpha=0.8)

    ax2.set_xlabel("Número de Estudiantes")
    ax2.set_ylabel("Capacidad Instalada (kW)")
    ax2.set_title("Capacidad vs Número de Estudiantes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Área utilizada vs disponible
    ax3 = axes[1, 0]
    areas_disponibles = [
        school_params[school]["area_techo_disponible_m2"]
        for school in df_systems["escuela"]
    ]
    areas_utilizadas = df_systems["area_paneles_m2"]

    x = np.arange(len(schools))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        areas_disponibles,
        width,
        label="Área Disponible",
        color="#95a5a6",
    )
    bars2 = ax3.bar(
        x + width / 2, areas_utilizadas, width, label="Área Utilizada", color="#f39c12"
    )

    ax3.set_xlabel("Institución Educativa")
    ax3.set_ylabel("Área (m²)")
    ax3.set_title("Utilización del Área Disponible para Paneles Solares")
    ax3.set_xticks(x)
    ax3.set_xticklabels(schools, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Eficiencias y vida útil
    ax4 = axes[1, 1]

    # Crear gráfico de radar simplificado
    categories = [
        "Efic. Paneles\n(%)",
        "Efic. Inversor\n(%)",
        "Efic. Baterías\n(%)",
        "Vida Paneles\n(años)",
        "Vida Baterías\n(años/5)",
    ]

    # Normalizar valores para comparación
    values_norm = np.array(
        [
            [17.5, 95, 90, 25, 20 / 5],  # Valores típicos del sistema
        ]
    )

    x_radar = np.arange(len(categories))
    bars = ax4.bar(
        x_radar,
        values_norm[0],
        color=["#f39c12", "#e74c3c", "#27ae60", "#3498db", "#9b59b6"],
    )

    ax4.set_xticks(x_radar)
    ax4.set_xticklabels(categories)
    ax4.set_ylabel("Valor")
    ax4.set_title("Características Técnicas del Sistema")
    ax4.grid(True, alpha=0.3, axis="y")

    # Añadir valores en las barras
    for bar, value in zip(bars, values_norm[0]):
        height = bar.get_height()
        if categories[bars.index(bar)] == "Vida Baterías\n(años/5)":
            label = f"{value*5:.0f}"
        else:
            label = f"{value:.1f}"
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0, height, label, ha="center", va="bottom"
        )

    plt.tight_layout()
    return fig


def generate_summary_report(dfs):
    """Genera un reporte resumido con estadísticas clave"""
    print("\n" + "=" * 80)
    print("REPORTE EJECUTIVO - SISTEMAS HÍBRIDOS DE ENERGÍA RENOVABLE")
    print("Escuelas Rurales de Chimborazo, Ecuador")
    print("=" * 80)

    economics = dfs["economics"]
    systems = dfs["systems"]

    print("\n1. RESUMEN ECONÓMICO GLOBAL")
    print("-" * 40)
    print(
        f"Inversión total en todas las escuelas: ${economics['inversion_total_usd'].sum():,.2f}"
    )
    print(f"Ahorro anual total: ${economics['ahorro_anual_usd'].sum():,.2f}")
    print(
        f"Período de retorno promedio: {economics['periodo_retorno_años'].mean():.1f} años"
    )
    print(
        f"CO2 evitado anualmente: {economics['co2_evitado_anual_kg'].sum()/1000:.1f} toneladas"
    )

    print("\n2. CAPACIDAD INSTALADA TOTAL")
    print("-" * 40)
    print(
        f"Capacidad solar total: {systems['capacidad_solar_instalada_kwp'].sum():.1f} kWp"
    )
    print(
        f"Capacidad eólica total: {systems['capacidad_eolica_instalada_kw'].sum():.1f} kW"
    )
    print(
        f"Capacidad de almacenamiento: {systems['capacidad_bateria_kwh'].sum():.1f} kWh"
    )

    print("\n3. RENDIMIENTO POR ESCUELA")
    print("-" * 40)
    for idx, row in economics.iterrows():
        print(f"\n{row['escuela']}:")
        print(f"  - Autosuficiencia: {row['autosuficiencia_promedio_%']:.1f}%")
        print(f"  - ROI: {row['periodo_retorno_años']:.1f} años")
        print(f"  - Ahorro anual: ${row['ahorro_anual_usd']:,.2f}")

    print("\n4. RECOMENDACIONES")
    print("-" * 40)
    print("• Las escuelas con autosuficiencia < 70% deberían considerar:")
    print("  - Aumentar capacidad de almacenamiento")
    print("  - Optimizar horarios de consumo")
    print("  - Implementar medidas de eficiencia energética")
    print("\n• Para mejorar el ROI:")
    print("  - Negociar mejores precios en compras grupales")
    print("  - Solicitar subsidios gubernamentales")
    print("  - Implementar sistemas de monitoreo en tiempo real")

    print("\n" + "=" * 80)


def main():
    """Función principal para ejecutar todas las visualizaciones"""
    print("Cargando datos...")
    dfs = load_data()

    if dfs is None:
        return

    print("Generando visualizaciones...")

    # Crear directorio para gráficos
    if not os.path.exists("graficos_energia_renovable"):
        os.makedirs("graficos_energia_renovable")

    # 1. Análisis económico
    fig1 = plot_economic_summary(dfs["economics"])
    fig1.savefig(
        "graficos_energia_renovable/01_analisis_economico.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Análisis económico generado")

    # 2. Patrones de generación
    fig2 = plot_generation_patterns(dfs["daily"])
    fig2.savefig(
        "graficos_energia_renovable/02_patrones_generacion.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Patrones de generación generados")

    # 3. Análisis de consumo
    fig3 = plot_consumption_analysis(dfs["consumption_sample"])
    fig3.savefig(
        "graficos_energia_renovable/03_analisis_consumo.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Análisis de consumo generado")

    # 4. Balance energético
    fig4 = plot_balance_analysis(dfs["daily"], dfs["monthly"])
    fig4.savefig(
        "graficos_energia_renovable/04_balance_energetico.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Balance energético generado")

    # 5. Correlación climática
    fig5 = plot_climate_correlation(dfs["climate_sample"], dfs["daily"])
    fig5.savefig(
        "graficos_energia_renovable/05_correlacion_climatica.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Correlación climática generada")

    # 6. Comparación de sistemas
    fig6 = plot_systems_comparison(dfs["systems"])
    fig6.savefig(
        "graficos_energia_renovable/06_comparacion_sistemas.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Comparación de sistemas generada")

    # Generar reporte resumido
    generate_summary_report(dfs)

    print(
        "\n✅ Todas las visualizaciones han sido generadas en la carpeta 'graficos_energia_renovable'"
    )

    # Mostrar gráficos si se ejecuta en entorno interactivo
    plt.show()


if __name__ == "__main__":
    main()
