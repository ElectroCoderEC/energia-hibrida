"""
Dashboard de Visualización y Análisis de Datos
Sistema de Energías Renovables - Instituciones Educativas Chimborazo

Genera gráficos comprensivos para analizar:
- Patrones climáticos y estacionales
- Generación vs consumo energético
- Análisis de correlaciones
- Performance del sistema
- Indicadores económicos y ambientales
- Comparativas entre escuelas

Autor: Sistema de Dimensionamiento IA
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Configuración de estilo
plt.style.use("seaborn-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class RenewableEnergyDashboard:
    """Dashboard principal para visualización de datos de energías renovables"""

    def __init__(self, data_path="datos_sintéticos_chimborazo/"):
        self.data_path = data_path
        self.load_all_data()
        self.setup_colors()

    def load_all_data(self):
        """Cargar todos los datasets"""
        try:
            # Cargar el dataset completo de ML
            self.df_ml = pd.read_csv(f"{self.data_path}clima_horario.csv")
            self.df_ml["fecha"] = pd.to_datetime(self.df_ml["fecha"])

            print(f"✅ Datos cargados: {len(self.df_ml):,} registros")
            print(
                f"📅 Período: {self.df_ml['fecha'].min()} - {self.df_ml['fecha'].max()}"
            )
            print(f"🏫 Escuelas: {self.df_ml['escuela'].nunique()}")

        except FileNotFoundError:
            print("❌ Error: No se encontraron los archivos de datos.")
            print("   Ejecutar primero: python generar_datos_sintéticos_chimborazo.py")
            return

    def setup_colors(self):
        """Configurar paleta de colores por escuela"""
        schools = self.df_ml["escuela"].unique()
        self.school_colors = dict(zip(schools, sns.color_palette("Set2", len(schools))))

        # Colores temáticos
        self.theme_colors = {
            "solar": "#FFA500",
            "wind": "#4169E1",
            "battery": "#32CD32",
            "consumption": "#DC143C",
            "savings": "#228B22",
            "investment": "#8B0000",
        }

    def plot_climate_patterns(self):
        """Análisis de patrones climáticos"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Patrones Climáticos - Provincia de Chimborazo",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Radiación solar por mes y escuela
        monthly_radiation = (
            self.df_ml.groupby(["mes", "escuela"])["radiacion_solar_kwh_m2"]
            .mean()
            .reset_index()
        )
        pivot_radiation = monthly_radiation.pivot(
            index="mes", columns="escuela", values="radiacion_solar_kwh_m2"
        )

        sns.heatmap(
            pivot_radiation.T,
            annot=True,
            cmap="YlOrRd",
            ax=axes[0, 0],
            cbar_kws={"label": "kWh/m²"},
        )
        axes[0, 0].set_title("Radiación Solar Promedio por Mes")
        axes[0, 0].set_xlabel("Mes")
        axes[0, 0].set_ylabel("Escuela")

        # 2. Velocidad de viento por altitud
        wind_altitude = (
            self.df_ml.groupby(["escuela", "altitude"])["velocidad_viento_m_s"]
            .mean()
            .reset_index()
        )
        axes[0, 1].scatter(
            wind_altitude["altitude"],
            wind_altitude["velocidad_viento_m_s"],
            c=[self.school_colors[school] for school in wind_altitude["escuela"]],
            s=100,
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Altitud (m)")
        axes[0, 1].set_ylabel("Velocidad Viento (m/s)")
        axes[0, 1].set_title("Viento vs Altitud")

        # Línea de tendencia
        z = np.polyfit(
            wind_altitude["altitude"], wind_altitude["velocidad_viento_m_s"], 1
        )
        p = np.poly1d(z)
        axes[0, 1].plot(
            wind_altitude["altitude"], p(wind_altitude["altitude"]), "r--", alpha=0.8
        )

        # 3. Temperatura por mes
        monthly_temp = (
            self.df_ml.groupby(["mes", "escuela"])["temperatura_c"].mean().reset_index()
        )
        for school in self.df_ml["escuela"].unique():
            school_data = monthly_temp[monthly_temp["escuela"] == school]
            axes[0, 2].plot(
                school_data["mes"],
                school_data["temperatura_c"],
                marker="o",
                label=school.split()[-1],
                color=self.school_colors[school],
            )
        axes[0, 2].set_xlabel("Mes")
        axes[0, 2].set_ylabel("Temperatura (°C)")
        axes[0, 2].set_title("Temperatura Promedio Mensual")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Distribución de radiación por escuela
        schools_sample = self.df_ml.groupby("escuela").sample(
            n=min(1000, len(self.df_ml) // 5)
        )
        sns.boxplot(
            data=schools_sample, x="escuela", y="radiacion_solar_kwh_m2", ax=axes[1, 0]
        )
        axes[1, 0].set_title("Distribución Radiación Solar")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 5. Correlación variables climáticas
        climate_vars = [
            "radiacion_solar_kwh_m2",
            "velocidad_viento_m_s",
            "temperatura_c",
            "humedad_relativa",
        ]
        correlation_matrix = self.df_ml[climate_vars].corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1, 1]
        )
        axes[1, 1].set_title("Correlación Variables Climáticas")

        # 6. Patrones estacionales
        seasonal_data = (
            self.df_ml.groupby(["mes"])
            .agg(
                {
                    "radiacion_solar_kwh_m2": "mean",
                    "velocidad_viento_m_s": "mean",
                    "temperatura_c": "mean",
                }
            )
            .reset_index()
        )

        ax2 = axes[1, 2].twinx()
        line1 = axes[1, 2].plot(
            seasonal_data["mes"],
            seasonal_data["radiacion_solar_kwh_m2"],
            "o-",
            color=self.theme_colors["solar"],
            label="Radiación Solar",
        )
        line2 = axes[1, 2].plot(
            seasonal_data["mes"],
            seasonal_data["velocidad_viento_m_s"],
            "s-",
            color=self.theme_colors["wind"],
            label="Viento",
        )
        line3 = ax2.plot(
            seasonal_data["mes"],
            seasonal_data["temperatura_c"],
            "^-",
            color="red",
            label="Temperatura",
        )

        axes[1, 2].set_xlabel("Mes")
        axes[1, 2].set_ylabel("Radiación (kWh/m²) / Viento (m/s)")
        ax2.set_ylabel("Temperatura (°C)")
        axes[1, 2].set_title("Patrones Estacionales")

        # Leyenda combinada
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        axes[1, 2].legend(lines, labels, loc="upper left")

        plt.tight_layout()
        plt.savefig(f"{self.data_path}clima_patterns.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_energy_analysis(self):
        """Análisis de generación y consumo energético"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Análisis Energético - Generación vs Consumo",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Generación vs Consumo diario por escuela
        daily_data = (
            self.df_ml.groupby(["fecha", "escuela"])
            .agg({"generacion_total_kwh": "sum", "consumo_kwh": "sum"})
            .reset_index()
        )

        for school in self.df_ml["escuela"].unique():
            school_data = daily_data[daily_data["escuela"] == school].head(
                365
            )  # Un año
            axes[0, 0].plot(
                school_data["fecha"],
                school_data["generacion_total_kwh"],
                alpha=0.7,
                label=f"{school.split()[-1]} - Gen",
            )
            axes[0, 0].plot(
                school_data["fecha"],
                school_data["consumo_kwh"],
                linestyle="--",
                alpha=0.7,
                label=f"{school.split()[-1]} - Cons",
            )

        axes[0, 0].set_title("Generación vs Consumo Diario (2021)")
        axes[0, 0].set_ylabel("Energía (kWh)")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Balance energético por hora del día
        hourly_balance = (
            self.df_ml.groupby(["hora", "escuela"])
            .agg({"generacion_total_kwh": "mean", "consumo_kwh": "mean"})
            .reset_index()
        )

        for school in self.df_ml["escuela"].unique():
            school_data = hourly_balance[hourly_balance["escuela"] == school]
            balance = school_data["generacion_total_kwh"] - school_data["consumo_kwh"]
            axes[0, 1].plot(
                school_data["hora"],
                balance,
                marker="o",
                label=school.split()[-1],
                color=self.school_colors[school],
            )

        axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
        axes[0, 1].set_title("Balance Energético por Hora")
        axes[0, 1].set_xlabel("Hora del Día")
        axes[0, 1].set_ylabel("Balance (kWh)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribución de autosuficiencia
        autosuf_data = (
            self.df_ml.groupby(["fecha", "escuela"])
            .agg({"generacion_total_kwh": "sum", "consumo_kwh": "sum"})
            .reset_index()
        )
        autosuf_data["autosuficiencia"] = np.minimum(
            100,
            (autosuf_data["generacion_total_kwh"] / autosuf_data["consumo_kwh"]) * 100,
        )

        sns.boxplot(data=autosuf_data, x="escuela", y="autosuficiencia", ax=axes[0, 2])
        axes[0, 2].set_title("Distribución Autosuficiencia (%)")
        axes[0, 2].tick_params(axis="x", rotation=45)
        axes[0, 2].set_ylabel("Autosuficiencia (%)")

        # 4. Factores de capacidad por tecnología
        capacity_factors = (
            self.df_ml.groupby("escuela")
            .agg({"factor_capacidad_solar": "mean", "factor_capacidad_eolica": "mean"})
            .reset_index()
        )

        x = np.arange(len(capacity_factors))
        width = 0.35

        bars1 = axes[1, 0].bar(
            x - width / 2,
            capacity_factors["factor_capacidad_solar"] * 100,
            width,
            label="Solar",
            color=self.theme_colors["solar"],
            alpha=0.8,
        )
        bars2 = axes[1, 0].bar(
            x + width / 2,
            capacity_factors["factor_capacidad_eolica"] * 100,
            width,
            label="Eólico",
            color=self.theme_colors["wind"],
            alpha=0.8,
        )

        axes[1, 0].set_xlabel("Escuela")
        axes[1, 0].set_ylabel("Factor de Capacidad (%)")
        axes[1, 0].set_title("Factores de Capacidad Promedio")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(
            [school.split()[-1] for school in capacity_factors["escuela"]], rotation=45
        )
        axes[1, 0].legend()

        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            axes[1, 0].annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 5. Consumo por hora de clases vs no clases
        consumo_clases = (
            self.df_ml.groupby(["hora", "es_hora_clases"])["consumo_kwh"]
            .mean()
            .reset_index()
        )

        for es_clase in [True, False]:
            data = consumo_clases[consumo_clases["es_hora_clases"] == es_clase]
            label = "Horas de Clase" if es_clase else "Fuera de Clase"
            axes[1, 1].plot(
                data["hora"], data["consumo_kwh"], marker="o", label=label, linewidth=2
            )

        axes[1, 1].set_title("Consumo: Horas de Clase vs Fuera de Clase")
        axes[1, 1].set_xlabel("Hora del Día")
        axes[1, 1].set_ylabel("Consumo Promedio (kWh)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Mapa de calor: Generación por mes y hora
        generation_heatmap = (
            self.df_ml.groupby(["mes", "hora"])["generacion_total_kwh"]
            .mean()
            .reset_index()
        )
        generation_pivot = generation_heatmap.pivot(
            index="hora", columns="mes", values="generacion_total_kwh"
        )

        sns.heatmap(
            generation_pivot, cmap="YlOrRd", ax=axes[1, 2], cbar_kws={"label": "kWh"}
        )
        axes[1, 2].set_title("Generación Total por Mes y Hora")
        axes[1, 2].set_xlabel("Mes")
        axes[1, 2].set_ylabel("Hora del Día")

        plt.tight_layout()
        plt.savefig(
            f"{self.data_path}energy_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def plot_correlation_analysis(self):
        """Análisis de correlaciones relevantes para el estudio"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Análisis de Correlaciones - Variables Clave",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Correlación entre variables técnicas
        technical_vars = [
            "capacidad_solar_kw",
            "capacidad_eolica_kw",
            "capacidad_bateria_kwh",
            "generacion_total_kwh",
            "consumo_kwh",
            "costo_total_usd",
            "students",
        ]

        # Tomar muestra para mejorar performance
        sample_data = self.df_ml.sample(n=min(10000, len(self.df_ml)))
        tech_corr = sample_data[technical_vars].corr()

        mask = np.triu(np.ones_like(tech_corr, dtype=bool))
        sns.heatmap(
            tech_corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=axes[0, 0],
            square=True,
            fmt=".2f",
        )
        axes[0, 0].set_title("Correlación Variables Técnicas")

        # 2. Scatter: Radiación vs Generación Solar
        sample_data_scatter = self.df_ml.sample(n=5000)
        scatter = axes[0, 1].scatter(
            sample_data_scatter["radiacion_solar_kwh_m2"],
            sample_data_scatter["generacion_solar_kwh"],
            c=[self.school_colors[school] for school in sample_data_scatter["escuela"]],
            alpha=0.6,
            s=20,
        )

        # Línea de tendencia
        z = np.polyfit(
            sample_data_scatter["radiacion_solar_kwh_m2"],
            sample_data_scatter["generacion_solar_kwh"],
            1,
        )
        p = np.poly1d(z)
        axes[0, 1].plot(
            sample_data_scatter["radiacion_solar_kwh_m2"],
            p(sample_data_scatter["radiacion_solar_kwh_m2"]),
            "r--",
            alpha=0.8,
        )

        axes[0, 1].set_xlabel("Radiación Solar (kWh/m²)")
        axes[0, 1].set_ylabel("Generación Solar (kWh)")
        axes[0, 1].set_title("Radiación vs Generación Solar")

        # Calcular R²
        correlation = np.corrcoef(
            sample_data_scatter["radiacion_solar_kwh_m2"],
            sample_data_scatter["generacion_solar_kwh"],
        )[0, 1]
        axes[0, 1].text(
            0.05,
            0.95,
            f"R² = {correlation**2:.3f}",
            transform=axes[0, 1].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 3. Viento vs Generación Eólica
        axes[1, 0].scatter(
            sample_data_scatter["velocidad_viento_m_s"],
            sample_data_scatter["generacion_eolica_kwh"],
            c=[self.school_colors[school] for school in sample_data_scatter["escuela"]],
            alpha=0.6,
            s=20,
        )

        axes[1, 0].set_xlabel("Velocidad Viento (m/s)")
        axes[1, 0].set_ylabel("Generación Eólica (kWh)")
        axes[1, 0].set_title("Viento vs Generación Eólica")

        # 4. Estudiantes vs Dimensionamiento del Sistema
        system_summary = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "students": "first",
                    "capacidad_solar_kw": "first",
                    "capacidad_eolica_kw": "first",
                    "capacidad_bateria_kwh": "first",
                    "costo_total_usd": "first",
                }
            )
            .reset_index()
        )

        system_summary["capacidad_total_kw"] = (
            system_summary["capacidad_solar_kw"] + system_summary["capacidad_eolica_kw"]
        )

        # Scatter con tamaño proporcional al costo
        scatter = axes[1, 1].scatter(
            system_summary["students"],
            system_summary["capacidad_total_kw"],
            s=system_summary["costo_total_usd"] / 100,  # Escalar tamaño
            c=[self.school_colors[school] for school in system_summary["escuela"]],
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        axes[1, 1].set_xlabel("Número de Estudiantes")
        axes[1, 1].set_ylabel("Capacidad Total Sistema (kW)")
        axes[1, 1].set_title("Estudiantes vs Dimensionamiento\n(Tamaño = Costo Total)")

        # Agregar etiquetas de escuelas
        for i, row in system_summary.iterrows():
            axes[1, 1].annotate(
                row["escuela"].split()[-1],
                (row["students"], row["capacidad_total_kw"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.data_path}correlation_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def plot_economic_analysis(self):
        """Análisis económico y de retorno de inversión"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            "Análisis Económico - Inversiones y Ahorros", fontsize=16, fontweight="bold"
        )

        # Datos por escuela
        economic_summary = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "costo_total_usd": "first",
                    "ahorro_diario_usd": "mean",
                    "roi_simple_años": "first",
                    "students": "first",
                    "generacion_total_kwh": "sum",
                    "co2_evitado_kg": "sum",
                }
            )
            .reset_index()
        )

        economic_summary["ahorro_anual_usd"] = (
            economic_summary["ahorro_diario_usd"] * 365
        )
        economic_summary["costo_por_estudiante"] = (
            economic_summary["costo_total_usd"] / economic_summary["students"]
        )

        # 1. Inversión vs ROI
        scatter = axes[0, 0].scatter(
            economic_summary["costo_total_usd"],
            economic_summary["roi_simple_años"],
            s=economic_summary["students"] * 2,  # Tamaño por estudiantes
            c=[self.school_colors[school] for school in economic_summary["escuela"]],
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        axes[0, 0].set_xlabel("Inversión Total (USD)")
        axes[0, 0].set_ylabel("ROI Simple (años)")
        axes[0, 0].set_title(
            "Inversión vs Retorno de Inversión\n(Tamaño = Estudiantes)"
        )

        # Agregar línea de ROI objetivo (ej: 10 años)
        axes[0, 0].axhline(
            y=10, color="red", linestyle="--", alpha=0.7, label="ROI Objetivo: 10 años"
        )
        axes[0, 0].legend()

        # Etiquetas
        for i, row in economic_summary.iterrows():
            axes[0, 0].annotate(
                row["escuela"].split()[-1],
                (row["costo_total_usd"], row["roi_simple_años"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # 2. Desglose de costos por escuela
        cost_breakdown = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "costo_solar_usd": "first",
                    "costo_eolico_usd": "first",
                    "costo_baterias_usd": "first",
                    "costo_instalacion_usd": "first",
                }
            )
            .reset_index()
        )

        schools_short = [school.split()[-1] for school in cost_breakdown["escuela"]]
        x = np.arange(len(schools_short))
        width = 0.6

        bottom1 = cost_breakdown["costo_solar_usd"]
        bottom2 = bottom1 + cost_breakdown["costo_eolico_usd"]
        bottom3 = bottom2 + cost_breakdown["costo_baterias_usd"]

        axes[0, 1].bar(
            x,
            cost_breakdown["costo_solar_usd"],
            width,
            label="Solar",
            color=self.theme_colors["solar"],
            alpha=0.8,
        )
        axes[0, 1].bar(
            x,
            cost_breakdown["costo_eolico_usd"],
            width,
            bottom=bottom1,
            label="Eólico",
            color=self.theme_colors["wind"],
            alpha=0.8,
        )
        axes[0, 1].bar(
            x,
            cost_breakdown["costo_baterias_usd"],
            width,
            bottom=bottom2,
            label="Baterías",
            color=self.theme_colors["battery"],
            alpha=0.8,
        )
        axes[0, 1].bar(
            x,
            cost_breakdown["costo_instalacion_usd"],
            width,
            bottom=bottom3,
            label="Instalación",
            color="gray",
            alpha=0.8,
        )

        axes[0, 1].set_xlabel("Escuela")
        axes[0, 1].set_ylabel("Costo (USD)")
        axes[0, 1].set_title("Desglose de Costos por Componente")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(schools_short, rotation=45)
        axes[0, 1].legend()

        # 3. Ahorro anual vs Inversión
        axes[1, 0].scatter(
            economic_summary["costo_total_usd"],
            economic_summary["ahorro_anual_usd"],
            s=100,
            c=[self.school_colors[school] for school in economic_summary["escuela"]],
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        # Línea de tendencia
        z = np.polyfit(
            economic_summary["costo_total_usd"], economic_summary["ahorro_anual_usd"], 1
        )
        p = np.poly1d(z)
        axes[1, 0].plot(
            economic_summary["costo_total_usd"],
            p(economic_summary["costo_total_usd"]),
            "r--",
            alpha=0.8,
        )

        axes[1, 0].set_xlabel("Inversión Total (USD)")
        axes[1, 0].set_ylabel("Ahorro Anual (USD)")
        axes[1, 0].set_title("Inversión vs Ahorro Anual")

        # 4. Impacto ambiental (CO2 evitado)
        co2_data = (
            self.df_ml.groupby(["escuela", "año"])
            .agg({"co2_evitado_kg": "sum"})
            .reset_index()
        )

        for school in co2_data["escuela"].unique():
            school_data = co2_data[co2_data["escuela"] == school]
            axes[1, 1].plot(
                school_data["año"],
                school_data["co2_evitado_kg"] / 1000,  # Convertir a toneladas
                marker="o",
                label=school.split()[-1],
                color=self.school_colors[school],
                linewidth=2,
            )

        axes[1, 1].set_xlabel("Año")
        axes[1, 1].set_ylabel("CO₂ Evitado (toneladas)")
        axes[1, 1].set_title("Impacto Ambiental - CO₂ Evitado por Año")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.data_path}economic_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def plot_comparative_performance(self):
        """Análisis comparativo de performance entre escuelas"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Análisis Comparativo de Performance entre Escuelas",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Radar chart - Performance multidimensional
        performance_metrics = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "factor_capacidad_solar": "mean",
                    "factor_capacidad_eolica": "mean",
                    "autosuficiencia_horaria_pct": "mean",
                    "ahorro_diario_usd": "mean",
                    "co2_evitado_kg": "mean",
                }
            )
            .reset_index()
        )

        # Normalizar métricas a escala 0-100
        performance_metrics["factor_capacidad_solar_norm"] = (
            performance_metrics["factor_capacidad_solar"] * 100
        )
        performance_metrics["factor_capacidad_eolica_norm"] = (
            performance_metrics["factor_capacidad_eolica"] * 100
        )
        performance_metrics["ahorro_norm"] = (
            performance_metrics["ahorro_diario_usd"]
            / performance_metrics["ahorro_diario_usd"].max()
        ) * 100
        performance_metrics["co2_norm"] = (
            performance_metrics["co2_evitado_kg"]
            / performance_metrics["co2_evitado_kg"].max()
        ) * 100

        # Crear gráfico de barras agrupadas en lugar de radar
        metrics = [
            "factor_capacidad_solar_norm",
            "factor_capacidad_eolica_norm",
            "autosuficiencia_horaria_pct",
            "ahorro_norm",
            "co2_norm",
        ]
        metric_labels = [
            "Cap. Solar (%)",
            "Cap. Eólica (%)",
            "Autosuf. (%)",
            "Ahorro (%)",
            "CO₂ (%)",
        ]

        x = np.arange(len(metrics))
        width = 0.15

        for i, school in enumerate(performance_metrics["escuela"]):
            values = [performance_metrics.iloc[i][metric] for metric in metrics]
            axes[0, 0].bar(
                x + i * width,
                values,
                width,
                label=school.split()[-1],
                color=self.school_colors[school],
                alpha=0.8,
            )

        axes[0, 0].set_xlabel("Métricas de Performance")
        axes[0, 0].set_ylabel("Valor Normalizado (%)")
        axes[0, 0].set_title("Performance Multidimensional por Escuela")
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(metric_labels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Eficiencia energética por estudiante
        efficiency_data = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "students": "first",
                    "generacion_total_kwh": "sum",
                    "consumo_kwh": "sum",
                    "costo_total_usd": "first",
                }
            )
            .reset_index()
        )

        efficiency_data["generacion_per_student"] = (
            efficiency_data["generacion_total_kwh"] / efficiency_data["students"]
        )
        efficiency_data["consumo_per_student"] = (
            efficiency_data["consumo_kwh"] / efficiency_data["students"]
        )
        efficiency_data["costo_per_student"] = (
            efficiency_data["costo_total_usd"] / efficiency_data["students"]
        )

        # Bubble chart
        scatter = axes[0, 1].scatter(
            efficiency_data["consumo_per_student"],
            efficiency_data["generacion_per_student"],
            s=efficiency_data["costo_per_student"] / 10,  # Tamaño proporcional al costo
            c=[self.school_colors[school] for school in efficiency_data["escuela"]],
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        # Línea de balance (generación = consumo)
        max_val = max(
            efficiency_data["consumo_per_student"].max(),
            efficiency_data["generacion_per_student"].max(),
        )
        axes[0, 1].plot(
            [0, max_val], [0, max_val], "r--", alpha=0.7, label="Balance Perfecto"
        )

        axes[0, 1].set_xlabel("Consumo per cápita (kWh/estudiante)")
        axes[0, 1].set_ylabel("Generación per cápita (kWh/estudiante)")
        axes[0, 1].set_title(
            "Eficiencia Energética per cápita\n(Tamaño = Costo/estudiante)"
        )
        axes[0, 1].legend()

        # Etiquetas
        for i, row in efficiency_data.iterrows():
            axes[0, 1].annotate(
                row["escuela"].split()[-1],
                (row["consumo_per_student"], row["generacion_per_student"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # 3. Variabilidad estacional por escuela
        seasonal_data = (
            self.df_ml.groupby(["escuela", "mes"])
            .agg({"generacion_total_kwh": "mean", "consumo_kwh": "mean"})
            .reset_index()
        )
        seasonal_data["balance"] = (
            seasonal_data["generacion_total_kwh"] - seasonal_data["consumo_kwh"]
        )

        # Heatmap del balance por mes y escuela
        balance_pivot = seasonal_data.pivot(
            index="escuela", columns="mes", values="balance"
        )
        sns.heatmap(
            balance_pivot,
            annot=True,
            cmap="RdYlGn",
            center=0,
            ax=axes[1, 0],
            cbar_kws={"label": "Balance (kWh)"},
            fmt=".1f",
        )
        axes[1, 0].set_title("Balance Energético Estacional")
        axes[1, 0].set_xlabel("Mes")
        axes[1, 0].set_ylabel("Escuela")

        # 4. Ranking general de escuelas
        # Calcular score compuesto
        ranking_data = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "autosuficiencia_horaria_pct": "mean",
                    "roi_simple_años": "first",
                    "factor_capacidad_solar": "mean",
                    "factor_capacidad_eolica": "mean",
                    "ahorro_diario_usd": "mean",
                    "students": "first",
                }
            )
            .reset_index()
        )

        # Normalizar y calcular score (mayor es mejor)
        ranking_data["autosuf_score"] = (
            ranking_data["autosuficiencia_horaria_pct"] / 100
        )
        ranking_data["roi_score"] = 1 / ranking_data["roi_simple_años"]  # Invertir ROI
        ranking_data["capacity_score"] = (
            ranking_data["factor_capacidad_solar"]
            + ranking_data["factor_capacidad_eolica"]
        ) / 2
        ranking_data["ahorro_score"] = (
            ranking_data["ahorro_diario_usd"] / ranking_data["ahorro_diario_usd"].max()
        )

        # Score compuesto (pesos iguales)
        ranking_data["score_total"] = (
            ranking_data["autosuf_score"]
            + ranking_data["roi_score"]
            + ranking_data["capacity_score"]
            + ranking_data["ahorro_score"]
        ) / 4

        ranking_data = ranking_data.sort_values("score_total", ascending=True)

        # Gráfico de barras horizontales
        y_pos = np.arange(len(ranking_data))
        bars = axes[1, 1].barh(
            y_pos,
            ranking_data["score_total"],
            color=[self.school_colors[school] for school in ranking_data["escuela"]],
            alpha=0.8,
        )

        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(
            [school.split()[-1] for school in ranking_data["escuela"]]
        )
        axes[1, 1].set_xlabel("Score Compuesto")
        axes[1, 1].set_title("Ranking General de Performance")
        axes[1, 1].grid(True, alpha=0.3, axis="x")

        # Agregar valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 1].text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.data_path}comparative_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def create_interactive_dashboard(self):
        """Crear dashboard interactivo con Plotly"""
        print("🚀 Creando dashboard interactivo...")

        # Preparar datos
        daily_data = (
            self.df_ml.groupby(["fecha", "escuela"])
            .agg(
                {
                    "generacion_total_kwh": "sum",
                    "consumo_kwh": "sum",
                    "co2_evitado_kg": "sum",
                    "ahorro_diario_usd": "sum",
                }
            )
            .reset_index()
        )

        # Dashboard con subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Generación vs Consumo por Escuela",
                "Autosuficiencia Energética",
                "Impacto Ambiental (CO₂)",
                "Análisis Económico",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Generación vs Consumo
        for school in daily_data["escuela"].unique():
            school_data = daily_data[daily_data["escuela"] == school]

            fig.add_trace(
                go.Scatter(
                    x=school_data["fecha"],
                    y=school_data["generacion_total_kwh"],
                    mode="lines",
                    name=f"{school.split()[-1]} - Gen",
                    line=dict(
                        color=px.colors.qualitative.Set2[
                            hash(school) % len(px.colors.qualitative.Set2)
                        ]
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=school_data["fecha"],
                    y=school_data["consumo_kwh"],
                    mode="lines",
                    name=f"{school.split()[-1]} - Cons",
                    line=dict(
                        dash="dash",
                        color=px.colors.qualitative.Set2[
                            hash(school) % len(px.colors.qualitative.Set2)
                        ],
                    ),
                ),
                row=1,
                col=1,
            )

        # 2. Autosuficiencia
        daily_data["autosuficiencia"] = np.minimum(
            100, (daily_data["generacion_total_kwh"] / daily_data["consumo_kwh"]) * 100
        )

        for school in daily_data["escuela"].unique():
            school_data = daily_data[daily_data["escuela"] == school]
            fig.add_trace(
                go.Scatter(
                    x=school_data["fecha"],
                    y=school_data["autosuficiencia"],
                    mode="lines+markers",
                    name=school.split()[-1],
                    marker=dict(size=4),
                ),
                row=1,
                col=2,
            )

        # 3. CO₂ acumulado
        for school in daily_data["escuela"].unique():
            school_data = daily_data[daily_data["escuela"] == school]
            school_data["co2_acumulado"] = school_data["co2_evitado_kg"].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=school_data["fecha"],
                    y=school_data["co2_acumulado"] / 1000,  # Toneladas
                    mode="lines",
                    name=school.split()[-1],
                    fill=(
                        "tonexty"
                        if school != daily_data["escuela"].unique()[0]
                        else None
                    ),
                ),
                row=2,
                col=1,
            )

        # 4. Ahorro acumulado
        for school in daily_data["escuela"].unique():
            school_data = daily_data[daily_data["escuela"] == school]
            school_data["ahorro_acumulado"] = school_data["ahorro_diario_usd"].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=school_data["fecha"],
                    y=school_data["ahorro_acumulado"],
                    mode="lines",
                    name=school.split()[-1],
                ),
                row=2,
                col=2,
            )

        # Actualizar layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Dashboard Interactivo - Sistemas de Energía Renovable Chimborazo",
            title_x=0.5,
            font=dict(size=10),
        )

        # Etiquetas de ejes
        fig.update_xaxes(title_text="Fecha", row=1, col=1)
        fig.update_yaxes(title_text="Energía (kWh)", row=1, col=1)

        fig.update_xaxes(title_text="Fecha", row=1, col=2)
        fig.update_yaxes(title_text="Autosuficiencia (%)", row=1, col=2)

        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="CO₂ Evitado Acumulado (ton)", row=2, col=1)

        fig.update_xaxes(title_text="Fecha", row=2, col=2)
        fig.update_yaxes(title_text="Ahorro Acumulado (USD)", row=2, col=2)

        # Guardar dashboard
        fig.write_html(f"{self.data_path}dashboard_interactivo.html")
        print(f"✅ Dashboard guardado: {self.data_path}dashboard_interactivo.html")

        return fig

    def generate_summary_report(self):
        """Generar reporte resumen con estadísticas clave"""
        print("\n" + "=" * 80)
        print("📊 REPORTE RESUMEN - ANÁLISIS DE SISTEMAS DE ENERGÍA RENOVABLE")
        print("=" * 80)

        # Estadísticas generales
        total_schools = self.df_ml["escuela"].nunique()
        total_students = self.df_ml.groupby("escuela")["students"].first().sum()
        date_range = f"{self.df_ml['fecha'].min().strftime('%Y-%m-%d')} - {self.df_ml['fecha'].max().strftime('%Y-%m-%d')}"

        print(f"🏫 Instituciones analizadas: {total_schools}")
        print(f"👥 Estudiantes beneficiados: {total_students:,}")
        print(f"📅 Período de análisis: {date_range}")
        print(f"📊 Registros procesados: {len(self.df_ml):,}")

        # Resumen por escuela
        school_summary = (
            self.df_ml.groupby("escuela")
            .agg(
                {
                    "students": "first",
                    "capacidad_solar_kw": "first",
                    "capacidad_eolica_kw": "first",
                    "capacidad_bateria_kwh": "first",
                    "costo_total_usd": "first",
                    "generacion_total_kwh": "sum",
                    "consumo_kwh": "sum",
                    "ahorro_diario_usd": "mean",
                    "co2_evitado_kg": "sum",
                    "roi_simple_años": "first",
                    "autosuficiencia_horaria_pct": "mean",
                }
            )
            .reset_index()
        )

        school_summary["ahorro_anual_usd"] = school_summary["ahorro_diario_usd"] * 365
        school_summary["co2_anual_ton"] = school_summary["co2_evitado_kg"] / 1000

        print(f"\n📈 RESUMEN POR INSTITUCIÓN:")
        print("-" * 120)
        print(
            f"{'Escuela':<20} {'Est.':<5} {'Solar':<8} {'Eólico':<8} {'Bat.':<8} {'Costo':<10} {'ROI':<6} {'Autosuf.':<8} {'CO₂/año':<8}"
        )
        print(
            f"{'':20} {'':5} {'(kW)':<8} {'(kW)':<8} {'(kWh)':<8} {'(USD)':<10} {'(años)':<6} {'(%)':<8} {'(ton)':<8}"
        )
        print("-" * 120)

        for _, row in school_summary.iterrows():
            school_short = row["escuela"].split()[-1][:15]
            print(
                f"{school_short:<20} {row['students']:<5.0f} {row['capacidad_solar_kw']:<8.1f} "
                f"{row['capacidad_eolica_kw']:<8.1f} {row['capacidad_bateria_kwh']:<8.1f} "
                f"{row['costo_total_usd']:<10,.0f} {row['roi_simple_años']:<6.1f} "
                f"{row['autosuficiencia_horaria_pct']:<8.1f} {row['co2_anual_ton']:<8.1f}"
            )

        # Totales y promedios
        total_investment = school_summary["costo_total_usd"].sum()
        total_annual_savings = school_summary["ahorro_anual_usd"].sum()
        total_co2_avoided = school_summary["co2_anual_ton"].sum()
        avg_roi = school_summary["roi_simple_años"].mean()
        avg_autosuficiencia = school_summary["autosuficiencia_horaria_pct"].mean()

        print("-" * 120)
        print(
            f"{'TOTALES/PROMEDIOS':<20} {total_students:<5.0f} {school_summary['capacidad_solar_kw'].sum():<8.1f} "
            f"{school_summary['capacidad_eolica_kw'].sum():<8.1f} {school_summary['capacidad_bateria_kwh'].sum():<8.1f} "
            f"{total_investment:<10,.0f} {avg_roi:<6.1f} {avg_autosuficiencia:<8.1f} {total_co2_avoided:<8.1f}"
        )

        print(f"\n💰 IMPACTO ECONÓMICO CONSOLIDADO:")
        print(f"   • Inversión total: ${total_investment:,.0f}")
        print(f"   • Ahorro anual: ${total_annual_savings:,.0f}")
        print(f"   • ROI promedio: {avg_roi:.1f} años")
        print(
            f"   • Payback period promedio: {total_investment/total_annual_savings:.1f} años"
        )

        print(f"\n🌱 IMPACTO AMBIENTAL CONSOLIDADO:")
        print(f"   • CO₂ evitado anual: {total_co2_avoided:.1f} toneladas")
        print(f"   • Equivalente a plantar: {total_co2_avoided*47:.0f} árboles")
        print(
            f"   • Equivalente a retirar: {total_co2_avoided/4.6:.1f} automóviles de la carretera"
        )

        print(f"\n⚡ PERFORMANCE TÉCNICO:")
        print(f"   • Autosuficiencia promedio: {avg_autosuficiencia:.1f}%")
        print(
            f"   • Capacidad instalada total: {school_summary['capacidad_solar_kw'].sum() + school_summary['capacidad_eolica_kw'].sum():.1f} kW"
        )
        print(
            f"   • Almacenamiento total: {school_summary['capacidad_bateria_kwh'].sum():.1f} kWh"
        )

        # Ranking de escuelas
        school_summary["score"] = (
            school_summary["autosuficiencia_horaria_pct"] / 100 * 0.3
            + (1 / school_summary["roi_simple_años"]) * 0.3
            + (
                school_summary["ahorro_anual_usd"]
                / school_summary["ahorro_anual_usd"].max()
            )
            * 0.4
        )

        ranking = school_summary.sort_values("score", ascending=False)

        print(f"\n🏆 RANKING DE PERFORMANCE:")
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            print(f"   {i}. {row['escuela']} (Score: {row['score']:.3f})")

        print("\n" + "=" * 80)

        # Guardar reporte en archivo
        with open(f"{self.data_path}reporte_resumen.txt", "w", encoding="utf-8") as f:
            f.write("REPORTE RESUMEN - ANÁLISIS DE SISTEMAS DE ENERGÍA RENOVABLE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Instituciones analizadas: {total_schools}\n")
            f.write(f"Estudiantes beneficiados: {total_students:,}\n")
            f.write(f"Período: {date_range}\n")
            f.write(f"Inversión total: ${total_investment:,.0f}\n")
            f.write(f"Ahorro anual: ${total_annual_savings:,.0f}\n")
            f.write(f"CO₂ evitado: {total_co2_avoided:.1f} ton/año\n")
            f.write(f"ROI promedio: {avg_roi:.1f} años\n")
            f.write(f"Autosuficiencia promedio: {avg_autosuficiencia:.1f}%\n")

        print(f"📄 Reporte guardado: {self.data_path}reporte_resumen.txt")


def main():
    """Función principal para ejecutar todas las visualizaciones"""
    print("🎨 Iniciando Dashboard de Visualización")
    print("Sistema de Energías Renovables - Chimborazo, Ecuador")
    print("=" * 60)

    # Crear dashboard
    dashboard = RenewableEnergyDashboard()

    # Verificar que los datos se cargaron correctamente
    if not hasattr(dashboard, "df_ml"):
        print("❌ No se pudieron cargar los datos. Saliendo...")
        return

    try:
        print("\n📊 Generando análisis climático...")
        dashboard.plot_climate_patterns()

        print("\n⚡ Generando análisis energético...")
        dashboard.plot_energy_analysis()

        print("\n🔗 Generando análisis de correlaciones...")
        dashboard.plot_correlation_analysis()

        print("\n💰 Generando análisis económico...")
        dashboard.plot_economic_analysis()

        print("\n🏆 Generando análisis comparativo...")
        dashboard.plot_comparative_performance()

        print("\n🌐 Creando dashboard interactivo...")
        dashboard.create_interactive_dashboard()

        print("\n📋 Generando reporte resumen...")
        dashboard.generate_summary_report()

        print("\n✅ ¡Análisis completo! Archivos generados:")
        print("   📈 clima_patterns.png")
        print("   ⚡ energy_analysis.png")
        print("   🔗 correlation_analysis.png")
        print("   💰 economic_analysis.png")
        print("   🏆 comparative_performance.png")
        print("   🌐 dashboard_interactivo.html")
        print("   📄 reporte_resumen.txt")

        print(f"\n🎯 PRÓXIMOS PASOS:")
        print("1. Revisar las visualizaciones generadas")
        print("2. Abrir dashboard_interactivo.html en navegador")
        print("3. Entrenar modelo ML: python modelo_ml_dimensionamiento.py")
        print("4. Usar utilidades: python utilidades_sistema_renovables.py")

    except Exception as e:
        print(f"❌ Error durante la generación: {e}")
        print("Verificar que el archivo de datos existe y es válido")


if __name__ == "__main__":
    main()
