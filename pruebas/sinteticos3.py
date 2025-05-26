import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
import random


class GeneradorDatosSinteticos:
    def __init__(self, fecha_inicio="2021-01-01", fecha_fin="2025-04-30"):
        # Parámetros iniciales
        self.fecha_inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        self.fecha_fin = datetime.strptime(fecha_fin, "%Y-%m-%d")

        # Calcular número de días entre fechas
        delta = self.fecha_fin - self.fecha_inicio
        self.dias = delta.days + 1

        # Generar lista de fechas
        self.fechas = [self.fecha_inicio + timedelta(days=i) for i in range(self.dias)]

        # Datos de muestra: 5 instituciones educativas
        self.school_names = [
            "Escuela 21 de Abril",
            "Escuela Sangay",
            "Politécnica de Chimborazo",
            "Colegio Condorazo",
            "Colegio Victor Proaño",
        ]

        # Parámetros de energía
        self.costo_kwh = 0.10  # 10 centavos por kWh
        self.max_pago_mensual = 160  # dólares

        # Crear directorio para guardar resultados
        self.output_dir = "resultados_simulacion"
        os.makedirs(self.output_dir, exist_ok=True)

        # Semilla para reproducibilidad
        np.random.seed(42)
        random.seed(42)

    def generar_radiacion_solar(self):
        """Genera datos de radiación solar diaria (kWh/m²/día)"""
        # Base anual con estacionalidad
        x = np.arange(self.dias)
        tendencia_anual = 4.5 + 1.5 * np.sin(2 * np.pi * x / 365)

        # Añadir variabilidad diaria
        ruido = np.random.normal(0, 0.8, self.dias)

        # Añadir efectos climáticos (días nublados)
        dias_nublados = np.random.choice([0, 1], size=self.dias, p=[0.7, 0.3])
        efecto_nubes = dias_nublados * np.random.uniform(-3, -1, self.dias)

        radiacion = tendencia_anual + ruido + efecto_nubes

        # Asegurar que los valores no sean negativos
        radiacion = np.maximum(radiacion, 0.1)

        return radiacion

    def generar_velocidad_viento(self):
        """Genera datos de velocidad de viento diaria (m/s)"""
        # Base con estacionalidad
        x = np.arange(self.dias)
        tendencia_anual = 3 + 1.2 * np.sin(2 * np.pi * x / 365 + np.pi / 2)

        # Añadir variabilidad diaria
        ruido = np.random.normal(0, 0.7, self.dias)

        velocidad = tendencia_anual + ruido

        # Asegurar que los valores no sean negativos
        velocidad = np.maximum(velocidad, 0.5)

        return velocidad

    def generar_consumo_base(self, escuela):
        """Genera un patrón de consumo base para cada escuela (kWh/día)"""
        # Diferentes perfiles de consumo según tipo de institución
        if "Politécnica" in escuela:
            # Institución grande
            base = np.random.uniform(40, 60)
            variabilidad = np.random.uniform(10, 15)
        elif "Colegio" in escuela:
            # Institución mediana
            base = np.random.uniform(25, 35)
            variabilidad = np.random.uniform(5, 10)
        else:
            # Escuela pequeña
            base = np.random.uniform(15, 25)
            variabilidad = np.random.uniform(3, 8)

        return base, variabilidad

    def generar_consumo_diario(self, escuela):
        """Genera datos de consumo energético diario para una escuela (kWh/día)"""
        base, variabilidad = self.generar_consumo_base(escuela)

        # Crear vector de días (0-6 para lunes a domingo)
        dias_semana = [
            (self.fecha_inicio + timedelta(days=i)).weekday() for i in range(self.dias)
        ]

        # Crear patrón semanal: menor consumo en fines de semana
        factor_semana = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.2])
        patron_semanal = np.array([factor_semana[d] for d in dias_semana])

        # Crear patrón anual (vacaciones escolares)
        x = np.arange(self.dias)
        # En Ecuador, principales vacaciones en febrero-marzo y agosto-septiembre
        factor_feb_mar = 0.3 * (
            1 - np.exp(-0.5 * ((x - 45) / 15) ** 2)
        )  # Febrero-Marzo
        factor_ago_sep = 0.3 * (
            1 - np.exp(-0.5 * ((x - 240) / 15) ** 2)
        )  # Agosto-Septiembre
        patron_vacaciones = 1 - factor_feb_mar - factor_ago_sep

        # Añadir variabilidad diaria
        ruido = np.random.normal(0, 0.1, self.dias)

        # Combinar todos los factores
        consumo = base * patron_semanal * patron_vacaciones * (1 + ruido)

        # Añadir eventos aleatorios (como actividades especiales)
        eventos_especiales = np.zeros(self.dias)
        num_eventos = np.random.randint(5, 15)
        dias_eventos = np.random.choice(
            range(self.dias), size=num_eventos, replace=False
        )
        eventos_especiales[dias_eventos] = np.random.uniform(1.2, 1.5, num_eventos)

        consumo = consumo * (1 + eventos_especiales)

        # Añadir variabilidad entre escuelas
        consumo *= np.random.uniform(0.9, 1.1)

        # Asegurar que los valores no sean negativos
        consumo = np.maximum(consumo, 0.1)

        return consumo

    def calcular_generacion_solar(self, radiacion, capacidad_instalada):
        """Calcula la generación solar basada en la radiación y capacidad instalada"""
        # Eficiencia del sistema (entre 15% y 20%)
        eficiencia = np.random.uniform(0.15, 0.20)

        # Generación solar diaria (kWh)
        generacion = radiacion * capacidad_instalada * eficiencia

        # Añadir degradación por temperatura y otros factores
        factor_degradacion = np.random.uniform(0.85, 0.95, self.dias)

        return generacion * factor_degradacion

    def calcular_generacion_eolica(self, velocidad_viento, capacidad_instalada):
        """Calcula la generación eólica basada en la velocidad del viento y capacidad instalada"""
        # Modelo simplificado de curva de potencia para turbinas pequeñas
        # Velocidad de arranque, nominal y de corte
        v_arranque = 2.5
        v_nominal = 12.0
        v_corte = 25.0

        # Inicializar generación
        generacion = np.zeros(self.dias)

        # Aplicar curva de potencia
        for i, v in enumerate(velocidad_viento):
            if v < v_arranque:
                # Por debajo de velocidad de arranque
                generacion[i] = 0
            elif v < v_nominal:
                # Región cúbica (proporcional a v^3)
                generacion[i] = (
                    capacidad_instalada
                    * ((v - v_arranque) / (v_nominal - v_arranque)) ** 3
                )
            elif v < v_corte:
                # Potencia nominal
                generacion[i] = capacidad_instalada
            else:
                # Por encima de velocidad de corte
                generacion[i] = 0

        # Añadir eficiencia y variabilidad
        factor_eficiencia = np.random.uniform(0.85, 0.95)
        ruido = np.random.normal(0, 0.05, self.dias)

        return generacion * factor_eficiencia * (1 + ruido)

    def calcular_estado_baterias(self, generacion_total, consumo, capacidad_baterias):
        """Simula el estado de carga de las baterías a lo largo del tiempo"""
        # Estado inicial (50% de carga)
        estado_inicial = 0.5 * capacidad_baterias
        estado_baterias = np.zeros(self.dias)
        estado_baterias[0] = estado_inicial

        # Eficiencia de carga/descarga
        eficiencia_carga = 0.95
        eficiencia_descarga = 0.95

        # Simular para cada día
        for i in range(1, self.dias):
            # Balance energético del día
            balance = generacion_total[i - 1] - consumo[i - 1]

            if balance > 0:  # Excedente para cargar baterías
                carga = min(
                    balance * eficiencia_carga,
                    capacidad_baterias - estado_baterias[i - 1],
                )
                estado_baterias[i] = estado_baterias[i - 1] + carga
            else:  # Déficit que se toma de las baterías
                descarga = min(
                    abs(balance) / eficiencia_descarga, estado_baterias[i - 1]
                )
                estado_baterias[i] = estado_baterias[i - 1] - descarga

        # Añadir degradación a largo plazo
        dias_transcurridos = np.arange(self.dias) / 365.0  # En años
        factor_degradacion = 1 - 0.03 * dias_transcurridos  # 3% degradación anual

        return estado_baterias * factor_degradacion

    def calcular_energia_red(
        self, generacion_total, consumo, estado_baterias, capacidad_baterias
    ):
        """Calcula la energía tomada o inyectada a la red"""
        energia_red = np.zeros(self.dias)

        for i in range(self.dias):
            # Balance energético del día
            balance = generacion_total[i] - consumo[i]

            # Capacidad disponible en baterías
            if balance < 0:  # Déficit de energía
                # Verificar cuánto podemos tomar de las baterías
                disponible_baterias = (
                    estado_baterias[i]
                    if i == 0
                    else max(0, estado_baterias[i] - 0.2 * capacidad_baterias)
                )

                if abs(balance) <= disponible_baterias:
                    # Las baterías cubren el déficit
                    energia_red[i] = 0
                else:
                    # Necesitamos tomar de la red
                    energia_red[i] = -(abs(balance) - disponible_baterias)
            elif balance > 0:  # Excedente de energía
                # Verificar cuánto espacio hay en las baterías
                espacio_baterias = capacidad_baterias - estado_baterias[i]

                if balance <= espacio_baterias:
                    # Todo el excedente va a las baterías
                    energia_red[i] = 0
                else:
                    # Inyectamos a la red el excedente
                    energia_red[i] = balance - espacio_baterias

        return energia_red

    def calcular_factura_mensual(self, energia_red):
        """Calcula la factura mensual basada en la energía tomada de la red"""
        # Agrupar por mes
        fechas_pd = pd.to_datetime(self.fechas)
        df_energia = pd.DataFrame({"fecha": fechas_pd, "energia_red": energia_red})

        # Considerar solo la energía tomada de la red (valores negativos)
        df_energia["energia_consumida"] = np.maximum(-df_energia["energia_red"], 0)

        # Extraer año y mes como strings para evitar operaciones con datetime64
        df_energia["anio"] = df_energia["fecha"].dt.year
        df_energia["mes"] = df_energia["fecha"].dt.month
        df_energia["anio_mes"] = df_energia["fecha"].dt.strftime("%Y-%m")

        # Agrupar por año-mes y sumar valores numéricos (sin incluir la fecha)
        df_mensual = (
            df_energia.groupby("anio_mes")
            .agg({"energia_consumida": "sum"})
            .reset_index()
        )

        # Calcular factura (costo * consumo)
        df_mensual["factura"] = df_mensual["energia_consumida"] * self.costo_kwh

        # Agregar cargo fijo y otros costos
        cargo_fijo = np.random.uniform(3, 5)
        impuestos = 0.12  # 12% de impuestos

        df_mensual["cargo_fijo"] = cargo_fijo
        df_mensual["impuestos"] = df_mensual["factura"] * impuestos
        df_mensual["factura_total"] = (
            df_mensual["factura"] + df_mensual["cargo_fijo"] + df_mensual["impuestos"]
        )

        # Asegurar que no exceda el máximo
        df_mensual["factura_total"] = np.minimum(
            df_mensual["factura_total"], self.max_pago_mensual
        )

        return df_mensual

    def generar_datos_escuela(self, escuela):
        """Genera todos los datos para una escuela específica"""
        print(f"Generando datos para: {escuela}")

        # Datos meteorológicos (comunes para la región)
        radiacion_solar = self.generar_radiacion_solar()
        velocidad_viento = self.generar_velocidad_viento()

        # Datos específicos de la escuela
        if "Politécnica" in escuela:
            # Institución grande con mayor capacidad
            capacidad_solar = np.random.uniform(20, 30)  # kW
            capacidad_eolica = np.random.uniform(5, 10)  # kW
            capacidad_baterias = np.random.uniform(80, 120)  # kWh
        elif "Colegio" in escuela:
            # Institución mediana
            capacidad_solar = np.random.uniform(10, 15)  # kW
            capacidad_eolica = np.random.uniform(2, 5)  # kW
            capacidad_baterias = np.random.uniform(40, 60)  # kWh
        else:
            # Escuela pequeña
            capacidad_solar = np.random.uniform(5, 10)  # kW
            capacidad_eolica = np.random.uniform(1, 3)  # kW
            capacidad_baterias = np.random.uniform(20, 40)  # kWh

        # Consumo energético
        consumo = self.generar_consumo_diario(escuela)

        # Generación de energía renovable
        generacion_solar = self.calcular_generacion_solar(
            radiacion_solar, capacidad_solar
        )
        generacion_eolica = self.calcular_generacion_eolica(
            velocidad_viento, capacidad_eolica
        )
        generacion_total = generacion_solar + generacion_eolica

        # Estado de baterías
        estado_baterias = self.calcular_estado_baterias(
            generacion_total, consumo, capacidad_baterias
        )

        # Interacción con la red
        energia_red = self.calcular_energia_red(
            generacion_total, consumo, estado_baterias, capacidad_baterias
        )

        # Factura mensual
        factura_mensual = self.calcular_factura_mensual(energia_red)

        # Crear DataFrame con todos los datos
        df = pd.DataFrame(
            {
                "fecha": self.fechas,
                "institucion": escuela,
                "radiacion_solar_kwh_m2": radiacion_solar,
                "velocidad_viento_ms": velocidad_viento,
                "capacidad_solar_kw": capacidad_solar,
                "capacidad_eolica_kw": capacidad_eolica,
                "capacidad_baterias_kwh": capacidad_baterias,
                "consumo_kwh": consumo,
                "generacion_solar_kwh": generacion_solar,
                "generacion_eolica_kwh": generacion_eolica,
                "generacion_total_kwh": generacion_total,
                "estado_baterias_kwh": estado_baterias,
                "energia_red_kwh": energia_red,
                "autoconsumo_pct": 100
                * np.minimum(generacion_total, consumo)
                / np.maximum(consumo, 0.001),  # Evitar división por cero
            }
        )

        # Crear columnas de año y mes como strings (evitar usar period)
        df_fecha = pd.to_datetime(df["fecha"])
        df["anio"] = df_fecha.dt.year
        df["mes"] = df_fecha.dt.month
        df["anio_mes"] = df_fecha.dt.strftime("%Y-%m")

        # Agregar indicadores de eficiencia
        df["balance_energetico"] = df["generacion_total_kwh"] - df["consumo_kwh"]
        df["autosuficiencia"] = np.where(df["balance_energetico"] >= 0, 1, 0)

        return df, factura_mensual

    def generar_dataset_completo(self):
        """Genera el dataset completo para todas las escuelas"""
        # Lista para almacenar DataFrames
        dfs = []
        df_facturas = []

        # Generar datos para cada escuela
        for escuela in self.school_names:
            df_escuela, factura_mensual = self.generar_datos_escuela(escuela)
            dfs.append(df_escuela)

            # Agregar nombre de escuela a factura
            factura_mensual["institucion"] = escuela
            df_facturas.append(factura_mensual)

        # Combinar todos los DataFrames
        df_completo = pd.concat(dfs, ignore_index=True)
        df_facturas_completo = pd.concat(df_facturas, ignore_index=True)

        # Resetear el índice de facturas para tener una columna de mes
        df_facturas_completo = df_facturas_completo.reset_index()
        df_facturas_completo.rename(columns={"index": "mes"}, inplace=True)
        df_facturas_completo["mes"] = df_facturas_completo["mes"].astype(str)

        return df_completo, df_facturas_completo

    def guardar_excel(
        self, df, df_facturas, nombre_archivo="datos_simulacion_energia.xlsx"
    ):
        """Guarda los datos en un archivo Excel con múltiples hojas"""
        ruta_archivo = os.path.join(self.output_dir, nombre_archivo)

        # Crear el escritor de Excel
        with pd.ExcelWriter(ruta_archivo, engine="openpyxl") as writer:
            # Guardar datos diarios
            df.to_excel(writer, sheet_name="Datos_Diarios", index=False)

            # Guardar facturas mensuales
            df_facturas.to_excel(writer, sheet_name="Facturas_Mensuales", index=False)

            # Crear hoja de resumen
            df_resumen = (
                df.groupby("institucion")
                .agg(
                    {
                        "consumo_kwh": "sum",
                        "generacion_solar_kwh": "sum",
                        "generacion_eolica_kwh": "sum",
                        "generacion_total_kwh": "sum",
                        "autoconsumo_pct": "mean",
                        "energia_red_kwh": "sum",
                    }
                )
                .reset_index()
            )

            # Calcular métricas adicionales
            df_resumen["porcentaje_solar"] = (
                df_resumen["generacion_solar_kwh"]
                / df_resumen["generacion_total_kwh"]
                * 100
            )
            df_resumen["porcentaje_eolica"] = (
                df_resumen["generacion_eolica_kwh"]
                / df_resumen["generacion_total_kwh"]
                * 100
            )
            df_resumen["balance_energia"] = (
                df_resumen["generacion_total_kwh"] - df_resumen["consumo_kwh"]
            )
            df_resumen["autosuficiencia"] = (
                df_resumen["generacion_total_kwh"] / df_resumen["consumo_kwh"] * 100
            )

            # Guardar resumen
            df_resumen.to_excel(writer, sheet_name="Resumen", index=False)

        print(f"Datos guardados en: {ruta_archivo}")
        return ruta_archivo

    def crear_graficas(self, df, df_facturas):
        """Crea gráficas estadísticas relevantes para el análisis"""
        # Configuración estética
        plt.style.use("seaborn-darkgrid")
        colores = sns.color_palette("viridis", len(self.school_names))

        # Crear directorio para gráficas
        directorio_graficas = os.path.join(self.output_dir, "graficas")
        os.makedirs(directorio_graficas, exist_ok=True)

        # 1. Consumo vs Generación por escuela (acumulado anual)
        plt.figure(figsize=(12, 8))
        df_resumen = (
            df.groupby("institucion")
            .agg(
                {
                    "consumo_kwh": "sum",
                    "generacion_solar_kwh": "sum",
                    "generacion_eolica_kwh": "sum",
                }
            )
            .reset_index()
        )

        # Ordenar por consumo
        df_resumen = df_resumen.sort_values("consumo_kwh", ascending=False)

        # Crear gráfico de barras apiladas
        ax = df_resumen.plot(
            x="institucion",
            y=["generacion_solar_kwh", "generacion_eolica_kwh", "consumo_kwh"],
            kind="bar",
            stacked=False,
            color=["#ffd700", "#4682b4", "#ff6347"],
            figsize=(12, 8),
        )

        plt.title("Consumo vs Generación Anual por Institución", fontsize=16)
        plt.xlabel("Institución Educativa", fontsize=14)
        plt.ylabel("Energía (kWh)", fontsize=14)
        plt.legend(
            ["Generación Solar", "Generación Eólica", "Consumo"], loc="upper right"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(directorio_graficas, "consumo_vs_generacion.png"), dpi=300
        )

        # 2. Nivel de autosuficiencia por escuela
        plt.figure(figsize=(10, 6))
        df_auto = (
            df.groupby("institucion")
            .agg({"generacion_total_kwh": "sum", "consumo_kwh": "sum"})
            .reset_index()
        )

        df_auto["autosuficiencia"] = (
            df_auto["generacion_total_kwh"] / df_auto["consumo_kwh"] * 100
        )
        df_auto = df_auto.sort_values("autosuficiencia", ascending=False)

        sns.barplot(
            x="institucion", y="autosuficiencia", data=df_auto, palette="viridis"
        )

        plt.title("Nivel de Autosuficiencia Energética", fontsize=16)
        plt.xlabel("Institución Educativa", fontsize=14)
        plt.ylabel("Autosuficiencia (%)", fontsize=14)
        plt.axhline(
            y=100, color="r", linestyle="--", alpha=0.7, label="100% Autosuficiencia"
        )
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(directorio_graficas, "autosuficiencia.png"), dpi=300)

        # 3. Evolución mensual de consumo, generación y factura
        plt.figure(figsize=(14, 8))

        # Preparar datos mensuales
        df["mes"] = pd.to_datetime(df["fecha"]).dt.to_period("M")
        df_mensual = (
            df.groupby(["mes", "institucion"])
            .agg({"consumo_kwh": "sum", "generacion_total_kwh": "sum"})
            .reset_index()
        )

        # Convertir período a datetime para gráfica
        df_mensual["mes_fecha"] = df_mensual["mes"].dt.to_timestamp()

        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # Plot de consumo y generación
        for i, escuela in enumerate(df_auto["institucion"]):
            df_esc = df_mensual[df_mensual["institucion"] == escuela]
            ax1.plot(
                df_esc["mes_fecha"],
                df_esc["consumo_kwh"],
                "o-",
                color=colores[i],
                label=f"{escuela} - Consumo",
            )
            ax1.plot(
                df_esc["mes_fecha"],
                df_esc["generacion_total_kwh"],
                "s--",
                color=colores[i],
                alpha=0.6,
                label=f"{escuela} - Generación",
            )

        ax1.set_title("Evolución Mensual de Consumo y Generación", fontsize=16)
        ax1.set_ylabel("Energía (kWh)", fontsize=14)
        ax1.legend(loc="upper right", ncol=2)
        ax1.grid(True, alpha=0.3)

        # Plot de factura
        # Convertir 'mes' a datetime para poder graficar
        df_facturas["mes_fecha"] = pd.to_datetime(
            [f"{m}-01" for m in df_facturas["mes"]]
        )

        for i, escuela in enumerate(df_auto["institucion"]):
            df_fact_esc = df_facturas[df_facturas["institucion"] == escuela]
            ax2.plot(
                df_fact_esc["mes_fecha"],
                df_fact_esc["factura_total"],
                "o-",
                color=colores[i],
                label=escuela,
            )

        ax2.set_title("Evolución Mensual de la Factura Eléctrica", fontsize=16)
        ax2.set_xlabel("Mes", fontsize=14)
        ax2.set_ylabel("Factura ($)", fontsize=14)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(directorio_graficas, "evolucion_mensual.png"), dpi=300)

        # 4. Distribución de generación renovable (solar vs eólica)
        plt.figure(figsize=(10, 10))

        # Calcular porcentajes
        df_pie = (
            df.groupby("institucion")
            .agg({"generacion_solar_kwh": "sum", "generacion_eolica_kwh": "sum"})
            .reset_index()
        )

        # Crear subplots para cada escuela
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        axs = axs.flatten()

        for i, escuela in enumerate(df_pie["institucion"]):
            if i < len(axs):
                df_esc = df_pie[df_pie["institucion"] == escuela]
                valores = [
                    df_esc["generacion_solar_kwh"].values[0],
                    df_esc["generacion_eolica_kwh"].values[0],
                ]
                labels = ["Solar", "Eólica"]

                axs[i].pie(
                    valores,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=["#FFA500", "#4682B4"],
                )
                axs[i].set_title(
                    f"Distribución de Generación Renovable\n{escuela}", fontsize=14
                )
                axs[i].axis("equal")

        # Ocultar subplot vacío si hay uno
        if len(df_pie["institucion"]) < len(axs):
            axs[-1].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(directorio_graficas, "distribucion_renovable.png"), dpi=300
        )

        # 5. Correlación entre variables meteorológicas y generación
        plt.figure(figsize=(12, 10))

        # Crear un DataFrame de correlación
        df_corr = df[
            [
                "radiacion_solar_kwh_m2",
                "velocidad_viento_ms",
                "generacion_solar_kwh",
                "generacion_eolica_kwh",
                "consumo_kwh",
                "autoconsumo_pct",
            ]
        ].copy()

        # Calcular matriz de correlación
        corr_matrix = df_corr.corr()

        # Crear heatmap
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f"
        )
        plt.title("Matriz de Correlación entre Variables", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(directorio_graficas, "matriz_correlacion.png"), dpi=300
        )

        # 6. Distribución diaria del consumo y generación
        plt.figure(figsize=(14, 8))

        # Seleccionar una semana representativa para visualizar
        semana_inicio = self.fecha_inicio + timedelta(
            days=90
        )  # Aproximadamente en abril
        semana_fin = semana_inicio + timedelta(days=7)

        df_semana = df[
            (df["fecha"] >= semana_inicio) & (df["fecha"] < semana_fin)
        ].copy()

        # Crear un gráfico para cada escuela
        for i, escuela in enumerate(self.school_names):
            plt.figure(figsize=(12, 6))

            df_esc_semana = df_semana[df_semana["institucion"] == escuela]

            plt.plot(
                df_esc_semana["fecha"],
                df_esc_semana["consumo_kwh"],
                "r-",
                label="Consumo",
            )
            plt.plot(
                df_esc_semana["fecha"],
                df_esc_semana["generacion_solar_kwh"],
                "y-",
                label="Gen. Solar",
            )
            plt.plot(
                df_esc_semana["fecha"],
                df_esc_semana["generacion_eolica_kwh"],
                "b-",
                label="Gen. Eólica",
            )
            plt.plot(
                df_esc_semana["fecha"],
                df_esc_semana["estado_baterias_kwh"],
                "g--",
                label="Estado Baterías",
            )

            plt.title(f"Perfil Energético Semanal - {escuela}", fontsize=16)
            plt.xlabel("Fecha", fontsize=14)
            plt.ylabel("Energía (kWh)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(
                os.path.join(directorio_graficas, f"perfil_semanal_{i+1}.png"), dpi=300
            )

        # 7. Relación entre radiación solar y generación solar
        plt.figure(figsize=(10, 6))

        # Crear un scatter plot con línea de regresión
        sns.regplot(
            x="radiacion_solar_kwh_m2",
            y="generacion_solar_kwh",
            data=df.sample(n=min(1000, len(df))),
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"},
        )

        plt.title("Relación entre Radiación Solar y Generación Solar", fontsize=16)
        plt.xlabel("Radiación Solar (kWh/m²/día)", fontsize=14)
        plt.ylabel("Generación Solar (kWh)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(directorio_graficas, "relacion_radiacion_generacion.png"),
            dpi=300,
        )

        # 8. Relación entre velocidad del viento y generación eólica
        plt.figure(figsize=(10, 6))

        # Crear un scatter plot con línea de regresión
        sns.regplot(
            x="velocidad_viento_ms",
            y="generacion_eolica_kwh",
            data=df.sample(n=min(1000, len(df))),
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "blue"},
        )

        plt.title(
            "Relación entre Velocidad del Viento y Generación Eólica", fontsize=16
        )
        plt.xlabel("Velocidad del Viento (m/s)", fontsize=14)
        plt.ylabel("Generación Eólica (kWh)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(directorio_graficas, "relacion_viento_generacion.png"), dpi=300
        )

        print(f"Gráficas guardadas en: {directorio_graficas}")


def main():
    """Función principal para generar los datos sintéticos"""
    # Inicializar generador
    generador = GeneradorDatosSinteticos(
        fecha_inicio="2021-01-01", fecha_fin="2025-04-30"
    )

    # Generar dataset completo
    print("Generando dataset completo...")
    df, df_facturas = generador.generar_dataset_completo()

    # Guardar en Excel
    print("Guardando datos en Excel...")
    ruta_archivo = generador.guardar_excel(df, df_facturas)

    # Crear gráficas
    print("Generando gráficas estadísticas...")
    generador.crear_graficas(df, df_facturas)

    print("\nProceso completado con éxito.")
    print(f"Archivo Excel generado: {ruta_archivo}")
    print(f"Gráficas guardadas en: {os.path.join(generador.output_dir, 'graficas')}")


if __name__ == "__main__":
    main()
