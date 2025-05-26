import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
import random
from datetime import datetime


class EnergyDataGenerator:
    def __init__(self, start_date="2021-01-01", end_date="2025-04-30"):
        """
        Inicializa el generador de datos sintéticos para redes híbridas de energía.

        Parameters:
        -----------
        start_date : str
            Fecha de inicio en formato 'YYYY-MM-DD'
        end_date : str
            Fecha de fin en formato 'YYYY-MM-DD'
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Datos de muestra: 5 instituciones educativas
        self.school_names = [
            "Escuela 21 de Abril",
            "Escuela Sangay",
            "Politécnica de Chimborazo",
            "Colegio Condorazo",
            "Colegio Victor Proaño",
        ]

        # Constantes para la simulación
        self.costo_kwh = 0.10  # $0.10 por kWh
        self.max_pago_mensual = 160  # Pago máximo mensual

        # Capacidades instaladas (en kW) - valores simulados
        self.capacidades_solares = {
            "Escuela 21 de Abril": 5.0,
            "Escuela Sangay": 3.5,
            "Politécnica de Chimborazo": 15.0,
            "Colegio Condorazo": 8.0,
            "Colegio Victor Proaño": 6.5,
        }

        self.capacidades_eolicas = {
            "Escuela 21 de Abril": 2.0,
            "Escuela Sangay": 1.0,
            "Politécnica de Chimborazo": 5.0,
            "Colegio Condorazo": 3.0,
            "Colegio Victor Proaño": 1.5,
        }

        # Capacidad de almacenamiento (en kWh)
        self.capacidad_almacenamiento = {
            "Escuela 21 de Abril": 10.0,
            "Escuela Sangay": 8.0,
            "Politécnica de Chimborazo": 30.0,
            "Colegio Condorazo": 15.0,
            "Colegio Victor Proaño": 12.0,
        }

        # Configuración de tamaño - más grande = más consumo
        self.tamano_escuela = {
            "Escuela 21 de Abril": 0.6,
            "Escuela Sangay": 0.4,
            "Politécnica de Chimborazo": 1.0,
            "Colegio Condorazo": 0.8,
            "Colegio Victor Proaño": 0.7,
        }

        # Coeficientes de eficiencia energética (1.0 = estándar)
        self.eficiencia = {
            "Escuela 21 de Abril": 0.85,
            "Escuela Sangay": 0.75,
            "Politécnica de Chimborazo": 0.90,
            "Colegio Condorazo": 0.80,
            "Colegio Victor Proaño": 0.82,
        }

    def generate_date_range(self):
        """
        Genera un rango de fechas desde la fecha de inicio hasta la fecha de fin.
        """
        days = (self.end_date - self.start_date).days + 1
        return [self.start_date + timedelta(days=i) for i in range(days)]

    def simulate_weather_data(self, dates):
        """
        Simula datos climáticos para el período especificado.
        """
        # Convertir las fechas a números ordinales para facilitar la simulación periódica
        days = [(date - self.start_date).days for date in dates]

        # Temperatura (°C): variación estacional con ruido aleatorio
        temp_base = 20  # Temperatura base promedio para Riobamba
        temp_amp = 5  # Amplitud de la oscilación estacional

        # Creamos variación estacional (ciclo anual)
        temperatures = [
            temp_base + temp_amp * np.sin(2 * np.pi * day / 365) for day in days
        ]

        # Añadimos ruido diario
        temperatures = [t + np.random.normal(0, 2) for t in temperatures]

        # Radiación solar (kWh/m²/day): mayor en verano, menor en invierno
        # Para Ecuador, la radiación es más estable pero varía por nubosidad
        solar_base = 5.0  # Valor base para Riobamba (promedio)
        solar_amp = 1.0  # Amplitud de variación

        solar_radiation = [
            solar_base
            + solar_amp * np.sin(2 * np.pi * day / 365)
            + np.random.normal(0, 0.8)
            for day in days
        ]
        # Nos aseguramos que no haya valores negativos
        solar_radiation = [max(0.5, rad) for rad in solar_radiation]

        # Velocidad del viento (m/s): varía estacionalmente y diariamente
        wind_base = 3.0  # Velocidad base
        wind_amp = 1.5  # Amplitud de variación

        wind_speed = [
            wind_base
            + wind_amp * np.sin(2 * np.pi * day / 182.5)
            + np.random.normal(0, 1.2)
            for day in days
        ]

        # Nos aseguramos que no haya valores negativos
        wind_speed = [max(0.1, w) for w in wind_speed]

        # Nubosidad (0-10): afecta a la radiación solar
        cloudiness = [
            5 + 3 * np.sin(2 * np.pi * (day + 45) / 365) + np.random.normal(0, 2)
            for day in days
        ]

        # Limitamos a rango 0-10
        cloudiness = [max(1, min(10, c)) for c in cloudiness]

        # Para simular eventos climáticos realistas, reducimos la radiación solar en días nublados
        for i in range(len(dates)):
            if cloudiness[i] > 7:  # Día muy nublado
                solar_radiation[i] *= (
                    1 - (cloudiness[i] - 7) / 6
                )  # Reducción proporcional

        # Precipitación (mm): días aleatorios con lluvia, más frecuentes en ciertas épocas
        precipitation = []
        for i, day in enumerate(days):
            # Probabilidad de lluvia basada en temporada (mayor en meses lluviosos)
            season_factor = np.sin(
                2 * np.pi * (day + 90) / 365
            )  # Desfase para época lluviosa
            rain_prob = 0.3 + 0.2 * season_factor

            if random.random() < rain_prob:
                # Si llueve, la cantidad depende de la nubosidad
                rain_amount = random.gammavariate(1, cloudiness[i] / 3)
                precipitation.append(rain_amount)
            else:
                precipitation.append(0)

        # Generar DataFrame con datos climáticos
        weather_data = pd.DataFrame(
            {
                "Fecha": dates,
                "Temperatura": temperatures,
                "Radiacion_Solar": solar_radiation,
                "Velocidad_Viento": wind_speed,
                "Nubosidad": cloudiness,
                "Precipitacion": precipitation,
            }
        )

        return weather_data

    def model_energy_generation(self, weather_data, escuela):
        """
        Modela la generación de energía solar y eólica basada en datos climáticos.
        """
        # Extraer datos relevantes
        solar_radiation = weather_data["Radiacion_Solar"].values
        wind_speed = weather_data["Velocidad_Viento"].values
        temperature = weather_data["Temperatura"].values

        # Capacidad instalada de la escuela específica
        solar_capacity = self.capacidades_solares[escuela]
        wind_capacity = self.capacidades_eolicas[escuela]

        # Factor de rendimiento de panel solar (afectado por temperatura)
        # Los paneles solares son menos eficientes a altas temperaturas
        temp_coef = (
            -0.004
        )  # Coeficiente de temperatura típico (-0.4% por °C sobre 25°C)

        # Generación solar (kWh)
        solar_generation = []
        for i in range(len(solar_radiation)):
            # Factor de rendimiento afectado por temperatura
            temp_factor = 1 + temp_coef * max(0, temperature[i] - 25)
            # Horas de sol productivas (aproximación)
            sun_hours = (
                5 + 2 * solar_radiation[i] / 7
            )  # entre 5-7 horas según radiación
            # Generación diaria estimada
            daily_gen = (
                solar_capacity * solar_radiation[i] / 5.0 * sun_hours * temp_factor
            )
            # Añadimos variabilidad por mantenimiento y otros factores
            daily_gen *= random.uniform(0.9, 1.0)
            solar_generation.append(max(0, daily_gen))

        # Generación eólica (kWh)
        wind_generation = []
        for speed in wind_speed:
            # Modelado simplificado de la curva de potencia de un aerogenerador
            if speed < 3.0:  # Velocidad de arranque
                power_factor = 0
            elif speed < 13.0:  # Rango operativo
                # Aproximación cúbica simplificada de la curva de potencia
                power_factor = min(1.0, (speed - 3.0) ** 3 / 1000)
            else:  # Velocidad de corte (se apaga por seguridad)
                power_factor = 0

            # Generación diaria estimada (24 horas)
            daily_gen = wind_capacity * power_factor * 24
            # Añadimos variabilidad
            daily_gen *= random.uniform(0.85, 1.0)
            wind_generation.append(max(0, daily_gen))

        return solar_generation, wind_generation

    def model_energy_consumption(self, dates, escuela):
        """
        Modela el consumo energético diario basado en patrones realistas para centros educativos.
        """
        consumption = []

        # Factores base por escuela
        size_factor = self.tamano_escuela[escuela]  # Tamaño relativo
        efficiency_factor = self.eficiencia[escuela]  # Eficiencia energética

        for date in dates:
            # Verificar si es fin de semana (0=lunes, 6=domingo)
            is_weekend = date.weekday() >= 5

            # Verificar si está en periodo de vacaciones (julio-agosto, y parte de diciembre-enero)
            month = date.month
            day = date.day

            is_vacation = (
                (month == 7 and day >= 15)
                or (month == 8)
                or (month == 9 and day <= 7)
                or (month == 12 and day >= 22)
                or (month == 1 and day <= 5)
            )

            # Consumo base diario (kWh) según tamaño de la institución
            # Se modela según el tipo de día
            if is_weekend:
                base_consumption = 10 * size_factor  # Menor consumo en fin de semana
            elif is_vacation:
                base_consumption = 15 * size_factor  # Reducido durante vacaciones
            else:
                base_consumption = 50 * size_factor  # Consumo normal en día escolar

            # Ajustar por eficiencia energética
            base_consumption /= efficiency_factor

            # Añadir variabilidad aleatoria (±15%)
            daily_consumption = base_consumption * random.uniform(0.85, 1.15)

            # Para la Politécnica (universidad), el consumo es más estable incluso en vacaciones
            if escuela == "Politécnica de Chimborazo":
                if is_vacation:
                    daily_consumption *= 1.5  # Menor reducción en vacaciones
                if is_weekend:
                    daily_consumption *= 1.3  # Menor reducción en fines de semana

            consumption.append(daily_consumption)

        return consumption

    def model_battery_storage(self, solar_gen, wind_gen, consumption, escuela):
        """
        Modela el almacenamiento en baterías y el balance energético diario.
        """
        # Capacidad de almacenamiento para esta escuela
        storage_capacity = self.capacidad_almacenamiento[escuela]

        # Eficiencia de carga/descarga de la batería
        batt_efficiency = 0.9

        # Inicializamos el estado de carga (SOC) de la batería
        soc = storage_capacity * 0.5  # Comenzamos con 50% de carga

        grid_import = []
        grid_export = []
        battery_soc = []

        for i in range(len(solar_gen)):
            # Generación total renovable
            total_renewable = solar_gen[i] + wind_gen[i]

            # Balance energético inicial
            energy_balance = total_renewable - consumption[i]

            # Almacenamos el SoC actual
            battery_soc.append(soc)

            # Caso 1: Exceso de energía (generación > consumo)
            if energy_balance > 0:
                # Cuánto podemos cargar en la batería
                max_charge = min(
                    energy_balance, (storage_capacity - soc) / batt_efficiency
                )
                soc += (
                    max_charge * batt_efficiency
                )  # Cargamos batería con pérdidas por eficiencia

                # Si aún hay exceso, lo exportamos a la red
                remaining_excess = energy_balance - max_charge
                grid_export.append(max(0, remaining_excess))
                grid_import.append(0)  # No importamos de la red

            # Caso 2: Déficit de energía (consumo > generación)
            else:
                energy_deficit = abs(energy_balance)

                # Cuánto podemos extraer de la batería
                max_discharge = min(energy_deficit, soc)
                soc -= max_discharge  # Descargamos la batería

                # Si aún hay déficit, importamos de la red
                remaining_deficit = energy_deficit - max_discharge
                grid_import.append(max(0, remaining_deficit))
                grid_export.append(0)  # No exportamos a la red

        return grid_import, grid_export, battery_soc

    def calculate_costs_and_savings(self, consumption, grid_import, grid_export):
        """
        Calcula costos, ahorros y autosuficiencia energética.
        """
        # Costo por kWh
        costo_kwh = self.costo_kwh

        # Precios diferenciados para exportación a la red (Feed-in Tariff)
        feed_in_tariff = 0.08  # $0.08 por kWh exportado a la red

        # Cálculo de costos mensuales
        total_costs = []
        savings = []
        self_sufficiency = []

        # Calculamos el costo sin el sistema renovable (solo consumo)
        baseline_cost = [c * costo_kwh for c in consumption]

        # Calculamos el costo real (importación - exportación)
        actual_cost = [
            (imp * costo_kwh) - (exp * feed_in_tariff)
            for imp, exp in zip(grid_import, grid_export)
        ]

        # Aseguramos que el costo no sea negativo (en caso de exportación masiva)
        actual_cost = [max(0, cost) for cost in actual_cost]

        # Calculamos el ahorro diario
        daily_savings = [base - act for base, act in zip(baseline_cost, actual_cost)]

        # Calculamos la autosuficiencia diaria (% de energía no tomada de la red)
        for i in range(len(consumption)):
            if consumption[i] > 0:
                sufficiency = (consumption[i] - grid_import[i]) / consumption[i] * 100
                self_sufficiency.append(max(0, min(100, sufficiency)))
            else:
                self_sufficiency.append(
                    100
                )  # Si no hay consumo, es 100% autosuficiente

        return baseline_cost, actual_cost, daily_savings, self_sufficiency

    def aggregate_to_monthly(self, dates, *data_series):
        """
        Agrega datos diarios a mensuales.
        """
        # Crear un DataFrame temporal con fechas y datos
        temp_df = pd.DataFrame({"date": dates})

        # Añadir cada serie de datos como columna
        for i, series in enumerate(data_series):
            temp_df[f"data_{i}"] = series

        # Extraer mes y año para agrupación
        temp_df["year_month"] = pd.to_datetime(temp_df["date"]).dt.to_period("M")

        # Agrupar y sumar por mes
        monthly_data = []
        grouped = temp_df.groupby("year_month")

        for name, group in grouped:
            month_values = [name.to_timestamp()]  # Primer elemento es la fecha

            # Sumar o promediar cada serie según corresponda
            for i in range(len(data_series)):
                if i in [3]:  # Promedio para autosuficiencia
                    month_values.append(group[f"data_{i}"].mean())
                else:  # Suma para el resto
                    month_values.append(group[f"data_{i}"].sum())

            monthly_data.append(month_values)

        return monthly_data

    def export_to_excel(self, data_df, filename="datos_energeticos.xlsx"):
        """
        Exporta los datos a un archivo Excel.
        """
        # Crear una copia de los datos para exportar
        export_df = data_df.copy()

        # Formatear la fecha para que sea más legible en Excel
        export_df["Fecha"] = pd.to_datetime(export_df["Fecha"]).dt.strftime("%Y-%m-%d")

        # Crear un escritor de Excel
        writer = pd.ExcelWriter(filename, engine="openpyxl")

        # Exportar datos diarios a una hoja
        export_df.to_excel(writer, sheet_name="Datos_Diarios", index=False)

        # Crear datos mensuales para exportar
        monthly_df = export_df.copy()
        monthly_df["Mes"] = (
            pd.to_datetime(monthly_df["Fecha"]).dt.to_period("M").dt.strftime("%Y-%m")
        )

        # Agrupar por mes e institución
        monthly_agg = (
            monthly_df.groupby(["Mes", "Institucion"])
            .agg(
                {
                    "Consumo_kWh": "sum",
                    "Generacion_Solar_kWh": "sum",
                    "Generacion_Eolica_kWh": "sum",
                    "Importacion_Red_kWh": "sum",
                    "Exportacion_Red_kWh": "sum",
                    "Costo_Real_USD": "sum",
                    "Costo_Referencia_USD": "sum",
                    "Ahorro_USD": "sum",
                    "Autosuficiencia_Pct": "mean",
                }
            )
            .reset_index()
        )

        # Exportar datos mensuales a otra hoja
        monthly_agg.to_excel(writer, sheet_name="Datos_Mensuales", index=False)

        # Calcular estadísticas por institución
        stats_by_school = export_df.groupby("Institucion").agg(
            {
                "Consumo_kWh": ["sum", "mean", "std"],
                "Generacion_Solar_kWh": ["sum", "mean", "std"],
                "Generacion_Eolica_kWh": ["sum", "mean", "std"],
                "Costo_Real_USD": ["sum", "mean", "std"],
                "Ahorro_USD": ["sum", "mean", "std"],
                "Autosuficiencia_Pct": ["mean", "min", "max"],
            }
        )

        # Exportar estadísticas a otra hoja
        stats_by_school.to_excel(writer, sheet_name="Estadisticas_Institucion")

        # Guardar el archivo Excel
        writer.close()

        print(f"Datos exportados a {filename}")

    def generate_complete_dataset(self):
        """
        Genera un conjunto completo de datos sintéticos para todas las escuelas.
        """
        # Generar fechas
        dates = self.generate_date_range()

        # Simular datos climáticos (compartidos para todas las escuelas en la misma región)
        weather_data = self.simulate_weather_data(dates)

        # Listas para almacenar resultados por escuela
        all_school_data = []

        for escuela in self.school_names:
            print(f"Generando datos para {escuela}...")

            # Modelar generación energética
            solar_gen, wind_gen = self.model_energy_generation(weather_data, escuela)

            # Modelar consumo energético
            consumption = self.model_energy_consumption(dates, escuela)

            # Modelar almacenamiento en baterías y balance con la red
            grid_import, grid_export, battery_soc = self.model_battery_storage(
                solar_gen, wind_gen, consumption, escuela
            )

            # Calcular costos, ahorros y autosuficiencia
            baseline_cost, actual_cost, savings, self_sufficiency = (
                self.calculate_costs_and_savings(consumption, grid_import, grid_export)
            )

            # Crear DataFrame diario
            daily_df = pd.DataFrame(
                {
                    "Fecha": dates,
                    "Institucion": escuela,
                    "Consumo_kWh": consumption,
                    "Generacion_Solar_kWh": solar_gen,
                    "Generacion_Eolica_kWh": wind_gen,
                    "Importacion_Red_kWh": grid_import,
                    "Exportacion_Red_kWh": grid_export,
                    "Estado_Bateria_kWh": battery_soc,
                    "Costo_Referencia_USD": baseline_cost,
                    "Costo_Real_USD": actual_cost,
                    "Ahorro_USD": savings,
                    "Autosuficiencia_Pct": self_sufficiency,
                    "Temperatura_C": weather_data["Temperatura"],
                    "Radiacion_Solar_kWh_m2": weather_data["Radiacion_Solar"],
                    "Velocidad_Viento_ms": weather_data["Velocidad_Viento"],
                }
            )

            # Agregar a la lista de resultados
            all_school_data.append(daily_df)

        # Combinar todos los datos
        combined_df = pd.concat(all_school_data, ignore_index=True)

        # Ajustar costos mensuales para que estén entre $0 y $160
        # Primero agregamos por mes e institución
        monthly_df = combined_df.copy()
        monthly_df["Mes"] = pd.to_datetime(monthly_df["Fecha"]).dt.to_period("M")

        monthly_costs = (
            monthly_df.groupby(["Mes", "Institucion"])["Costo_Real_USD"]
            .sum()
            .reset_index()
        )

        # Encontrar los valores máximo y mínimo actuales
        max_cost = monthly_costs["Costo_Real_USD"].max()
        min_cost = monthly_costs["Costo_Real_USD"].min()

        # Crear un escalador
        scaler = MinMaxScaler(feature_range=(0, self.max_pago_mensual))

        # Crear un diccionario de factores de escala por institución y mes
        scale_factors = {}

        for _, row in monthly_costs.iterrows():
            mes = row["Mes"]
            institucion = row["Institucion"]
            costo = row["Costo_Real_USD"]

            # Calcular el factor de escala
            if max_cost > min_cost:  # Evitar división por cero
                scaled_cost = scaler.fit_transform([[costo]])[0][0]
                scale_factor = scaled_cost / costo if costo > 0 else 1
            else:
                scale_factor = self.max_pago_mensual / 2 / costo if costo > 0 else 1

            scale_factors[(mes, institucion)] = scale_factor

        # Aplicar los factores de escala a los datos diarios
        for i, row in combined_df.iterrows():
            mes = pd.to_datetime(row["Fecha"]).to_period("M")
            institucion = row["Institucion"]

            # Obtener el factor de escala
            scale_factor = scale_factors.get((mes, institucion), 1)

            # Aplicar escala a costos y ahorros
            combined_df.at[i, "Costo_Real_USD"] *= scale_factor
            combined_df.at[i, "Costo_Referencia_USD"] *= scale_factor
            combined_df.at[i, "Ahorro_USD"] *= scale_factor

        # También debemos ajustar la relación entre el costo y el consumo
        # para mantener el costo del kWh a 10 centavos
        # Recalcular consumo basado en costo de referencia
        for i, row in combined_df.iterrows():
            cost_ref = row["Costo_Referencia_USD"]
            implied_consumption = cost_ref / self.costo_kwh

            # Ajustar consumo y generación proporcionalmente
            scaling = (
                implied_consumption / row["Consumo_kWh"]
                if row["Consumo_kWh"] > 0
                else 1
            )

            combined_df.at[i, "Consumo_kWh"] = implied_consumption
            combined_df.at[i, "Generacion_Solar_kWh"] *= scaling
            combined_df.at[i, "Generacion_Eolica_kWh"] *= scaling
            combined_df.at[i, "Importacion_Red_kWh"] *= scaling
            combined_df.at[i, "Exportacion_Red_kWh"] *= scaling
            combined_df.at[i, "Estado_Bateria_kWh"] *= scaling

        return combined_df

    def visualize_data(self, data_df):
        """
        Genera visualizaciones estadísticas para los datos.
        """
        # Directorio para guardar las visualizaciones
        output_dir = "visualizaciones"
        os.makedirs(output_dir, exist_ok=True)

        # Configuración de estilo para los gráficos
        plt.style.use("seaborn-darkgrid")
        colors = sns.color_palette("viridis", n_colors=len(self.school_names))
        school_colors = dict(zip(self.school_names, colors))

        # 1. Consumo energético mensual por institución
        plt.figure(figsize=(14, 8))

        # Agregar datos por mes
        data_monthly = data_df.copy()
        data_monthly["Mes"] = (
            pd.to_datetime(data_monthly["Fecha"]).dt.to_period("M").dt.to_timestamp()
        )
        monthly_consumption = (
            data_monthly.groupby(["Mes", "Institucion"])["Consumo_kWh"]
            .sum()
            .reset_index()
        )

        for escuela in self.school_names:
            school_data = monthly_consumption[
                monthly_consumption["Institucion"] == escuela
            ]
            plt.plot(
                school_data["Mes"].values,
                school_data["Consumo_kWh"].values,
                marker="o",
                linestyle="-",
                label=escuela,
                color=school_colors[escuela],
            )

        plt.title("Consumo Energético Mensual por Institución", fontsize=16)
        plt.xlabel("Mes", fontsize=12)
        plt.ylabel("Consumo (kWh)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "consumo_mensual.png"), dpi=300)

        # 2. Comparación entre generación renovable y consumo (promedio mensual)
        plt.figure(figsize=(14, 8))

        # Calcular generación total renovable
        data_monthly["Generacion_Total"] = (
            data_monthly["Generacion_Solar_kWh"] + data_monthly["Generacion_Eolica_kWh"]
        )

        # Agregar por mes
        renewable_vs_consumption = (
            data_monthly.groupby("Mes")
            .agg({"Generacion_Total": "sum", "Consumo_kWh": "sum"})
            .reset_index()
        )

        plt.plot(
            renewable_vs_consumption["Mes"].values,
            renewable_vs_consumption["Generacion_Total"].values,
            marker="o",
            linestyle="-",
            label="Generación Renovable",
            color="green",
        )
        plt.plot(
            renewable_vs_consumption["Mes"].values,
            renewable_vs_consumption["Consumo_kWh"].values,
            marker="o",
            linestyle="-",
            label="Consumo",
            color="red",
        )

        plt.title("Generación Renovable vs Consumo (Total Mensual)", fontsize=16)
        plt.xlabel("Mes", fontsize=12)
        plt.ylabel("Energía (kWh)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "generacion_vs_consumo.png"), dpi=300)

        # 3. Distribución de autosuficiencia por institución (boxplot)
        plt.figure(figsize=(14, 8))

        sns.boxplot(
            x="Institucion",
            y="Autosuficiencia_Pct",
            data=data_df,
            palette=school_colors,
        )

        plt.title(
            "Distribución de Autosuficiencia Energética por Institución", fontsize=16
        )
        plt.xlabel("Institución", fontsize=12)
        plt.ylabel("Autosuficiencia (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "autosuficiencia_boxplot.png"), dpi=300)

        # 4. Correlación entre variables meteorológicas y generación
        plt.figure(figsize=(14, 8))

        # Seleccionamos variables relevantes
        corr_vars = [
            "Temperatura_C",
            "Radiacion_Solar_kWh_m2",
            "Velocidad_Viento_ms",
            "Generacion_Solar_kWh",
            "Generacion_Eolica_kWh",
        ]

        corr_matrix = data_df[corr_vars].corr()

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)

        plt.title(
            "Correlación entre Variables Meteorológicas y Generación", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "correlacion_meteo_generacion.png"), dpi=300
        )

        # 5. Costos mensuales por institución
        plt.figure(figsize=(14, 8))

        # Agregar costos por mes
        monthly_costs = (
            data_monthly.groupby(["Mes", "Institucion"])["Costo_Real_USD"]
            .sum()
            .reset_index()
        )

        for escuela in self.school_names:
            school_data = monthly_costs[monthly_costs["Institucion"] == escuela]
            plt.plot(
                school_data["Mes"].values,
                school_data["Costo_Real_USD"].values,
                marker="o",
                linestyle="-",
                label=escuela,
                color=school_colors[escuela],
            )

        plt.title("Costo Energético Mensual por Institución", fontsize=16)
        plt.xlabel("Mes", fontsize=12)
        plt.ylabel("Costo (USD)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "costo_mensual.png"), dpi=300)

        # 6. Ahorros acumulados por institución
        plt.figure(figsize=(14, 8))

        # Calcular ahorros acumulados
        monthly_savings = (
            data_monthly.groupby(["Mes", "Institucion"])["Ahorro_USD"]
            .sum()
            .reset_index()
        )

        # Crear un DataFrame para almacenar ahorros acumulados
        cumulative_savings = pd.DataFrame()

        for escuela in self.school_names:
            school_data = monthly_savings[
                monthly_savings["Institucion"] == escuela
            ].copy()
            school_data = school_data.sort_values("Mes")
            school_data["Ahorro_Acumulado"] = school_data["Ahorro_USD"].cumsum()

            plt.plot(
                school_data["Mes"].values,
                school_data["Ahorro_Acumulado"].values,
                marker="o",
                linestyle="-",
                label=escuela,
                color=school_colors[escuela],
            )

            # Agregar al DataFrame acumulativo
            if cumulative_savings.empty:
                cumulative_savings = school_data.copy()
            else:
                cumulative_savings = pd.concat([cumulative_savings, school_data])

        plt.title("Ahorro Acumulado por Institución", fontsize=16)
        plt.xlabel("Mes", fontsize=12)
        plt.ylabel("Ahorro Acumulado (USD)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ahorro_acumulado.png"), dpi=300)

        # 7. Relación entre radiación solar y generación solar
        plt.figure(figsize=(14, 8))

        # Crear un gráfico de dispersión con línea de tendencia
        sns.regplot(
            x="Radiacion_Solar_kWh_m2",
            y="Generacion_Solar_kWh",
            data=data_df,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
        )

        plt.title("Relación entre Radiación Solar y Generación Solar", fontsize=16)
        plt.xlabel("Radiación Solar (kWh/m²)", fontsize=12)
        plt.ylabel("Generación Solar (kWh)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radiacion_vs_generacion.png"), dpi=300)

        # 8. Relación entre velocidad del viento y generación eólica
        plt.figure(figsize=(14, 8))

        # Crear un gráfico de dispersión con línea de tendencia
        sns.regplot(
            x="Velocidad_Viento_ms",
            y="Generacion_Eolica_kWh",
            data=data_df,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "blue"},
        )

        plt.title(
            "Relación entre Velocidad del Viento y Generación Eólica", fontsize=16
        )
        plt.xlabel("Velocidad del Viento (m/s)", fontsize=12)
        plt.ylabel("Generación Eólica (kWh)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "viento_vs_generacion.png"), dpi=300)

        # 9. Composición de la generación renovable por institución
        plt.figure(figsize=(15, 10))

        # Calcular promedios mensuales
        monthly_gen = (
            data_monthly.groupby("Institucion")
            .agg({"Generacion_Solar_kWh": "sum", "Generacion_Eolica_kWh": "sum"})
            .reset_index()
        )

        # Crear gráfico de barras apiladas
        bar_width = 0.6
        indices = range(len(self.school_names))

        solar_data = [
            monthly_gen[monthly_gen["Institucion"] == school][
                "Generacion_Solar_kWh"
            ].values[0]
            for school in self.school_names
        ]
        wind_data = [
            monthly_gen[monthly_gen["Institucion"] == school][
                "Generacion_Eolica_kWh"
            ].values[0]
            for school in self.school_names
        ]

        plt.bar(indices, solar_data, bar_width, label="Generación Solar", color="gold")
        plt.bar(
            indices,
            wind_data,
            bar_width,
            bottom=solar_data,
            label="Generación Eólica",
            color="skyblue",
        )

        plt.title("Composición de la Generación Renovable por Institución", fontsize=16)
        plt.xlabel("Institución", fontsize=12)
        plt.ylabel("Generación Total (kWh)", fontsize=12)
        plt.xticks(indices, self.school_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "composicion_generacion.png"), dpi=300)

        # 10. Eficiencia económica (ahorro por kWh generado)
        plt.figure(figsize=(14, 8))

        # Calcular eficiencia económica (USD ahorrados por kWh generado)
        efficiency_data = (
            data_monthly.groupby("Institucion")
            .agg({"Ahorro_USD": "sum", "Generacion_Total": "sum"})
            .reset_index()
        )

        efficiency_data["Eficiencia_USD_kWh"] = (
            efficiency_data["Ahorro_USD"] / efficiency_data["Generacion_Total"]
        )

        # Crear gráfico de barras
        plt.bar(
            efficiency_data["Institucion"],
            efficiency_data["Eficiencia_USD_kWh"],
            color=[school_colors[school] for school in efficiency_data["Institucion"]],
        )

        plt.title("Eficiencia Económica por Institución", fontsize=16)
        plt.xlabel("Institución", fontsize=12)
        plt.ylabel("Ahorro por kWh Generado (USD/kWh)", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eficiencia_economica.png"), dpi=300)
