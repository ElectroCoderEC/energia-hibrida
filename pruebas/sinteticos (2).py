import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import t


class RenewableEnergySimulator:
    """
    Simulador de redes híbridas de energía renovable para centros educativos rurales
    en Riobamba, Ecuador.
    """

    def __init__(self):
        """Inicializa el simulador con los nombres de las escuelas."""
        # Datos de muestra: 5 instituciones educativas
        self.school_names = [
            "Escuela 21 de Abril",
            "Escuela Sangay",
            "Politécnica de Chimborazo",
            "Colegio Condorazo",
            "Colegio Victor Proaño",
        ]

        # Directorio para guardar resultados
        self.output_dir = "resultados"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_synthetic_data(
        self,
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2025, 4, 30),
        time_step=1,
    ):
        """
        Genera datos sintéticos para las 5 instituciones educativas incluyendo:
        - Radiación solar
        - Velocidad del viento
        - Consumo energético
        - Temperatura ambiental

        Args:
            start_date: Fecha de inicio de la simulación (default: 1-Ene-2021)
            end_date: Fecha de fin de la simulación (default: 30-Abr-2025)
            time_step: Paso de tiempo en horas (default: 1)

        Returns:
            dict: Datos generados para cada escuela
        """
        # Calcular número de días
        days = (end_date - start_date).days + 1
        print(
            f"Generando datos sintéticos desde {start_date.strftime('%d-%b-%Y')} hasta {end_date.strftime('%d-%b-%Y')} ({days} días) con paso de {time_step} horas..."
        )

        # Parámetros base para Riobamba (ajustar según datos reales)
        latitude = -1.67  # Latitud de Riobamba
        base_solar_radiation = 5.2  # kWh/m²/día en promedio
        base_wind_speed = 3.5  # m/s en promedio
        base_temperature = 14.0  # °C en promedio

        # Diccionario para almacenar datos por escuela
        self.data = {}

        # Horas en el período de simulación
        hours = days * 24
        timesteps = int(hours / time_step)

        # Fechas para el período de simulación
        dates = [start_date + timedelta(hours=i * time_step) for i in range(timesteps)]

        for school in self.school_names:
            # Factores aleatorios para crear variabilidad entre escuelas
            size_factor = np.random.uniform(0.8, 1.5)  # Tamaño relativo de la escuela
            modern_factor = np.random.uniform(0.7, 1.3)  # Modernidad de instalaciones

            # Generar series temporales
            time_idx = np.arange(timesteps)

            # 1. Radiación solar (kW/m²)
            # Componentes: tendencia estacional + patrón diario + ruido
            seasonal_component = 0.2 * np.sin(
                2 * np.pi * time_idx / (365 * 24 / time_step)
            )  # Variación anual
            daily_component = np.array(
                [
                    max(
                        0,
                        np.sin(
                            2 * np.pi * (i % (24 / time_step)) / (24 / time_step) - 0.2
                        ),
                    )
                    for i in time_idx
                ]
            )  # Patrón diario
            noise = np.random.normal(0, 0.05, timesteps)
            cloud_events = np.zeros(timesteps)

            # Simulación de eventos de nubosidad
            n_cloudy_days = int(days * 0.3)  # 30% de días nublados
            cloudy_days = np.random.choice(days, n_cloudy_days, replace=False)
            for day in cloudy_days:
                start_idx = day * int(24 / time_step)
                end_idx = start_idx + int(24 / time_step)
                if start_idx < timesteps:
                    cloud_factor = np.random.uniform(
                        0.3, 0.7
                    )  # Reducción por nubosidad
                    end_idx = min(end_idx, timesteps)
                    cloud_events[start_idx:end_idx] = 1 - cloud_factor

            solar_radiation = base_solar_radiation * (
                seasonal_component + daily_component - cloud_events + noise
            )
            solar_radiation = np.maximum(0, solar_radiation)  # No puede ser negativa

            # 2. Velocidad del viento (m/s)
            # Componentes: valor base + tendencia estacional + variación diaria + ruido
            seasonal_wind = 0.5 * np.sin(2 * np.pi * time_idx / (365 * 24 / time_step))
            daily_wind = 0.3 * np.sin(2 * np.pi * time_idx / (24 / time_step))
            wind_noise = np.random.normal(0, 0.3, timesteps)

            wind_speed = base_wind_speed + seasonal_wind + daily_wind + wind_noise
            wind_speed = np.maximum(0, wind_speed)  # No puede ser negativa

            # 3. Consumo energético (kWh)
            # Factores que afectan el consumo
            is_weekend = np.array(
                [
                    (
                        1
                        if (start_date + timedelta(hours=i * time_step)).weekday() >= 5
                        else 0
                    )
                    for i in range(timesteps)
                ]
            )
            is_school_hour = np.array(
                [
                    (
                        1
                        if 7 <= (start_date + timedelta(hours=i * time_step)).hour < 18
                        else 0
                    )
                    for i in range(timesteps)
                ]
            )

            # Consumo base
            base_consumption = 5 * size_factor  # kWh

            # Patrón semanal y diario
            weekday_factor = 1 - 0.8 * is_weekend  # 80% menos consumo en fin de semana
            hour_factor = 0.2 + 0.8 * is_school_hour  # Más consumo en horario escolar

            # Patrón estacional (menos consumo en vacaciones)
            # Vacaciones aproximadas: julio-septiembre, y última semana de diciembre
            month = np.array(
                [
                    (start_date + timedelta(hours=i * time_step)).month
                    for i in range(timesteps)
                ]
            )
            day = np.array(
                [
                    (start_date + timedelta(hours=i * time_step)).day
                    for i in range(timesteps)
                ]
            )
            is_vacation = ((month >= 7) & (month <= 9)) | ((month == 12) & (day >= 24))
            vacation_factor = 1 - 0.7 * is_vacation.astype(
                int
            )  # 70% menos consumo en vacaciones

            # Consumo total con variabilidad aleatoria
            consumption_noise = np.random.normal(0, 0.05, timesteps)
            energy_consumption = (
                base_consumption
                * size_factor
                * weekday_factor
                * hour_factor
                * vacation_factor
                * (1 + consumption_noise)
            )
            energy_consumption = np.maximum(
                0, energy_consumption
            )  # No puede ser negativo

            # 4. Temperatura (°C)
            seasonal_temp = 2 * np.sin(2 * np.pi * time_idx / (365 * 24 / time_step))
            daily_temp = 3 * np.sin(
                2 * np.pi * (time_idx % (24 / time_step)) / (24 / time_step) - 0.5
            )
            temp_noise = np.random.normal(0, 0.5, timesteps)

            temperature = base_temperature + seasonal_temp + daily_temp + temp_noise

            # Crear DataFrame con todos los datos
            school_data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "solar_radiation": solar_radiation,
                    "wind_speed": wind_speed,
                    "energy_consumption": energy_consumption,
                    "temperature": temperature,
                    "is_weekend": is_weekend,
                    "is_school_hour": is_school_hour,
                    "is_vacation": is_vacation.astype(int),
                }
            )

            # Parámetros específicos de la escuela
            school_params = {
                "size_factor": size_factor,
                "modern_factor": modern_factor,
                "location": {
                    "latitude": latitude + np.random.uniform(-0.05, 0.05),
                    "longitude": -78.65 + np.random.uniform(-0.05, 0.05),
                    "altitude": 2750 + np.random.uniform(-100, 100),
                },
                "solar_capacity": 5 * size_factor,  # kWp
                "wind_capacity": (
                    2 * size_factor if np.random.random() > 0.3 else 0
                ),  # kW (algunas escuelas sin eólica)
                "battery_capacity": 10 * size_factor,  # kWh
                "average_daily_consumption": np.mean(energy_consumption)
                * 24,  # kWh/día
            }

            self.data[school] = {"timeseries": school_data, "parameters": school_params}

        print(f"Datos sintéticos generados para {len(self.school_names)} escuelas.")
        return self.data

    def save_data_to_excel(self):
        """
        Guarda los datos generados en archivos Excel.
        """
        if not hasattr(self, "data"):
            print("Generando datos antes de guardar...")
            self.generate_synthetic_data()

        # Definir el nombre del archivo con la fecha actual
        current_date = datetime.now().strftime("%Y%m%d")
        excel_path = f"{self.output_dir}/datos_sinteticos_escuelas_{current_date}.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Guardar series temporales de cada escuela
            for school, data in self.data.items():
                # Guardar series temporales
                data["timeseries"].to_excel(writer, sheet_name=school[:31], index=False)

                # Guardar parámetros en otra hoja
                params_df = pd.DataFrame(
                    {
                        "Parámetro": [
                            "Tamaño relativo",
                            "Factor de modernidad",
                            "Latitud",
                            "Longitud",
                            "Altitud",
                            "Capacidad solar (kWp)",
                            "Capacidad eólica (kW)",
                            "Capacidad de batería (kWh)",
                            "Consumo diario promedio (kWh/día)",
                        ],
                        "Valor": [
                            data["parameters"]["size_factor"],
                            data["parameters"]["modern_factor"],
                            data["parameters"]["location"]["latitude"],
                            data["parameters"]["location"]["longitude"],
                            data["parameters"]["location"]["altitude"],
                            data["parameters"]["solar_capacity"],
                            data["parameters"]["wind_capacity"],
                            data["parameters"]["battery_capacity"],
                            data["parameters"]["average_daily_consumption"],
                        ],
                    }
                )
                params_df.to_excel(
                    writer, sheet_name=f"{school[:25]}_params", index=False
                )

            # Crear una hoja con resumen de todas las escuelas
            summary_data = {}
            for parameter in [
                "size_factor",
                "modern_factor",
                "solar_capacity",
                "wind_capacity",
                "battery_capacity",
                "average_daily_consumption",
            ]:
                summary_data[parameter] = [
                    self.data[school]["parameters"][parameter]
                    for school in self.school_names
                ]

            summary_df = pd.DataFrame(summary_data, index=self.school_names)
            summary_df.to_excel(writer, sheet_name="Resumen")

        print(f"Datos guardados en {excel_path}")
        return excel_path

    def plot_sample_data(self):
        """
        Genera gráficos de muestra para visualizar los datos generados.
        """
        if not hasattr(self, "data"):
            print("Generando datos antes de graficar...")
            self.generate_synthetic_data()

        # Crear directorio para gráficos
        plots_dir = f"{self.output_dir}/graficos"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # 1. Consumo energético por escuela (una semana de muestra)
        plt.figure(figsize=(12, 8))
        for school in self.school_names:
            # Tomar una semana de muestra (primera semana de febrero de 2023)
            sample_start = datetime(2023, 2, 1)
            sample_end = sample_start + timedelta(days=7)
            mask = (self.data[school]["timeseries"]["timestamp"] >= sample_start) & (
                self.data[school]["timeseries"]["timestamp"] <= sample_end
            )
            sample_data = self.data[school]["timeseries"][mask]

            # Convertir a arrays numpy antes de graficar para evitar problemas de indexación
            x = sample_data["timestamp"].to_numpy()
            y = sample_data["energy_consumption"].to_numpy()

            # Graficar consumo
            plt.plot(x, y, label=school)

        plt.title("Consumo Energético por Escuela")
        plt.xlabel("Fecha y Hora")
        plt.ylabel("Consumo (kWh)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/consumo_semanal.png", dpi=300)

        # 2. Radiación solar promedio diaria por mes
        plt.figure(figsize=(12, 8))
        monthly_solar = {}

        for school in self.school_names:
            # Agregar columna de mes
            data_copy = self.data[school]["timeseries"].copy()
            data_copy["month"] = data_copy["timestamp"].dt.month

            # Calcular promedio diario por mes
            monthly_avg = data_copy.groupby("month")["solar_radiation"].mean()
            monthly_solar[school] = monthly_avg

        # Crear DataFrame para graficar
        monthly_df = pd.DataFrame(monthly_solar)

        # Convertir a arrays numpy para graficar
        months = monthly_df.index.to_numpy()

        plt.figure(figsize=(12, 8))
        for school in self.school_names:
            plt.bar(
                months + (self.school_names.index(school) - 2) * 0.15,
                monthly_df[school].to_numpy(),
                width=0.15,
                label=school,
            )

        plt.title("Radiación Solar Promedio por Mes")
        plt.xlabel("Mes")
        plt.xticks(range(1, 13))
        plt.ylabel("Radiación Solar Promedio (kW/m²)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/radiacion_solar_mensual.png", dpi=300)

        # 3. Correlación entre variables para una escuela
        school = self.school_names[0]  # Primera escuela

        plt.figure(figsize=(10, 8))
        corr_vars = [
            "solar_radiation",
            "wind_speed",
            "energy_consumption",
            "temperature",
        ]
        corr_data = self.data[school]["timeseries"][corr_vars]

        # Calcular correlación
        corr_matrix = corr_data.corr()

        # Graficar heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlación entre Variables para {school}")
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/correlacion_variables.png", dpi=300)

        print(f"Gráficos guardados en {plots_dir}")


# Ejecutar el generador de datos
if __name__ == "__main__":
    # Configuración de la simulación
    np.random.seed(42)  # Para reproducibilidad
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2025, 4, 30)
    time_step = 1  # en horas

    print(
        f"Período de simulación: {start_date.strftime('%d-%b-%Y')} a {end_date.strftime('%d-%b-%Y')}"
    )
    print(f"Total de días: {(end_date - start_date).days + 1}")

    # Crear instancia del simulador
    simulator = RenewableEnergySimulator()

    # Generar datos sintéticos
    simulator.generate_synthetic_data(
        start_date=start_date, end_date=end_date, time_step=time_step
    )

    # Guardar datos en Excel
    excel_file = simulator.save_data_to_excel()
    print(f"Archivo Excel generado: {excel_file}")

    # Generar gráficos de muestra
    simulator.plot_sample_data()

    print("Proceso completado con éxito.")
