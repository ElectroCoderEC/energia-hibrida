"""
Script para ejecutar el generador de datos sintéticos para redes híbridas de energía.
Este script importa la clase EnergyDataGenerator y ejecuta el proceso completo de generación.
"""

from synthetic_data_generator import EnergyDataGenerator
import time


def main():
    """Función principal para ejecutar el generador de datos"""

    print("=== GENERADOR DE DATOS SINTÉTICOS PARA REDES HÍBRIDAS DE ENERGÍA ===")
    print(
        "Este script generará datos desde el 1 de enero de 2021 hasta el 30 de abril de 2025"
    )
    print("para 5 instituciones educativas en el cantón Riobamba.\n")

    start_time = time.time()

    # Crear instancia del generador
    generator = EnergyDataGenerator(start_date="2021-01-01", end_date="2025-04-30")

    # Generar datos
    print("\nGenerando conjunto de datos completo...")
    data = generator.generate_complete_dataset()

    # Visualizar datos
    print("\nCreando visualizaciones estadísticas...")
    generator.visualize_data(data)

    # Exportar a Excel
    print("\nExportando datos a Excel...")
    generator.export_to_excel(data, "datos_energeticos.xlsx")

    # Información de resumen
    elapsed_time = time.time() - start_time
    print(f"\nProceso completado en {elapsed_time:.2f} segundos")
    print(f"Total de registros generados: {len(data)}")
    print(
        f"Período cubierto: {data['Fecha'].min().strftime('%Y-%m-%d')} hasta {data['Fecha'].max().strftime('%Y-%m-%d')}"
    )

    # Estadísticas básicas
    print("\nEstadísticas básicas por institución:")
    for escuela in generator.school_names:
        school_data = data[data["Institucion"] == escuela]
        total_consumption = school_data["Consumo_kWh"].sum()
        total_generation = (
            school_data["Generacion_Solar_kWh"].sum()
            + school_data["Generacion_Eolica_kWh"].sum()
        )
        autosuficiencia = school_data["Autosuficiencia_Pct"].mean()
        ahorro_total = school_data["Ahorro_USD"].sum()

        print(f"\n{escuela}:")
        print(f"  - Consumo total: {total_consumption:.2f} kWh")
        print(f"  - Generación renovable total: {total_generation:.2f} kWh")
        print(f"  - Autosuficiencia promedio: {autosuficiencia:.2f}%")
        print(f"  - Ahorro total: ${ahorro_total:.2f}")

    print(
        "\nRevise el archivo Excel 'datos_energeticos.xlsx' y las visualizaciones en la carpeta 'visualizaciones'"
    )


if __name__ == "__main__":
    main()
