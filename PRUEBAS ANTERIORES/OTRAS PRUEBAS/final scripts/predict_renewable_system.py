#!/usr/bin/env python3
"""
Script de despliegue para predicci�n de sistemas h�bridos de energ�a renovable
"""

import joblib
import pandas as pd
import numpy as np
from predict_renewable_system import HybridEnergySystemPredictor, EquipmentRecommender


def predict_system(school_data):
    """
    Predice el sistema �ptimo para una escuela

    Parameters:
    -----------
    school_data : dict
        Datos de la escuela con las siguientes claves:
        - latitud, longitud, altitud_m
        - area_disponible_techo_m2, area_disponible_terreno_m2
        - numero_estudiantes
        - consumo_mensual_kwh
        - presupuesto_usd (opcional)

    Returns:
    --------
    dict : Diccionario con el sistema recomendado y equipos
    """
    # Cargar modelo
    predictor = HybridEnergySystemPredictor()
    predictor.load_model("modelo_energia_renovable")

    # Realizar predicci�n
    resultado = predictor.predict_optimal_system(school_data)

    return resultado


if __name__ == "__main__":
    # Ejemplo de uso
    escuela = {
        "latitud": -1.65,
        "longitud": -78.65,
        "altitud_m": 3200,
        "area_disponible_techo_m2": 400,
        "area_disponible_terreno_m2": 800,
        "numero_estudiantes": 75,
        "consumo_mensual_kwh": 350,
        "presupuesto_usd": 40000,
    }

    resultado = predict_system(escuela)

    print(f"Sistema recomendado:")
    print(f"- Solar: {resultado['capacidad_solar_kwp']:.2f} kWp")
    print(f"- E�lica: {resultado['capacidad_eolica_kw']:.2f} kW")
    print(f"- Bater�as: {resultado['capacidad_bateria_kwh']:.2f} kWh")
    print(f"- Ahorro anual: ${resultado['ahorro_mensual_usd']*12:.2f}")
    print(
        f"- Costo total: ${resultado['equipos_recomendados']['resumen_costos']['total']:,.2f}"
    )
