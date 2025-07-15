from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sys
import pyaudio
import json
from gtts import gTTS
from datetime import datetime
import pygame
import os
import threading
import time
from flask import Flask, render_template, redirect
from flask import Flask, render_template, request, redirect, session
import random
import pandas as pd
import logging
import calendar

# app = Flask(__name__)

app = Flask(
    __name__,
    static_url_path="",
    static_folder="static",
    template_folder="templates",
)


socketio = SocketIO(app)


strlatitud = "0"
strlongitud = "0"
altitud = "0"
areaTerreno = 0
areaTecho = 0
numEstudiantes = 0
consumoDiario = 0
consumoMensual = 0
presupuestoSolar = 0
presupuestoEolico = 0

# Configuración del archivo Excel
EXCEL_FILE = "calculadora_consumos.xlsx"


class ConsumosManager:
    def __init__(self):
        self.excel_file = EXCEL_FILE
        self.init_excel()

    def init_excel(self):
        """Inicializar archivo Excel si no existe"""
        if not os.path.exists(self.excel_file):
            # Datos iniciales de ejemplo
            data = {
                "Equipo": [
                    "Computadora Escritorio",
                    "Proyector",
                    "Lámpara LED Aula",
                    "Impresora Multifuncional",
                    "Pizarra Digital Interactiva",
                    "Equipo de Sonido",
                    "Laptops",
                ],
                "Cantidad": [1, 1, 8, 1, 1, 1, 1],
                "Horas_Dia": [6, 4, 8, 2, 4, 2, 8],
                "Consumo_Wh_dia": [1200, 1120, 1152, 100, 480, 160, 800],
                "Fecha_Registro": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 7,
            }
            df = pd.DataFrame(data)
            df.to_excel(self.excel_file, index=False)

    def get_all_equipos(self):
        """Obtener todos los equipos del Excel"""
        try:
            df = pd.read_excel(self.excel_file)
            return df.to_dict("records")
        except Exception as e:
            print(f"Error leyendo Excel: {e}")
            return []

    def add_equipo(self, equipo_data):
        """Agregar nuevo equipo al Excel"""
        try:
            df = pd.read_excel(self.excel_file)

            # Calcular consumo total
            consumo_total = (
                equipo_data["cantidad"]
                * equipo_data["horas_dia"]
                * equipo_data["potencia_watts"]
            )

            new_row = {
                "Equipo": equipo_data["nombre"],
                "Cantidad": equipo_data["cantidad"],
                "Horas_Dia": equipo_data["horas_dia"],
                "Consumo_Wh_dia": consumo_total,
                "Fecha_Registro": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_excel(self.excel_file, index=False)

            return True, "Equipo agregado exitosamente"
        except Exception as e:
            return False, f"Error agregando equipo: {str(e)}"

    def delete_equipo(self, equipo_nombre):
        """Eliminar equipo del Excel"""
        try:
            df = pd.read_excel(self.excel_file)

            # Filtrar el DataFrame para excluir el equipo
            df_filtered = df[df["Equipo"] != equipo_nombre]

            if len(df_filtered) == len(df):
                return False, "Equipo no encontrado"

            df_filtered.to_excel(self.excel_file, index=False)
            return True, "Equipo eliminado exitosamente"
        except Exception as e:
            return False, f"Error eliminando equipo: {str(e)}"

    def update_equipo(self, equipo_nombre, equipo_data):
        """Actualizar equipo existente"""
        try:
            df = pd.read_excel(self.excel_file)

            # Encontrar el índice del equipo
            mask = df["Equipo"] == equipo_nombre
            if not mask.any():
                return False, "Equipo no encontrado"

            # Calcular nuevo consumo
            consumo_total = (
                equipo_data["cantidad"]
                * equipo_data["horas_dia"]
                * equipo_data["potencia_watts"]
            )

            # Actualizar datos
            df.loc[mask, "Cantidad"] = equipo_data["cantidad"]
            df.loc[mask, "Horas_Dia"] = equipo_data["horas_dia"]
            df.loc[mask, "Consumo_Wh_dia"] = consumo_total
            df.loc[mask, "Fecha_Registro"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            df.to_excel(self.excel_file, index=False)
            return True, "Equipo actualizado exitosamente"
        except Exception as e:
            return False, f"Error actualizando equipo: {str(e)}"


# Instancia del manager
consumos_manager = ConsumosManager()


@app.route("/api/equipos", methods=["GET"])
def get_equipos():
    """Obtener todos los equipos"""
    equipos = consumos_manager.get_all_equipos()
    return jsonify(equipos)


@app.route("/api/equipos", methods=["POST"])
def add_equipo():
    """Agregar nuevo equipo"""
    data = request.get_json()

    # Validar datos requeridos
    required_fields = ["nombre", "cantidad", "horas_dia", "potencia_watts"]
    for field in required_fields:
        if field not in data:
            return (
                jsonify({"success": False, "message": f"Campo {field} requerido"}),
                400,
            )

    success, message = consumos_manager.add_equipo(data)

    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "message": message}), 500


@app.route("/api/equipos/<string:equipo_nombre>", methods=["DELETE"])
def delete_equipo(equipo_nombre):
    """Eliminar equipo"""
    success, message = consumos_manager.delete_equipo(equipo_nombre)

    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "message": message}), 500


@app.route("/api/equipos/<string:equipo_nombre>", methods=["PUT"])
def update_equipo(equipo_nombre):
    """Actualizar equipo"""
    data = request.get_json()

    success, message = consumos_manager.update_equipo(equipo_nombre, data)

    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "message": message}), 500


@app.route("/")
def portada():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("main.html")


@app.route("/instrucciones")
def pagina2():
    return render_template("indicaciones.html")


@app.route("/gps")
def gps():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("gps.html")


@app.route("/consumo")
def consumo():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("consumo.html")


@app.route("/presupuesto")
def presupuesto():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("presupuesto.html")


@app.route("/resultados")
def resultados():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("resultados.html")


@app.route("/get_consumo", methods=["POST"])
def get_consumo():
    global strlatitud, strlongitud, strlatitud, areaTerreno, areaTecho, numEstudiantes, consumoDiario, consumoMensual

    datos = request.get_json()

    consumoDiario = float(datos.get("consumo"))
    # Obtener el mes y año actual
    now = datetime.now()
    year = now.year
    month = now.month

    # Obtener el número de días del mes actual
    dias_mes = calendar.monthrange(year, month)[1]

    consumoMensual = consumoDiario * dias_mes

    print(
        "consumo diario: "
        + str(consumoDiario)
        + " Consumo mensual: "
        + str(consumoMensual)
    )

    return jsonify({"Diario": consumoDiario, "Mensual": consumoMensual})


@app.route("/get_area", methods=["POST"])
def get_area():
    global strlatitud, strlongitud, strlatitud, areaTerreno, areaTecho, numEstudiantes, consumoDiario, consumoMensual

    datos = request.get_json()

    areaTerreno = float(datos.get("area"))
    areaTecho = areaTerreno * random.randrange(3, 7) / 10
    numEstudiantes = (areaTecho / 2) * random.randrange(1, 3) / 100

    areaTecho = round(areaTecho)
    areaTerreno = round(areaTerreno)
    numEstudiantes = round(numEstudiantes)

    print(
        "TERRENO: "
        + str(areaTerreno)
        + " TECHO: "
        + str(areaTecho)
        + " ESTUDIANTES: "
        + str(numEstudiantes)
    )

    return jsonify(
        {"terreno": areaTerreno, "techo": areaTecho, "estudiantes": numEstudiantes}
    )


@app.route("/get_presupuesto", methods=["POST"])
def get_presupuesto():
    global strlatitud, strlongitud, strlatitud, areaTerreno, areaTecho, numEstudiantes, consumoDiario, consumoMensual

    datos = request.get_json()

    presupuestoSolar = float(datos.get("solar"))
    presupuestoEolico = float(datos.get("eolico"))

    print("SOLAR: " + str(presupuestoSolar) + " EOLICO: " + str(presupuestoEolico))

    return jsonify({"existo": True})


@app.route("/get_coordenadas", methods=["POST"])
def get_coordenadas():
    global strlatitud, strlongitud, strlatitud, altitud
    datos = request.get_json()
    strlatitud = round(float(datos.get("lat")), 6)
    strlongitud = round(float(datos.get("lng")), 6)
    altitud = datos.get("altitud")

    print(
        "LAT: "
        + str(strlatitud)
        + " LNG: "
        + str(strlongitud)
        + " ALTITUD: "
        + str(altitud)
    )

    return jsonify({"latitud": strlatitud, "longitud": strlongitud, "altitud": altitud})


@app.route("/get_resumen", methods=["POST"])
def get_resumen():

    global strlatitud, strlongitud, strlatitud, altitud, areaTerreno, areaTecho, numEstudiantes, consumoDiario, consumoMensual

    datos = request.get_json()

    print("Enviando variables resumen")

    variables = jsonify(
        {
            "lat": strlatitud,
            "lng": strlongitud,
            "altitud": altitud,
            "terreno": areaTerreno,
            "techo": areaTecho,
            "estudiantes": numEstudiantes,
            "diario": consumoDiario,
            "mensual": consumoMensual,
        }
    )
    print(str(variables))
    print(strlatitud)
    print(strlongitud)

    return variables


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
