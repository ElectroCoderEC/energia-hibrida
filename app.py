# Al inicio del archivo, después de los imports básicos, agrega:
import sys
print(sys.version)

import os
from flask import Flask, render_template, jsonify, request
#from flask_socketio import SocketIO, emit
import json
from datetime import datetime
import pygame
import os
from flask import Flask, render_template, request, redirect, session
import random
import calendar

import joblib
import pandas as pd
import numpy as np

import json


import pygame
import pyttsx3
import tempfile
import os
import threading
from pathlib import Path


# Detectar si estamos en producción
IS_PRODUCTION = os.environ.get('RENDER') is not None or os.environ.get('PORT') is not None

if not IS_PRODUCTION:
    try:
        import pygame
        import pyttsx3
        AUDIO_AVAILABLE = True
    except ImportError:
        AUDIO_AVAILABLE = False
        print("⚠️ Audio libraries not available")
else:
    AUDIO_AVAILABLE = False
    print("🌐 Running in production mode - Audio disabled")


'''
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
from gtts import gTTS
from datetime import datetime
import pygame
import os
import threading
from flask import Flask, render_template, redirect
from flask import Flask, render_template, request, redirect, session
import random
import pandas as pd
import calendar

import joblib
import pandas as pd
import numpy as np

import json


import pygame
import pyttsx3
import tempfile
import os
import threading
from pathlib import Path
'''



# app = Flask(__name__)

app = Flask(
    __name__,
    static_url_path="",
    static_folder="static",
    template_folder="templates",
)


#socketio = SocketIO(app)


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


# Reemplazar la clase TextToSpeechEngine con esta versión:
class TextToSpeechEngine:
    def __init__(self):
        """Inicializa el motor de texto a voz (solo en desarrollo)"""
        if AUDIO_AVAILABLE:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.engine = pyttsx3.init()
            self.temp_files = []
            self.setup_voice_settings()
        else:
            self.engine = None
            self.temp_files = []
            print("ℹ️ Audio engine disabled in production")

    def setup_voice_settings(self):
        """Configura los ajustes básicos de voz"""
        if not AUDIO_AVAILABLE or not self.engine:
            return
        
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 0.9)
        voices = self.engine.getProperty("voices")
        if voices:
            self.engine.setProperty("voice", voices[0].id)

    def load_and_play_audio(self, audio_file_path):
        """Carga y reproduce un archivo de audio (solo en desarrollo)"""
        if not AUDIO_AVAILABLE:
            print(f"ℹ️ Audio playback disabled: {audio_file_path}")
            return False
        
        try:
            if not os.path.exists(audio_file_path):
                print(f"Error: El archivo {audio_file_path} no existe")
                return False

            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            return True

        except Exception as e:
            print(f"Error al reproducir archivo {audio_file_path}: {e}")
            return False

    def cleanup(self):
        """Limpia archivos temporales"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        self.temp_files.clear()

    def __del__(self):
        """Destructor para limpiar recursos"""
        self.cleanup()

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


class RenewableSystemPredictor:
    def __init__(self, model_path="modelo_energia_renovable"):
        self.predictor = None
        self.load_model(model_path)

    def load_model(self, path):
        """Cargar modelo entrenado"""
        try:
            self.best_model = joblib.load(f"{path}/best_model.pkl")
            self.scaler_X = joblib.load(f"{path}/scaler_X.pkl")
            self.scaler_y = joblib.load(f"{path}/scaler_y.pkl")
            self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")

            metadata = joblib.load(f"{path}/metadata.pkl")
            self.feature_names = metadata["feature_names"]
            self.target_names = metadata["target_names"]

            print(f"✅ Modelo cargado exitosamente")

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            print(f"🔄 Usando valores por defecto simulados")
            self.best_model = None

    def predict(self, location_data):
        """Predecir sistema óptimo"""
        if self.best_model is None:
            # Valores simulados si no hay modelo
            return self._predict_simulated(location_data)

        try:
            # Preparar entrada
            defaults = {
                "mes": 6,
                "hora": 12,
                "es_fin_semana": 0,
                "es_hora_pico": 1,
                "es_epoca_seca": 1,
                "radiacion_solar_kwh_m2": 4.5,
                "velocidad_viento_m_s": 5.0,
                "temperatura_c": 15.0,
                "consumo_kwh": location_data.get("consumo_mensual_kwh", 400)
                / (30 * 24),
            }

            input_dict = {**defaults, **location_data}
            input_data = pd.DataFrame([input_dict])[self.feature_names]

            # Rellenar faltantes
            for col in input_data.columns:
                if input_data[col].isna().any():
                    input_data[col] = input_data[col].fillna(defaults.get(col, 0))

            # Predecir
            X_scaled = self.scaler_X.transform(input_data)
            y_pred_scaled = self.best_model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(1, -1))

            # Resultado
            prediction = {}
            for i, target_name in enumerate(self.target_names):
                prediction[target_name.replace("target_", "")] = float(y_pred[0][i])

            return prediction

        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return self._predict_simulated(location_data)

    def _predict_simulated(self, location_data):
        """Predicción simulada para cuando no hay modelo"""
        # Simulación basada en los datos de entrada
        consumo = location_data.get("consumo_mensual_kwh", 400)
        area = location_data.get("area_disponible_m2", 300)
        altitude = location_data.get("altitude_m", 3000)

        # Cálculos básicos simulados
        cap_solar = min(area * 0.15, consumo * 0.15)  # 15% del área o 15% del consumo
        cap_eolica = min(area * 0.05, consumo * 0.08)  # 5% del área o 8% del consumo
        cap_bateria = (cap_solar + cap_eolica) * 2  # 2 veces la capacidad de generación

        # Ajuste por altitud
        factor_altitud = 1 + (altitude - 3000) * 0.0001
        cap_solar *= factor_altitud
        cap_eolica *= factor_altitud

        return {
            "capacidad_solar": max(5, cap_solar),
            "capacidad_eolica": max(2, cap_eolica),
            "capacidad_bateria": max(10, cap_bateria),
            "costo_total": (cap_solar * 1200 + cap_eolica * 2500 + cap_bateria * 800),
            "autosuficiencia": min(95, 60 + (cap_solar + cap_eolica) * 2),
            "roi": max(4, 50000 / (consumo * 0.092 * 12)),
        }


# Inicializar predictor global
predictor = RenewableSystemPredictor()

# Instancia del manager
consumos_manager = ConsumosManager()
tts = TextToSpeechEngine()


def calcular_equipos_especificos(prediction):
    """Calcular equipos específicos basado en predicciones"""
    cap_solar = prediction.get("capacidad_solar", 0)
    cap_eolica = prediction.get("capacidad_eolica", 0)
    cap_bateria = prediction.get("capacidad_bateria", 0)

    equipos = {
        "solar": {"capacidad_kwp": cap_solar, "equipos": []},
        "eolico": {"capacidad_kw": cap_eolica, "equipos": []},
        "baterias": {"capacidad_kwh": cap_bateria, "equipos": []},
        "accesorios": {"equipos": []},
    }

    # Equipos solares
    if cap_solar > 0:
        num_paneles_450w = int(cap_solar * 1000 / 450) + 1
        precio_paneles = num_paneles_450w * 180

        equipos["solar"]["equipos"] = [
            {
                "categoria": "Paneles Solares",
                "items": [
                    {
                        "nombre": "Panel Monocristalino 450W",
                        "cantidad": num_paneles_450w,
                        "precio_unitario": 180,
                        "precio_total": precio_paneles,
                    },
                    {
                        "nombre": "Estructura de Montaje",
                        "cantidad": int(cap_solar),
                        "precio_unitario": 150,
                        "precio_total": int(cap_solar * 150),
                    },
                ],
            },
            {
                "categoria": "Equipos de Control",
                "items": [
                    {
                        "nombre": "Inversor Híbrido 5kW",
                        "cantidad": int(cap_solar / 5) + 1,
                        "precio_unitario": 980,
                        "precio_total": (int(cap_solar / 5) + 1) * 980,
                    },
                    {
                        "nombre": "Controlador MPPT 60A",
                        "cantidad": int(cap_solar / 3) + 1,
                        "precio_unitario": 280,
                        "precio_total": (int(cap_solar / 3) + 1) * 280,
                    },
                ],
            },
        ]

    # Equipos eólicos
    if cap_eolica > 0:
        num_turbinas = int(cap_eolica / 2) + 1
        equipos["eolico"]["equipos"] = [
            {
                "categoria": "Aerogeneradores",
                "items": [
                    {
                        "nombre": "Aerogenerador Horizontal 2kW",
                        "cantidad": num_turbinas,
                        "precio_unitario": 4200,
                        "precio_total": num_turbinas * 4200,
                    },
                    {
                        "nombre": "Torre 12m Galvanizada",
                        "cantidad": num_turbinas,
                        "precio_unitario": 2800,
                        "precio_total": num_turbinas * 2800,
                    },
                ],
            },
            {
                "categoria": "Sistema de Control",
                "items": [
                    {
                        "nombre": "Controlador Eólico 60A",
                        "cantidad": num_turbinas,
                        "precio_unitario": 320,
                        "precio_total": num_turbinas * 320,
                    }
                ],
            },
        ]

    # Baterías
    if cap_bateria > 0:
        num_baterias = int(cap_bateria / 2.4) + 1
        equipos["baterias"]["equipos"] = [
            {
                "categoria": "Sistema de Almacenamiento",
                "items": [
                    {
                        "nombre": "Batería LiFePO4 200Ah 12V",
                        "cantidad": num_baterias,
                        "precio_unitario": 720,
                        "precio_total": num_baterias * 720,
                    },
                    {
                        "nombre": "BMS Sistema de Gestión",
                        "cantidad": int(num_baterias / 3) + 1,
                        "precio_unitario": 400,
                        "precio_total": (int(num_baterias / 3) + 1) * 400,
                    },
                ],
            }
        ]

    # Accesorios
    potencia_total = cap_solar + cap_eolica
    equipos["accesorios"]["equipos"] = [
        {
            "categoria": "Instalación y Accesorios",
            "items": [
                {
                    "nombre": "Cableado DC y AC",
                    "cantidad": 1,
                    "precio_unitario": int(potencia_total * 100),
                    "precio_total": int(potencia_total * 100),
                },
                {
                    "nombre": "Gabinetes de Protección",
                    "cantidad": 2,
                    "precio_unitario": 250,
                    "precio_total": 500,
                },
                {
                    "nombre": "Sistema de Monitoreo",
                    "cantidad": 1,
                    "precio_unitario": 800,
                    "precio_total": 800,
                },
                {
                    "nombre": "Instalación y Puesta en Marcha",
                    "cantidad": 1,
                    "precio_unitario": int(potencia_total * 200),
                    "precio_total": int(potencia_total * 200),
                },
            ],
        }
    ]

    return equipos


def calcular_metricas_economicas(prediction, equipos):
    """Calcular métricas económicas y ambientales"""
    cap_solar = prediction.get("capacidad_solar", 0)
    cap_eolica = prediction.get("capacidad_eolica", 0)

    # Calcular costo total
    costo_total = 0
    for sistema in equipos.values():
        for categoria in sistema.get("equipos", []):
            for item in categoria.get("items", []):
                costo_total += item.get("precio_total", 0)

    # Producción anual estimada
    produccion_solar_anual = cap_solar * 1500  # kWh/año (factor típico Ecuador)
    produccion_eolica_anual = cap_eolica * 2200  # kWh/año (factor típico altitud)
    produccion_total_anual = produccion_solar_anual + produccion_eolica_anual

    # Ahorros económicos
    precio_kwh = 0.092  # USD/kWh Ecuador
    ahorro_solar_anual = produccion_solar_anual * precio_kwh
    ahorro_eolica_anual = produccion_eolica_anual * precio_kwh
    ahorro_total_anual = ahorro_solar_anual + ahorro_eolica_anual

    # Impacto ambiental
    factor_co2 = 0.385  # kg CO2/kWh
    co2_solar_anual = produccion_solar_anual * factor_co2 / 1000  # toneladas
    co2_eolica_anual = produccion_eolica_anual * factor_co2 / 1000  # toneladas
    co2_total_anual = co2_solar_anual + co2_eolica_anual

    # Generación mensual simulada
    generacion_solar_mensual = [
        int(produccion_solar_anual * factor / 12)
        for factor in [
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.12,
            1.08,
            1.02,
            0.95,
            0.88,
        ]
    ]

    generacion_eolica_mensual = [
        int(produccion_eolica_anual * factor / 12)
        for factor in [
            1.10,
            1.05,
            1.00,
            0.95,
            0.90,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
        ]
    ]

    return {
        "costo_total": costo_total,
        "produccion_solar_anual": produccion_solar_anual,
        "produccion_eolica_anual": produccion_eolica_anual,
        "produccion_total_anual": produccion_total_anual,
        "ahorro_solar_anual": ahorro_solar_anual,
        "ahorro_eolica_anual": ahorro_eolica_anual,
        "ahorro_total_anual": ahorro_total_anual,
        "co2_solar_anual": co2_solar_anual,
        "co2_eolica_anual": co2_eolica_anual,
        "co2_total_anual": co2_total_anual,
        "generacion_solar_mensual": generacion_solar_mensual,
        "generacion_eolica_mensual": generacion_eolica_mensual,
        "roi_años": costo_total / ahorro_total_anual if ahorro_total_anual > 0 else 50,
    }


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
    """

    # Mostrar voces disponibles
    voices = tts.get_available_voices()
    print(f"\nVoces disponibles ({len(voices)}):")
    for voice in voices:
        print(f"  {voice['id']}: {voice['name']} ({voice['gender']})")

    # Guardar audio en archivo
    print("\nGuardando audio en archivo...")
    audio_file = "audios/ubicacion.wav"
    tts.play_text(
        "Coloca la ubicación de la institución en el mapa y su área aproximada",
        save_file=True,
        filename=audio_file,
    )
    print(f"Audio guardado en: {audio_file}")

    # Limpiar recursos
    tts.cleanup()
    print("\nDemo completada.")
    """
    # Cargar archivo existente:
    tts.load_and_play_audio("audios/bienvenida.wav")

    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("main.html")


@app.route("/instrucciones")
def pagina2():
    # Cargar archivo existente:
    tts.load_and_play_audio("audios/instrucciones.wav")
    return render_template("indicaciones.html")


@app.route("/gps")
def gps():
    tts.load_and_play_audio("audios/ubicacion.wav")
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("gps.html")


@app.route("/consumo")
def consumo():
    tts.load_and_play_audio("audios/consumo.wav")
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("consumo.html")


@app.route("/presupuesto")
def presupuesto():
    tts.load_and_play_audio("audios/presupuesto.wav")
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("presupuesto.html")


@app.route("/resultados")
def resultados():

    tts.load_and_play_audio("audios/prediccion.wav")

    global presupuestoSolar, presupuestoEolico, strlatitud, strlongitud, strlatitud, altitud, areaTerreno, areaTecho, numEstudiantes, consumoDiario, consumoMensual

    presupuestoT = presupuestoSolar + presupuestoEolico

    # Obtener datos de entrada de la sesión
    datos_entrada = session.get(
        "datos_entrada",
        {
            "latitud": strlatitud,
            "longitud": strlongitud,
            "altitude_m": altitud,
            "area_disponible_m2": areaTerreno,
            "consumo_mensual_kwh": consumoMensual,
            "presupuesto_usd": presupuestoT,
            "tipo_sistema": "hibrido",
        },
    )

    try:
        # Realizar predicción con ML
        print("🔮 Realizando predicción con ML...")
        prediction = predictor.predict(datos_entrada)

        # Calcular equipos específicos
        equipos = calcular_equipos_especificos(prediction)

        # Calcular métricas económicas
        metricas = calcular_metricas_economicas(prediction, equipos)

        # Preparar estructura de datos para el frontend
        sistema_calculado = {
            "solar": {
                "capacidad_kwp": round(prediction.get("capacidad_solar", 0), 1),
                "energia_anual_kwh": int(metricas["produccion_solar_anual"]),
                "ahorro_anual_usd": round(metricas["ahorro_solar_anual"], 2),
                "co2_evitado_ton": round(metricas["co2_solar_anual"], 1),
                "generacion_mensual": metricas["generacion_solar_mensual"],
                "equipos": equipos["solar"]["equipos"],
            },
            "eolico": {
                "capacidad_kw": round(prediction.get("capacidad_eolica", 0), 1),
                "energia_anual_kwh": int(metricas["produccion_eolica_anual"]),
                "ahorro_anual_usd": round(metricas["ahorro_eolica_anual"], 2),
                "co2_evitado_ton": round(metricas["co2_eolica_anual"], 1),
                "generacion_mensual": metricas["generacion_eolica_mensual"],
                "equipos": equipos["eolico"]["equipos"],
            },
            "baterias": {
                "capacidad_kwh": round(prediction.get("capacidad_bateria", 0), 1),
                "equipos": equipos["baterias"]["equipos"],
            },
            "accesorios": {"equipos": equipos["accesorios"]["equipos"]},
            "resumen": {
                "costo_total": metricas["costo_total"],
                "ahorro_total_anual": metricas["ahorro_total_anual"],
                "co2_total_anual": metricas["co2_total_anual"],
                "roi_años": round(metricas["roi_años"], 1),
                "autosuficiencia_pct": round(prediction.get("autosuficiencia", 0), 1),
                "capacidad_total_kw": round(
                    prediction.get("capacidad_solar", 0)
                    + prediction.get("capacidad_eolica", 0),
                    1,
                ),
                "produccion_total_anual": int(metricas["produccion_total_anual"]),
            },
        }

        # Convertir a JSON para JavaScript
        sistema_json = json.dumps(sistema_calculado, ensure_ascii=False)

        print("✅ Predicción completada exitosamente")
        print(f"📊 Solar: {sistema_calculado['solar']['capacidad_kwp']} kWp")
        print(f"💨 Eólica: {sistema_calculado['eolico']['capacidad_kw']} kW")
        print(f"🔋 Baterías: {sistema_calculado['baterias']['capacidad_kwh']} kWh")
        print(f"💰 Costo total: ${sistema_calculado['resumen']['costo_total']:,.2f}")
        print(f"⏱️ ROI: {sistema_calculado['resumen']['roi_años']} años")

        return render_template(
            "resultados.html",
            sistema_calculado=sistema_json,
            datos_entrada=datos_entrada,
        )

    except Exception as e:
        print(f"❌ Error en predicción: {e}")

        # Datos de fallback en caso de error
        sistema_fallback = {
            "solar": {
                "capacidad_kwp": 15.0,
                "energia_anual_kwh": 22500,
                "ahorro_anual_usd": 2070.0,
                "co2_evitado_ton": 8.7,
                "generacion_mensual": [
                    1800,
                    1900,
                    2000,
                    2100,
                    2200,
                    2300,
                    2400,
                    2300,
                    2200,
                    2100,
                    1900,
                    1800,
                ],
                "equipos": [],
            },
            "eolico": {
                "capacidad_kw": 6.0,
                "energia_anual_kwh": 13200,
                "ahorro_anual_usd": 1214.4,
                "co2_evitado_ton": 5.1,
                "generacion_mensual": [
                    1200,
                    1100,
                    1000,
                    900,
                    800,
                    700,
                    800,
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                ],
                "equipos": [],
            },
            "baterias": {"capacidad_kwh": 42.0, "equipos": []},
            "accesorios": {"equipos": []},
            "resumen": {
                "costo_total": 45000,
                "ahorro_total_anual": 3284.4,
                "co2_total_anual": 13.8,
                "roi_años": 13.7,
                "autosuficiencia_pct": 78.5,
                "capacidad_total_kw": 21.0,
                "produccion_total_anual": 35700,
            },
        }

        return render_template(
            "resultados.html",
            sistema_calculado=json.dumps(sistema_fallback, ensure_ascii=False),
            datos_entrada=datos_entrada,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint para predicciones desde JavaScript"""
    try:
        datos = request.get_json()
        prediction = predictor.predict(datos)
        equipos = calcular_equipos_especificos(prediction)
        metricas = calcular_metricas_economicas(prediction, equipos)

        resultado = {
            "success": True,
            "prediction": prediction,
            "equipos": equipos,
            "metricas": metricas,
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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
    global strlatitud, strlongitud, altitud, areaTerreno, areaTecho, numEstudiantes, consumoDiario, consumoMensual
    
    datos = request.get_json()

    print("Enviando variables resumen")
    
    return jsonify({
        "lat": strlatitud,
        "lng": strlongitud,
        "altitud": altitud,
        "terreno": areaTerreno,
        "techo": areaTecho,
        "estudiantes": numEstudiantes,
        "diario": consumoDiario,
        "mensual": consumoMensual,
    })
     

    #print(str(variables))
    

# DESPUÉS:
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)