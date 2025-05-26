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


# app = Flask(__name__)

app = Flask(
    __name__,
    static_url_path="",
    static_folder="static",
    template_folder="templates",
)

socketio = SocketIO(app)


@app.route("/")
def portada():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("main.html")


@app.route("/gps")
def gps():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("gps.html")


@app.route("/consumo")
def consumo():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("consumo.html")


@app.route("/instrucciones")
def pagina2():
    return render_template("indicaciones.html")


@app.route("/presupuesto")
def presupuesto():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("presupuesto.html")


@app.route("/resultados")
def resultados():
    # Aquí puedes ajustar el contador o cualquier otra lógica necesaria
    return render_template("resultados.html")


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
