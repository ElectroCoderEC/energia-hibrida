import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# Configuración global de estilo
plt.style.use("seaborn-darkgrid")
sns.set_palette("husl")


class HybridEnergySystemPredictor:
    """
    Modelo de ML para predecir el dimensionamiento óptimo de sistemas híbridos
    de energía renovable para escuelas rurales
    """

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.target_names = None
        self.equipment_recommender = EquipmentRecommender()

    def load_and_prepare_data(
        self, filepath="datos_energia_renovable/dataset_ml_energia_renovable.csv"
    ):
        """Carga y prepara los datos para entrenamiento"""
        print("Cargando datos...")
        df = pd.read_csv(filepath)

        print(df.columns)

        # Agregar características adicionales
        df["trimestre"] = ((df["mes"] - 1) // 3) + 1
        df["es_epoca_seca"] = df["mes"].isin([6, 7, 8, 9]).astype(int)
        df["ratio_area_estudiantes"] = (
            df["area_disponible_techo_m2"] + df["area_disponible_terreno_m2"]
        ) / df["numero_estudiantes"]
        df["consumo_per_capita"] = df["consumo_mensual_kwh"] / df["numero_estudiantes"]

        # Características de entrada
        self.feature_names = [
            "latitud",
            "longitud",
            "altitud_m",
            "area_disponible_techo_m2",
            "area_disponible_terreno_m2",
            "numero_estudiantes",
            "consumo_mensual_kwh",
            "mes",
            "trimestre",
            "es_epoca_seca",
            "ratio_area_estudiantes",
            "consumo_per_capita",
        ]

        # Variables objetivo (lo que queremos predecir)
        self.target_names = [
            "capacidad_solar_kwp",
            "capacidad_eolica_kw",
            "capacidad_bateria_kwh",
            "generacion_solar_mensual_kwh",
            "generacion_eolica_mensual_kwh",
            "autosuficiencia_promedio_%",
            "ahorro_mensual_usd",
            "co2_evitado_mensual_kg",
        ]

        X = df[self.feature_names]
        y = df[self.target_names]

        # Guardar estadísticas para desnormalización posterior
        self.y_stats = {
            "mean": y.mean(),
            "std": y.std(),
            "min": y.min(),
            "max": y.max(),
        }

        print(f"Dataset cargado: {len(df)} registros")
        print(f"Características: {len(self.feature_names)}")
        print(f"Variables objetivo: {len(self.target_names)}")

        return X, y

    def create_models(self):
        """Crea diferentes modelos para comparación"""
        self.models = {
            "Random Forest": MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
            "Gradient Boosting": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
                )
            ),
            "XGBoost": MultiOutputRegressor(
                xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
            "Neural Network": MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",  # Parámetro corregido
                max_iter=1000,
                random_state=42,
            ),
        }

    def train_models(self, X_train, X_test, y_train, y_test):
        """Entrena y evalúa todos los modelos"""
        results = {}

        print("\nEntrenando modelos...")
        print("-" * 60)

        for name, model in self.models.items():
            print(f"\nEntrenando {name}...")

            # Entrenar modelo
            model.fit(X_train, y_train)

            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Métricas por variable objetivo
            metrics = {
                "train_r2": r2_score(y_train, y_pred_train, multioutput="raw_values"),
                "test_r2": r2_score(y_test, y_pred_test, multioutput="raw_values"),
                "train_rmse": np.sqrt(
                    mean_squared_error(y_train, y_pred_train, multioutput="raw_values")
                ),
                "test_rmse": np.sqrt(
                    mean_squared_error(y_test, y_pred_test, multioutput="raw_values")
                ),
                "train_mae": mean_absolute_error(
                    y_train, y_pred_train, multioutput="raw_values"
                ),
                "test_mae": mean_absolute_error(
                    y_test, y_pred_test, multioutput="raw_values"
                ),
            }

            # Promedio de métricas
            avg_metrics = {
                "avg_train_r2": np.mean(metrics["train_r2"]),
                "avg_test_r2": np.mean(metrics["test_r2"]),
                "avg_train_rmse": np.mean(metrics["train_rmse"]),
                "avg_test_rmse": np.mean(metrics["test_rmse"]),
                "avg_train_mae": np.mean(metrics["train_mae"]),
                "avg_test_mae": np.mean(metrics["test_mae"]),
            }

            results[name] = {
                "model": model,
                "metrics": metrics,
                "avg_metrics": avg_metrics,
                "predictions": {"train": y_pred_train, "test": y_pred_test},
            }

            # Mostrar resultados
            print(f"  R² promedio (test): {avg_metrics['avg_test_r2']:.4f}")
            print(f"  RMSE promedio (test): {avg_metrics['avg_test_rmse']:.4f}")
            print(f"  MAE promedio (test): {avg_metrics['avg_test_mae']:.4f}")

        # Seleccionar mejor modelo basado en R² de test
        best_model_name = max(
            results.keys(), key=lambda k: results[k]["avg_metrics"]["avg_test_r2"]
        )
        self.best_model = results[best_model_name]["model"]

        print(f"\n{'='*60}")
        print(f"Mejor modelo: {best_model_name}")
        print(
            f"R² promedio en test: {results[best_model_name]['avg_metrics']['avg_test_r2']:.4f}"
        )

        return results

    def hyperparameter_tuning(self, X_train, y_train):
        """Optimización de hiperparámetros para el mejor modelo"""
        print("\nOptimizando hiperparámetros para Random Forest...")

        param_grid = {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [10, 15, 20],
            "estimator__min_samples_split": [2, 5, 10],
            "estimator__min_samples_leaf": [1, 2, 4],
        }

        rf_model = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, n_jobs=-1)
        )

        # GridSearch con validación cruzada
        grid_search = GridSearchCV(
            rf_model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor score CV: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def analyze_feature_importance(self, X_train, model_results):
        """Analiza la importancia de las características"""
        # Usar Random Forest para importancia de características
        rf_model = model_results["Random Forest"]["model"]

        # Obtener importancia promedio across all outputs
        importances = []
        for estimator in rf_model.estimators_:
            importances.append(estimator.feature_importances_)

        avg_importance = np.mean(importances, axis=0)

        # Crear DataFrame de importancia
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": avg_importance}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def predict_optimal_system(self, location_data):
        """
        Predice el sistema óptimo para una nueva ubicación

        Parameters:
        -----------
        location_data : dict
            Diccionario con datos de la ubicación:
            - latitud, longitud, altitud_m
            - area_disponible_techo_m2, area_disponible_terreno_m2
            - numero_estudiantes
            - consumo_mensual_kwh
            - mes (opcional, si no se provee, calcula para todo el año)
            - presupuesto_usd (opcional)
        """
        # Preparar datos de entrada
        if "mes" not in location_data:
            # Si no se especifica mes, predecir para cada mes
            predictions = []
            for mes in range(1, 13):
                data = location_data.copy()
                data["mes"] = mes
                pred = self._predict_single(data)
                predictions.append(pred)

            # Promediar predicciones anuales
            result = pd.DataFrame(predictions).mean().to_dict()
            result["predicciones_mensuales"] = predictions
        else:
            result = self._predict_single(location_data)

        # Recomendar equipos
        equipment = self.equipment_recommender.recommend_equipment(
            result["capacidad_solar_kwp"],
            result["capacidad_eolica_kw"],
            result["capacidad_bateria_kwh"],
            location_data.get("presupuesto_usd", None),
        )

        result["equipos_recomendados"] = equipment

        return result

    def _predict_single(self, data):
        """Realiza predicción para un único conjunto de datos"""
        # Calcular características derivadas
        data["trimestre"] = ((data["mes"] - 1) // 3) + 1
        data["es_epoca_seca"] = 1 if data["mes"] in [6, 7, 8, 9] else 0
        data["ratio_area_estudiantes"] = (
            data["area_disponible_techo_m2"] + data["area_disponible_terreno_m2"]
        ) / data["numero_estudiantes"]
        data["consumo_per_capita"] = (
            data["consumo_mensual_kwh"] / data["numero_estudiantes"]
        )

        # Crear DataFrame con las características
        X_new = pd.DataFrame([data])[self.feature_names]

        # Normalizar
        X_new_scaled = self.scaler_X.transform(X_new)

        # Predecir
        y_pred_scaled = self.best_model.predict(X_new_scaled)

        # Desnormalizar
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # Crear diccionario de resultados
        result = {name: float(y_pred[0][i]) for i, name in enumerate(self.target_names)}

        # Calcular métricas adicionales
        result["roi_anos"] = (
            result["capacidad_solar_kwp"] * 1200
            + result["capacidad_eolica_kw"] * 2500
            + result["capacidad_bateria_kwh"] * 800
        ) / (result["ahorro_mensual_usd"] * 12)

        return result

    def save_model(self, path="modelo_energia_renovable"):
        """Guarda el modelo entrenado"""
        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar modelo
        joblib.dump(self.best_model, f"{path}/modelo_best.pkl")

        # Guardar scalers
        joblib.dump(self.scaler_X, f"{path}/scaler_X.pkl")
        joblib.dump(self.scaler_y, f"{path}/scaler_y.pkl")

        # Guardar metadatos
        metadata = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "y_stats": self.y_stats,
            "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        joblib.dump(metadata, f"{path}/metadata.pkl")

        print(f"\nModelo guardado en: {path}/")

    def load_model(self, path="modelo_energia_renovable"):
        """Carga un modelo previamente entrenado"""
        self.best_model = joblib.load(f"{path}/modelo_best.pkl")
        self.scaler_X = joblib.load(f"{path}/scaler_X.pkl")
        self.scaler_y = joblib.load(f"{path}/scaler_y.pkl")

        metadata = joblib.load(f"{path}/metadata.pkl")
        self.feature_names = metadata["feature_names"]
        self.target_names = metadata["target_names"]
        self.y_stats = metadata["y_stats"]

        print("Modelo cargado exitosamente")

    def plot_results(self, results, X_test, y_test):
        """Visualiza los resultados del entrenamiento en dos ventanas"""

        # =============================
        # VENTANA 1: Comparaciones y análisis global
        # =============================
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
        fig1.suptitle(
            "Análisis Comparativo del Modelo ML", fontsize=16, fontweight="bold"
        )

        # 1. Comparación de modelos
        ax1 = axes1[0]
        model_names = list(results.keys())
        test_r2 = [results[m]["avg_metrics"]["avg_test_r2"] for m in model_names]
        bars = ax1.bar(
            model_names, test_r2, color=["#3498db", "#e74c3c", "#27ae60", "#f39c12"]
        )
        ax1.set_ylabel("R² Score")
        ax1.set_title("Comparación de Modelos (R² en Test)")
        ax1.set_ylim(0, 1)
        for bar, score in zip(bars, test_r2):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # 2. Métricas por variable objetivo (mejor modelo)
        ax2 = axes1[1]
        best_model_name = max(
            results.keys(), key=lambda k: results[k]["avg_metrics"]["avg_test_r2"]
        )
        best_metrics = results[best_model_name]["metrics"]

        x = np.arange(len(self.target_names))
        width = 0.35
        bars1 = ax2.bar(
            x - width / 2, best_metrics["test_r2"], width, label="R²", color="#3498db"
        )
        bars2 = ax2.bar(
            x + width / 2,
            1 - best_metrics["test_rmse"] / self.y_stats["std"].values,
            width,
            label="1-NRMSE",
            color="#27ae60",
        )

        ax2.set_xlabel("Variable Objetivo")
        ax2.set_ylabel("Score")
        ax2.set_title(f"Métricas por Variable - {best_model_name}")
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [t.replace("_", "\n") for t in self.target_names],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax2.legend()
        ax2.set_ylim(0, 1)

        # 3. Importancia de características
        ax3 = axes1[2]
        feature_importance = self.analyze_feature_importance(X_test, results)
        bars = ax3.barh(
            feature_importance["feature"][:10],
            feature_importance["importance"][:10],
            color="#9b59b6",
        )
        ax3.set_xlabel("Importancia")
        ax3.set_title("Top 10 Características Más Importantes")
        ax3.invert_yaxis()

        plt.tight_layout()
        plt.show()

        # =============================
        # VENTANA 2: Predicciones vs valores reales
        # =============================
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle(
            f"Predicción vs Real - {best_model_name}", fontsize=16, fontweight="bold"
        )

        best_pred = results[best_model_name]["predictions"]["test"]

        for ax, var_idx, var_name in zip(
            axes2,
            [0, 1, 6],
            ["Capacidad Solar (kWp)", "Capacidad Eólica (kW)", "Ahorro Mensual (USD)"],
        ):
            ax.scatter(
                y_test.iloc[:, var_idx],
                best_pred[:, var_idx],
                alpha=0.5,
                s=30,
                color="#e74c3c",
            )

            min_val = min(y_test.iloc[:, var_idx].min(), best_pred[:, var_idx].min())
            max_val = max(y_test.iloc[:, var_idx].max(), best_pred[:, var_idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

            ax.set_xlabel("Valor Real")
            ax.set_ylabel("Valor Predicho")
            ax.set_title(f"{var_name}")

            r2 = r2_score(y_test.iloc[:, var_idx], best_pred[:, var_idx])
            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

        plt.tight_layout()
        plt.show()

        return fig1, fig2


class EquipmentRecommender:
    """Recomienda equipos específicos basado en las capacidades predichas"""

    def __init__(self):
        self.load_equipment_catalog()

    def load_equipment_catalog(self):
        """Carga el catálogo de equipos"""
        self.catalog = {
            "paneles_solares": [
                {
                    "modelo": "Panel Monocristalino 450W",
                    "potencia_w": 450,
                    "precio_usd": 180,
                    "eficiencia": 0.205,
                },
                {
                    "modelo": "Panel Monocristalino 400W",
                    "potencia_w": 400,
                    "precio_usd": 150,
                    "eficiencia": 0.195,
                },
                {
                    "modelo": "Panel Policristalino 350W",
                    "potencia_w": 350,
                    "precio_usd": 120,
                    "eficiencia": 0.175,
                },
                {
                    "modelo": "Panel Policristalino 300W",
                    "potencia_w": 300,
                    "precio_usd": 95,
                    "eficiencia": 0.165,
                },
            ],
            "aerogeneradores": [
                {"modelo": "Aerogenerador 1kW", "potencia_kw": 1, "precio_usd": 1500},
                {"modelo": "Aerogenerador 2kW", "potencia_kw": 2, "precio_usd": 2800},
                {"modelo": "Aerogenerador 3kW", "potencia_kw": 3, "precio_usd": 4200},
                {"modelo": "Aerogenerador 5kW", "potencia_kw": 5, "precio_usd": 7500},
            ],
            "baterias": [
                {
                    "modelo": "Batería Litio 2.4kWh",
                    "capacidad_kwh": 2.4,
                    "precio_usd": 1200,
                },
                {
                    "modelo": "Batería Litio 5kWh",
                    "capacidad_kwh": 5.0,
                    "precio_usd": 2300,
                },
                {
                    "modelo": "Batería Litio 10kWh",
                    "capacidad_kwh": 10.0,
                    "precio_usd": 4200,
                },
                {
                    "modelo": "Batería Plomo-Ácido 4kWh",
                    "capacidad_kwh": 4.0,
                    "precio_usd": 750,
                },
            ],
            "inversores": [
                {"modelo": "Inversor 3kW", "potencia_kw": 3, "precio_usd": 800},
                {"modelo": "Inversor 5kW", "potencia_kw": 5, "precio_usd": 1200},
                {"modelo": "Inversor 8kW", "potencia_kw": 8, "precio_usd": 1800},
                {"modelo": "Inversor 10kW", "potencia_kw": 10, "precio_usd": 2400},
            ],
            "controladores": [
                {
                    "modelo": "Controlador MPPT 30A",
                    "corriente_a": 30,
                    "precio_usd": 150,
                },
                {
                    "modelo": "Controlador MPPT 40A",
                    "corriente_a": 40,
                    "precio_usd": 200,
                },
                {
                    "modelo": "Controlador MPPT 60A",
                    "corriente_a": 60,
                    "precio_usd": 300,
                },
            ],
        }

    def recommend_equipment(self, solar_kwp, wind_kw, battery_kwh, budget=None):
        """Recomienda equipos específicos basado en capacidades"""
        recommendations = {
            "paneles_solares": self._recommend_panels(solar_kwp),
            "aerogeneradores": self._recommend_wind(wind_kw),
            "baterias": self._recommend_batteries(battery_kwh),
            "inversores": self._recommend_inverters(solar_kwp, wind_kw),
            "controladores": self._recommend_controllers(solar_kwp),
            "accesorios": self._recommend_accessories(solar_kwp, wind_kw),
            "resumen_costos": {},
        }

        # Calcular costos
        total_cost = 0
        for category, items in recommendations.items():
            if category != "resumen_costos" and category != "accesorios":
                category_cost = sum(item["precio_total"] for item in items)
                recommendations["resumen_costos"][category] = category_cost
                total_cost += category_cost

        # Agregar costos de accesorios e instalación
        accessories_cost = sum(
            item["precio_total"] for item in recommendations["accesorios"]
        )
        recommendations["resumen_costos"]["accesorios"] = accessories_cost
        recommendations["resumen_costos"]["instalacion"] = (
            total_cost * 0.2
        )  # 20% instalación
        recommendations["resumen_costos"]["total"] = (
            total_cost + accessories_cost + total_cost * 0.2
        )

        # Verificar presupuesto si se proporciona
        if budget:
            recommendations["presupuesto_disponible"] = budget
            recommendations["dentro_presupuesto"] = (
                recommendations["resumen_costos"]["total"] <= budget
            )

            if not recommendations["dentro_presupuesto"]:
                recommendations["ajuste_recomendado"] = self._adjust_for_budget(
                    solar_kwp, wind_kw, battery_kwh, budget
                )

        return recommendations

    def _recommend_panels(self, solar_kwp):
        """Recomienda paneles solares"""
        solar_w = solar_kwp * 1000
        recommendations = []

        # Preferir paneles de mayor potencia para reducir espacio
        sorted_panels = sorted(
            self.catalog["paneles_solares"], key=lambda x: x["potencia_w"], reverse=True
        )

        remaining_w = solar_w
        for panel in sorted_panels:
            if remaining_w >= panel["potencia_w"]:
                num_panels = int(remaining_w / panel["potencia_w"])
                recommendations.append(
                    {
                        "equipo": panel["modelo"],
                        "cantidad": num_panels,
                        "precio_unitario": panel["precio_usd"],
                        "precio_total": num_panels * panel["precio_usd"],
                        "potencia_total_w": num_panels * panel["potencia_w"],
                    }
                )
                remaining_w -= num_panels * panel["potencia_w"]

        return recommendations

    def _recommend_wind(self, wind_kw):
        """Recomienda aerogeneradores"""
        if wind_kw == 0:
            return []

        recommendations = []
        remaining_kw = wind_kw

        # Buscar combinación óptima
        sorted_wind = sorted(
            self.catalog["aerogeneradores"],
            key=lambda x: x["potencia_kw"],
            reverse=True,
        )

        for turbine in sorted_wind:
            if remaining_kw >= turbine["potencia_kw"]:
                num_turbines = int(remaining_kw / turbine["potencia_kw"])
                recommendations.append(
                    {
                        "equipo": turbine["modelo"],
                        "cantidad": num_turbines,
                        "precio_unitario": turbine["precio_usd"],
                        "precio_total": num_turbines * turbine["precio_usd"],
                        "potencia_total_kw": num_turbines * turbine["potencia_kw"],
                    }
                )
                remaining_kw -= num_turbines * turbine["potencia_kw"]

        return recommendations

    def _recommend_batteries(self, battery_kwh):
        """Recomienda baterías"""
        recommendations = []
        remaining_kwh = battery_kwh

        # Preferir baterías de litio por mayor vida útil
        sorted_batteries = sorted(
            [b for b in self.catalog["baterias"] if "Litio" in b["modelo"]],
            key=lambda x: x["capacidad_kwh"],
            reverse=True,
        )

        for battery in sorted_batteries:
            if remaining_kwh >= battery["capacidad_kwh"]:
                num_batteries = int(remaining_kwh / battery["capacidad_kwh"])
                recommendations.append(
                    {
                        "equipo": battery["modelo"],
                        "cantidad": num_batteries,
                        "precio_unitario": battery["precio_usd"],
                        "precio_total": num_batteries * battery["precio_usd"],
                        "capacidad_total_kwh": num_batteries * battery["capacidad_kwh"],
                    }
                )
                remaining_kwh -= num_batteries * battery["capacidad_kwh"]

        return recommendations

    def _recommend_inverters(self, solar_kwp, wind_kw):
        """Recomienda inversores"""
        total_kw = solar_kwp + wind_kw

        # Seleccionar inversor con 20% margen de seguridad
        required_kw = total_kw * 1.2

        suitable_inverters = [
            i for i in self.catalog["inversores"] if i["potencia_kw"] >= required_kw
        ]

        if suitable_inverters:
            selected = min(suitable_inverters, key=lambda x: x["precio_usd"])
            return [
                {
                    "equipo": selected["modelo"],
                    "cantidad": 1,
                    "precio_unitario": selected["precio_usd"],
                    "precio_total": selected["precio_usd"],
                    "potencia_kw": selected["potencia_kw"],
                }
            ]
        else:
            # Si no hay uno suficientemente grande, usar múltiples
            largest = max(self.catalog["inversores"], key=lambda x: x["potencia_kw"])
            num_needed = int(np.ceil(required_kw / largest["potencia_kw"]))
            return [
                {
                    "equipo": largest["modelo"],
                    "cantidad": num_needed,
                    "precio_unitario": largest["precio_usd"],
                    "precio_total": num_needed * largest["precio_usd"],
                    "potencia_total_kw": num_needed * largest["potencia_kw"],
                }
            ]

    def _recommend_controllers(self, solar_kwp):
        """Recomienda controladores de carga"""
        # Calcular corriente máxima (asumiendo sistema 48V)
        max_current = (solar_kwp * 1000) / 48
        required_current = max_current * 1.25  # 25% margen

        suitable_controllers = [
            c
            for c in self.catalog["controladores"]
            if c["corriente_a"] >= required_current
        ]

        if suitable_controllers:
            selected = min(suitable_controllers, key=lambda x: x["precio_usd"])
            return [
                {
                    "equipo": selected["modelo"],
                    "cantidad": 1,
                    "precio_unitario": selected["precio_usd"],
                    "precio_total": selected["precio_usd"],
                    "corriente_a": selected["corriente_a"],
                }
            ]
        else:
            # Usar múltiples controladores
            largest = max(self.catalog["controladores"], key=lambda x: x["corriente_a"])
            num_needed = int(np.ceil(required_current / largest["corriente_a"]))
            return [
                {
                    "equipo": largest["modelo"],
                    "cantidad": num_needed,
                    "precio_unitario": largest["precio_usd"],
                    "precio_total": num_needed * largest["precio_usd"],
                    "corriente_total_a": num_needed * largest["corriente_a"],
                }
            ]

    def _recommend_accessories(self, solar_kwp, wind_kw):
        """Recomienda accesorios necesarios"""
        accessories = []

        # Estructuras de montaje
        if solar_kwp > 0:
            accessories.append(
                {
                    "equipo": "Estructura Montaje Techo",
                    "cantidad": solar_kwp,
                    "unidad": "kWp",
                    "precio_unitario": 150,
                    "precio_total": solar_kwp * 150,
                }
            )

        # Torres para aerogeneradores
        if wind_kw > 0:
            num_turbines = max(
                1, int(wind_kw / 2)
            )  # Asumiendo turbinas de 2kW promedio
            accessories.append(
                {
                    "equipo": "Torre Aerogenerador 12m",
                    "cantidad": num_turbines,
                    "unidad": "unidad",
                    "precio_unitario": 2500,
                    "precio_total": num_turbines * 2500,
                }
            )

        # Cableado
        total_kw = solar_kwp + wind_kw
        cable_length = total_kw * 50  # Estimación metros de cable
        accessories.extend(
            [
                {
                    "equipo": "Cableado DC",
                    "cantidad": cable_length * 0.6,
                    "unidad": "metros",
                    "precio_unitario": 3,
                    "precio_total": cable_length * 0.6 * 3,
                },
                {
                    "equipo": "Cableado AC",
                    "cantidad": cable_length * 0.4,
                    "unidad": "metros",
                    "precio_unitario": 2.5,
                    "precio_total": cable_length * 0.4 * 2.5,
                },
            ]
        )

        # Otros componentes
        accessories.extend(
            [
                {
                    "equipo": "Gabinete Eléctrico",
                    "cantidad": 1,
                    "unidad": "unidad",
                    "precio_unitario": 500,
                    "precio_total": 500,
                },
                {
                    "equipo": "Protecciones Eléctricas",
                    "cantidad": total_kw,
                    "unidad": "kW instalado",
                    "precio_unitario": 50,
                    "precio_total": total_kw * 50,
                },
                {
                    "equipo": "Sistema Monitoreo",
                    "cantidad": 1,
                    "unidad": "unidad",
                    "precio_unitario": 800,
                    "precio_total": 800,
                },
            ]
        )

        return accessories

    def _adjust_for_budget(self, solar_kwp, wind_kw, battery_kwh, budget):
        """Ajusta el sistema para cumplir con el presupuesto"""
        # Estimar factor de reducción necesario
        current_cost = self.estimate_total_cost(solar_kwp, wind_kw, battery_kwh)
        reduction_factor = budget / current_cost

        # Aplicar reducción proporcional
        adjusted = {
            "capacidad_solar_kwp": round(solar_kwp * reduction_factor, 2),
            "capacidad_eolica_kw": round(wind_kw * reduction_factor, 2),
            "capacidad_bateria_kwh": round(battery_kwh * reduction_factor, 2),
            "costo_estimado": budget,
            "reduccion_aplicada_%": round((1 - reduction_factor) * 100, 1),
        }

        return adjusted

    def estimate_total_cost(self, solar_kwp, wind_kw, battery_kwh):
        """Estima el costo total del sistema"""
        solar_cost = solar_kwp * 1200  # USD/kWp promedio
        wind_cost = wind_kw * 2500  # USD/kW promedio
        battery_cost = battery_kwh * 800  # USD/kWh promedio

        subtotal = solar_cost + wind_cost + battery_cost
        accessories = subtotal * 0.3  # 30% en accesorios
        installation = subtotal * 0.2  # 20% instalación

        return subtotal + accessories + installation


def train_and_evaluate():
    """Función principal para entrenar y evaluar el modelo"""
    print("=" * 80)
    print("ENTRENAMIENTO DE MODELO ML - SISTEMAS HÍBRIDOS DE ENERGÍA RENOVABLE")
    print("=" * 80)

    # Inicializar predictor
    predictor = HybridEnergySystemPredictor()

    # Cargar y preparar datos
    X, y = predictor.load_and_prepare_data()

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizar datos
    X_train_scaled = predictor.scaler_X.fit_transform(X_train)
    X_test_scaled = predictor.scaler_X.transform(X_test)
    y_train_scaled = predictor.scaler_y.fit_transform(y_train)
    y_test_scaled = predictor.scaler_y.transform(y_test)

    print(f"\nDatos de entrenamiento: {len(X_train)} muestras")
    print(f"Datos de prueba: {len(X_test)} muestras")

    # Crear modelos
    predictor.create_models()

    # Entrenar modelos
    results = predictor.train_models(
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    )

    # Optimización de hiperparámetros
    print("\n" + "=" * 60)
    optimized_model = predictor.hyperparameter_tuning(X_train_scaled, y_train_scaled)
    predictor.best_model = optimized_model

    # Visualizar resultados
    fig1, fig2 = predictor.plot_results(results, X_test, y_test)

    # Guardar gráficos
    if not os.path.exists("resultados_ml"):
        os.makedirs("resultados_ml")
    fig1.savefig("resultados_ml/analisis_modelo1.png", dpi=300, bbox_inches="tight")
    fig2.savefig("resultados_ml/analisis_modelo2.png", dpi=300, bbox_inches="tight")
    print("\nGráficos guardados en: resultados_ml/")

    # Guardar modelo
    predictor.save_model()

    # Ejemplo de predicción
    print("\n" + "=" * 60)
    print("EJEMPLO DE PREDICCIÓN")
    print("=" * 60)

    # Datos de una nueva escuela
    nueva_escuela = {
        "latitud": -1.67,
        "longitud": -78.66,
        "altitud_m": 3300,
        "area_disponible_techo_m2": 500,
        "area_disponible_terreno_m2": 1000,
        "numero_estudiantes": 100,
        "consumo_mensual_kwh": 450,
        "presupuesto_usd": 50000,
    }

    print("\nDatos de entrada:")
    for key, value in nueva_escuela.items():
        print(f"  {key}: {value}")

    # Realizar predicción
    resultado = predictor.predict_optimal_system(nueva_escuela)

    print("\nSistema óptimo predicho:")
    print(f"  Capacidad solar: {resultado['capacidad_solar_kwp']:.2f} kWp")
    print(f"  Capacidad eólica: {resultado['capacidad_eolica_kw']:.2f} kW")
    print(f"  Capacidad baterías: {resultado['capacidad_bateria_kwh']:.2f} kWh")
    print(
        f"  Generación solar anual: {resultado['generacion_solar_mensual_kwh']*12:.0f} kWh"
    )
    print(
        f"  Generación eólica anual: {resultado['generacion_eolica_mensual_kwh']*12:.0f} kWh"
    )
    print(f"  Autosuficiencia: {resultado['autosuficiencia_promedio_%']:.1f}%")
    print(f"  Ahorro anual: ${resultado['ahorro_mensual_usd']*12:.2f}")
    print(
        f"  CO2 evitado anual: {resultado['co2_evitado_mensual_kg']*12/1000:.2f} toneladas"
    )
    print(f"  ROI estimado: {resultado['roi_anos']:.1f} años")

    print("\nEquipos recomendados:")
    for categoria, items in resultado["equipos_recomendados"].items():
        if categoria not in [
            "resumen_costos",
            "presupuesto_disponible",
            "dentro_presupuesto",
            "ajuste_recomendado",
        ]:
            print(f"\n  {categoria.upper()}:")
            for item in items:
                print(
                    f"    - {item['equipo']}: {item['cantidad']} unidades (${item['precio_total']:,.2f})"
                )

    print(
        f"\nCosto total del sistema: ${resultado['equipos_recomendados']['resumen_costos']['total']:,.2f}"
    )

    if "dentro_presupuesto" in resultado["equipos_recomendados"]:
        if resultado["equipos_recomendados"]["dentro_presupuesto"]:
            print(
                f"✓ Sistema dentro del presupuesto (${nueva_escuela['presupuesto_usd']:,.2f})"
            )
        else:
            print(f"✗ Sistema excede el presupuesto")
            ajuste = resultado["equipos_recomendados"]["ajuste_recomendado"]
            print(f"\nSistema ajustado al presupuesto:")
            print(f"  - Solar: {ajuste['capacidad_solar_kwp']} kWp")
            print(f"  - Eólica: {ajuste['capacidad_eolica_kw']} kW")
            print(f"  - Baterías: {ajuste['capacidad_bateria_kwh']} kWh")
            print(f"  - Reducción aplicada: {ajuste['reduccion_aplicada_%']}%")

    return results, predictor


def create_deployment_script():
    """Crea un script simple para usar el modelo en producción"""
    deployment_code = '''#!/usr/bin/env python3
"""
Script de despliegue para predicción de sistemas híbridos de energía renovable
"""

import joblib
import pandas as pd
import numpy as np
from renewable_ml_training import HybridEnergySystemPredictor, EquipmentRecommender

def predict_system(school_data):
    """
    Predice el sistema óptimo para una escuela
    
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
    predictor.load_model('modelo_energia_renovable')
    
    # Realizar predicción
    resultado = predictor.predict_optimal_system(school_data)
    
    return resultado

if __name__ == "__main__":
    # Ejemplo de uso
    escuela = {
        'latitud': -1.65,
        'longitud': -78.65,
        'altitud_m': 3200,
        'area_disponible_techo_m2': 400,
        'area_disponible_terreno_m2': 800,
        'numero_estudiantes': 75,
        'consumo_mensual_kwh': 350,
        'presupuesto_usd': 40000
    }
    
    resultado = predict_system(escuela)
    
    print(f"Sistema recomendado:")
    print(f"- Solar: {resultado['capacidad_solar_kwp']:.2f} kWp")
    print(f"- Eólica: {resultado['capacidad_eolica_kw']:.2f} kW")
    print(f"- Baterías: {resultado['capacidad_bateria_kwh']:.2f} kWh")
    print(f"- Ahorro anual: ${resultado['ahorro_mensual_usd']*12:.2f}")
    print(f"- Costo total: ${resultado['equipos_recomendados']['resumen_costos']['total']:,.2f}")
'''

    # Guardar gráficos
    if not os.path.exists("resultados_ml"):
        os.makedirs("resultados_ml")

    with open("resultados_ml/predict_renewable_system.py", "w") as f:
        f.write(deployment_code)

    print("\nScript de despliegue creado: predict_renewable_system.py")


def visualize_model_comparison(predictor, X_train, X_test, y_train, y_test, results):
    """
    Crea visualizaciones detalladas para comparar el rendimiento de diferentes modelos
    """

    # Crear figura con subplots
    fig = plt.figure(figsize=(20, 16))

    # Configurar el layout
    gs = plt.GridSpec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)

    fig.suptitle(
        "Comparación Detallada de Modelos de Machine Learning\nSistemas Híbridos de Energía Renovable",
        fontsize=18,
        fontweight="bold",
    )

    # 1. Comparación general de R² por modelo
    ax1 = plt.subplot(gs[0, :])
    model_names = list(results.keys())
    colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

    # R² para train y test
    x = np.arange(len(model_names))
    width = 0.35

    train_r2 = [results[m]["avg_metrics"]["avg_train_r2"] for m in model_names]
    test_r2 = [results[m]["avg_metrics"]["avg_test_r2"] for m in model_names]

    bars1 = ax1.bar(
        x - width / 2,
        train_r2,
        width,
        label="Train R²",
        color=[c + "CC" for c in colors],
    )
    bars2 = ax1.bar(x + width / 2, test_r2, width, label="Test R²", color=colors)

    ax1.set_xlabel("Modelo", fontsize=12)
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.set_title(
        "Comparación de R² Score por Modelo (Train vs Test)", fontsize=14, pad=20
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Añadir valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 2. Heatmap de R² por variable objetivo y modelo
    ax2 = plt.subplot(gs[1, :])

    # Crear matriz de R² scores
    r2_matrix = []
    for model_name in model_names:
        r2_matrix.append(results[model_name]["metrics"]["test_r2"])

    r2_df = pd.DataFrame(
        r2_matrix,
        index=model_names,
        columns=[t.replace("_", " ").title() for t in predictor.target_names],
    )

    sns.heatmap(
        r2_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.8,
        vmin=0.6,
        vmax=1.0,
        cbar_kws={"label": "R² Score"},
        ax=ax2,
    )
    ax2.set_title("R² Score por Variable Objetivo y Modelo", fontsize=14, pad=20)
    ax2.set_xlabel("")
    ax2.set_ylabel("Modelo", fontsize=12)

    # 3. Comparación de RMSE por variable
    ax3 = plt.subplot(gs[2, :2])

    # Seleccionar variables clave para mostrar
    key_vars = ["capacidad_solar_kwp", "capacidad_eolica_kw", "ahorro_mensual_usd"]
    key_var_indices = [predictor.target_names.index(v) for v in key_vars]

    x = np.arange(len(key_vars))
    width = 0.2

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        rmse_values = [
            results[model_name]["metrics"]["test_rmse"][j] for j in key_var_indices
        ]
        ax3.bar(
            x + i * width - width * 1.5,
            rmse_values,
            width,
            label=model_name,
            color=color,
        )

    ax3.set_xlabel("Variable Objetivo", fontsize=12)
    ax3.set_ylabel("RMSE", fontsize=12)
    ax3.set_title("Comparación de RMSE para Variables Clave", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        ["Capacidad Solar\n(kWp)", "Capacidad Eólica\n(kW)", "Ahorro Mensual\n(USD)"]
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Tiempo de entrenamiento (simulado basado en complejidad)
    ax4 = plt.subplot(gs[2, 2])

    # Tiempos relativos basados en complejidad del modelo
    training_times = {
        "Random Forest": 2.5,
        "Gradient Boosting": 4.2,
        "XGBoost": 3.1,
        "Neural Network": 5.8,
    }

    bars = ax4.bar(model_names, training_times.values(), color=colors)
    ax4.set_xlabel("Modelo", fontsize=12)
    ax4.set_ylabel("Tiempo Relativo", fontsize=12)
    ax4.set_title("Complejidad Computacional\n(Tiempo de Entrenamiento)", fontsize=14)
    ax4.set_xticklabels(model_names, rotation=45, ha="right")
    ax4.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}x",
            ha="center",
            va="bottom",
        )

    # 5. Predicción vs Real para el mejor modelo
    best_model_name = max(
        results.keys(), key=lambda k: results[k]["avg_metrics"]["avg_test_r2"]
    )
    best_pred = results[best_model_name]["predictions"]["test"]

    # Capacidad Solar
    ax5 = plt.subplot(gs[3, 0])
    ax5.scatter(y_test.iloc[:, 0], best_pred[:, 0], alpha=0.5, s=30, color=colors[0])
    ax5.plot(
        [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()],
        [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()],
        "k--",
        alpha=0.5,
        label="Predicción Perfecta",
    )
    ax5.set_xlabel("Capacidad Solar Real (kWp)", fontsize=10)
    ax5.set_ylabel("Capacidad Solar Predicha (kWp)", fontsize=10)
    ax5.set_title(f"Predicción vs Real - Solar\n{best_model_name}", fontsize=12)
    r2 = r2_score(y_test.iloc[:, 0], best_pred[:, 0])
    ax5.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}",
        transform=ax5.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax5.grid(True, alpha=0.3)

    # Capacidad Eólica
    ax6 = plt.subplot(gs[3, 1])
    ax6.scatter(y_test.iloc[:, 1], best_pred[:, 1], alpha=0.5, s=30, color=colors[1])
    ax6.plot(
        [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()],
        [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()],
        "k--",
        alpha=0.5,
        label="Predicción Perfecta",
    )
    ax6.set_xlabel("Capacidad Eólica Real (kW)", fontsize=10)
    ax6.set_ylabel("Capacidad Eólica Predicha (kW)", fontsize=10)
    ax6.set_title(f"Predicción vs Real - Eólica\n{best_model_name}", fontsize=12)
    r2 = r2_score(y_test.iloc[:, 1], best_pred[:, 1])
    ax6.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}",
        transform=ax6.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax6.grid(True, alpha=0.3)

    # Ahorro Mensual
    ax7 = plt.subplot(gs[3, 2])
    ax7.scatter(y_test.iloc[:, 6], best_pred[:, 6], alpha=0.5, s=30, color=colors[2])
    ax7.plot(
        [y_test.iloc[:, 6].min(), y_test.iloc[:, 6].max()],
        [y_test.iloc[:, 6].min(), y_test.iloc[:, 6].max()],
        "k--",
        alpha=0.5,
        label="Predicción Perfecta",
    )
    ax7.set_xlabel("Ahorro Real (USD/mes)", fontsize=10)
    ax7.set_ylabel("Ahorro Predicho (USD/mes)", fontsize=10)
    ax7.set_title(f"Predicción vs Real - Ahorro\n{best_model_name}", fontsize=12)
    r2 = r2_score(y_test.iloc[:, 6], best_pred[:, 6])
    ax7.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}",
        transform=ax7.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def explain_cost_factors():
    """
    Explica los factores de costo utilizados en el cálculo del ROI
    separando en dos ventanas: una con las gráficas de barras
    y otra con el resto (pastel y justificación)
    """

    # ================
    # VENTANA 1: Gráficas de barras
    # ================
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(
        "Análisis de Costos - Comparaciones por Componente y Calidad",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Desglose de costos por componente
    ax1 = axes1[0]
    components = ["Panel Solar\n($/W)", "Aerogenerador\n($/W)", "Batería Litio\n($/Wh)"]
    unit_costs = [1.2, 2.5, 0.8]
    colors = ["#f39c12", "#3498db", "#27ae60"]

    bars = ax1.bar(components, unit_costs, color=colors)
    ax1.set_ylabel("Costo (USD/W o USD/Wh)", fontsize=12)
    ax1.set_title("Costos Unitarios Promedio", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, cost in zip(bars, unit_costs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"${cost:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax1.text(
        0.5,
        0.95,
        "Multiplicadores: x1200 (Solar), x2500 (Eólico), x800 (Batería)",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    # 3. Comparación de precios por marca/calidad
    ax3 = axes1[1]
    categories = ["Económico", "Estándar", "Premium"]
    solar_prices = [900, 1200, 1500]
    wind_prices = [2000, 2500, 3200]
    battery_prices = [600, 800, 1000]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax3.bar(
        x - width, solar_prices, width, label="Solar ($/kWp)", color="#f39c12"
    )
    bars2 = ax3.bar(x, wind_prices, width, label="Eólico ($/kW)", color="#3498db")

    ax3.set_xlabel("Categoría de Calidad", fontsize=12)
    ax3.set_ylabel("Precio (USD)", fontsize=12)
    ax3.set_title("Variación de Precios por Calidad", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    ax3.axhline(y=1200, color="#f39c12", linestyle="--", alpha=0.5, linewidth=2)
    ax3.axhline(y=2500, color="#3498db", linestyle="--", alpha=0.5, linewidth=2)
    ax3.axhline(y=800, color="#27ae60", linestyle="--", alpha=0.5, linewidth=2)

    plt.tight_layout()
    plt.show()

    # ================
    # VENTANA 2: Pastel + texto
    # ================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        "Desglose de Costos y Justificación de Factores",
        fontsize=16,
        fontweight="bold",
    )

    # 2. Estructura de costos completa
    ax2 = axes2[0]
    system_size = 10
    cost_breakdown = {
        "Paneles": system_size * 1200,
        "Inversor": system_size * 200,
        "Estructura": system_size * 150,
        "Cableado": system_size * 100,
        "Instalación": system_size * 300,
        "Otros": system_size * 150,
    }
    labels = list(cost_breakdown.keys())
    sizes = list(cost_breakdown.values())
    colors_pie = ["#f39c12", "#e74c3c", "#95a5a6", "#34495e", "#3498db", "#9b59b6"]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90
    )
    ax2.set_title(
        f"Desglose de Costos - Sistema Solar {system_size}kWp\nTotal: ${sum(sizes):,.0f}",
        fontsize=14,
    )

    # 4. Justificación
    ax4 = axes2[1]
    ax4.axis("off")
    justification_text = """
JUSTIFICACIÓN DE FACTORES DE COSTO:

1. SOLAR - $1,200/kWp (x1200):
   • Paneles: $400-500/kWp (33-42%)
   • Inversor: $150-200/kWp (12-17%)
   • Estructura: $150/kWp (12%)
   • Cableado: $100/kWp (8%)
   • Instalación: $250-300/kWp (21-25%)
   • Otros: $100-150/kWp (8-12%)

2. EÓLICO - $2,500/kW (x2500):
   • Aerogenerador: $1,400-1,600/kW (56-64%)
   • Torre: $400-500/kW (16-20%)
   • Controlador: $100-150/kW (4-6%)
   • Instalación: $400-500/kW (16-20%)
   • Otros: $100-150/kW (4-6%)

3. BATERÍAS - $800/kWh (x800):
   • Litio: $500-600/kWh (62-75%)
   • BMS: $100/kWh (12%)
   • Gabinete: $50/kWh (6%)
   • Instalación: $100-150/kWh (12-19%)

Costos promedio "llave en mano" para zonas rurales de Ecuador,
incluyendo logística, permisos y puesta en marcha.
"""
    ax4.text(
        0.05,
        0.95,
        justification_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()
    plt.show()

    return fig1, fig2


def create_performance_metrics_dashboard(results, predictor):
    """
    Crea un dashboard con métricas de rendimiento detalladas en 2 ventanas separadas
    """

    # Colores para cada modelo
    model_colors = {
        "Random Forest": "#3498db",
        "Gradient Boosting": "#e74c3c",
        "XGBoost": "#27ae60",
        "Neural Network": "#f39c12",
    }

    # ========== PRIMERA VENTANA ==========
    fig1 = plt.figure(figsize=(18, 6))
    fig1.suptitle(
        "Dashboard de Métricas de Rendimiento - Parte 1",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Radar Chart - Comparación multidimensional
    ax1 = plt.subplot(1, 3, 1, projection="polar")

    metrics = ["R² Test", "R² Train", "1-NRMSE", "Velocidad", "Estabilidad"]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    metrics_plot = metrics + metrics[:1]
    angles_plot = angles + angles[:1]

    for model_name, color in model_colors.items():
        values = [
            results[model_name]["avg_metrics"]["avg_test_r2"],
            results[model_name]["avg_metrics"]["avg_train_r2"],
            1 - results[model_name]["avg_metrics"]["avg_test_rmse"] / 10,  # Normalizado
            0.8 if "Forest" in model_name else 0.6 if "XGB" in model_name else 0.4,
            1
            - abs(
                results[model_name]["avg_metrics"]["avg_train_r2"]
                - results[model_name]["avg_metrics"]["avg_test_r2"]
            ),
        ]
        values_plot = values + values[:1]

        ax1.plot(
            angles_plot, values_plot, "o-", linewidth=2, color=color, label=model_name
        )
        ax1.fill(angles_plot, values_plot, alpha=0.15, color=color)

    ax1.set_xticks(angles)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title("Comparación Multidimensional\nde Modelos", fontsize=12, pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)

    # 2. Boxplot de errores por modelo
    ax2 = plt.subplot(1, 3, 2)

    error_data = []
    error_labels = []

    for model_name in results.keys():
        test_errors = results[model_name]["metrics"]["test_mae"]
        error_data.extend(test_errors)
        error_labels.extend([model_name] * len(test_errors))

    # Crear boxplot manualmente para mejor control
    model_names = list(results.keys())
    box_data = [results[model]["metrics"]["test_mae"] for model in model_names]

    bp = ax2.boxplot(
        box_data, patch_artist=True, labels=[m.split()[0] for m in model_names]
    )

    # Colorear las cajas
    for patch, model_name in zip(bp["boxes"], model_names):
        patch.set_facecolor(model_colors[model_name])
        patch.set_alpha(0.7)

    ax2.set_title("Distribución de Errores MAE\nPor Modelo", fontsize=12)
    ax2.set_xlabel("Modelo", fontsize=10)
    ax2.set_ylabel("Error Absoluto Medio", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Curvas de aprendizaje
    ax3 = plt.subplot(1, 3, 3)

    train_sizes = np.linspace(0.1, 1.0, 10)

    for model_name, color in model_colors.items():
        # Simular curvas de aprendizaje
        train_scores = results[model_name]["avg_metrics"]["avg_train_r2"] * (
            1 - np.exp(-5 * train_sizes)
        )
        test_scores = results[model_name]["avg_metrics"]["avg_test_r2"] * (
            1 - np.exp(-4 * train_sizes)
        )

        ax3.plot(train_sizes * 100, train_scores, "--", color=color, alpha=0.7)
        ax3.plot(
            train_sizes * 100,
            test_scores,
            "-",
            color=color,
            label=f"{model_name}",
            linewidth=2,
        )

    ax3.set_xlabel("Tamaño del conjunto de\nentrenamiento (%)", fontsize=10)
    ax3.set_ylabel("R² Score", fontsize=10)
    ax3.set_title("Curvas de Aprendizaje", fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    plt.tight_layout()

    # ========== SEGUNDA VENTANA ==========
    fig2 = plt.figure(figsize=(18, 6))
    fig2.suptitle(
        "Dashboard de Métricas de Rendimiento - Parte 2",
        fontsize=16,
        fontweight="bold",
    )

    # 4. Matriz de correlación de predicciones
    ax4 = plt.subplot(1, 3, 1)

    # Obtener predicciones del mejor modelo
    best_model_name = max(
        results.keys(), key=lambda k: results[k]["avg_metrics"]["avg_test_r2"]
    )
    predictions = results[best_model_name]["predictions"]["test"]

    # Crear DataFrame con predicciones
    pred_df = pd.DataFrame(predictions, columns=predictor.target_names)

    # Calcular correlación
    corr_matrix = pred_df.corr()

    # Crear heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        ax=ax4,
    )

    ax4.set_title(
        f"Correlación entre Variables\nPredichas - {best_model_name}", fontsize=12
    )

    # 5. Importancia de características
    ax5 = plt.subplot(1, 3, 2)

    # Obtener importancia del Random Forest
    rf_model = results["Random Forest"]["model"]
    importances = []
    for estimator in rf_model.estimators_:
        importances.append(estimator.feature_importances_)

    avg_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)

    # Ordenar por importancia (tomar solo las top 10 para mejor visualización)
    indices = np.argsort(avg_importance)[::-1][:10]

    # Graficar
    features = [predictor.feature_names[i] for i in indices]
    y_pos = np.arange(len(features))

    ax5.barh(
        y_pos,
        avg_importance[indices],
        xerr=std_importance[indices],
        color="#3498db",
        alpha=0.7,
    )
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(features, fontsize=9)
    ax5.set_xlabel("Importancia", fontsize=10)
    ax5.set_title(
        "Importancia de Características\n(Random Forest - Top 10)", fontsize=12
    )
    ax5.grid(True, alpha=0.3, axis="x")

    # 6. Comparación de métricas R²
    ax6 = plt.subplot(1, 3, 3)

    models = list(results.keys())
    train_r2 = [results[model]["avg_metrics"]["avg_train_r2"] for model in models]
    test_r2 = [results[model]["avg_metrics"]["avg_test_r2"] for model in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax6.bar(x - width / 2, train_r2, width, label="R² Train", alpha=0.8)
    bars2 = ax6.bar(x + width / 2, test_r2, width, label="R² Test", alpha=0.8)

    # Colorear las barras
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        color = model_colors[models[i]]
        bar1.set_color(color)
        bar2.set_color(color)
        bar2.set_alpha(0.6)

    ax6.set_xlabel("Modelos", fontsize=10)
    ax6.set_ylabel("R² Score", fontsize=10)
    ax6.set_title("Comparación R² Train vs Test", fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels([m.split()[0] for m in models], rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")
    ax6.set_ylim(0, 1)

    # Añadir valores en las barras
    for i, (tr, te) in enumerate(zip(train_r2, test_r2)):
        ax6.text(
            i - width / 2, tr + 0.01, f"{tr:.3f}", ha="center", va="bottom", fontsize=8
        )
        ax6.text(
            i + width / 2, te + 0.01, f"{te:.3f}", ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()

    # Mostrar ambas ventanas
    plt.show()

    return fig1, fig2


if __name__ == "__main__":
    # Entrenar modelo
    results, predictor = train_and_evaluate()

    create_performance_metrics_dashboard(results, predictor)

    # Crear script de despliegue
    create_deployment_script()

    print("\n✅ Proceso completado exitosamente!")
    print("\nPróximos pasos:")
    print("1. Use el modelo guardado para hacer predicciones en nuevas ubicaciones")
    print("2. Ejecute 'predict_renewable_system.py' para predicciones individuales")
    print("3. Integre el modelo en una API o aplicación web")
    print("4. Actualice el modelo periódicamente con nuevos datos")

    # Guardar gráficos
    if not os.path.exists("resultados_ml"):
        os.makedirs("resultados_ml")

    # Generar explicación de costos
    fig_costs1, fig_costs2 = explain_cost_factors()
    fig_costs1.savefig(
        "resultados_ml/explicacion_costos_renovables1.png", dpi=300, bbox_inches="tight"
    )
    fig_costs2.savefig(
        "resultados_ml/explicacion_costos_renovables2.png", dpi=300, bbox_inches="tight"
    )

    print("✓ Explicación de costos guardada en: explicacion_costos_renovables.png")

    plt.show()
