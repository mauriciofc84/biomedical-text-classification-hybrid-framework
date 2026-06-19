import ast
import sys
import os
import gc
import copy
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

# === LECTURA DE ARGUMENTOS DE LÍNEA DE COMANDOS ===
# Este script está diseñado para ser ejecutado desde la terminal.
# Ejemplo: python main.py DATASET 1 5 regex LC
FILENAME = sys.argv[1]   # Nombre del archivo/dataset (ej: "TEST")
min_idx = int(sys.argv[2]) # Índice inicial del Fold (para paralelizar)
max_idx = int(sys.argv[3]) # Índice final del Fold
MODELS = [sys.argv[4]]   # Modelo a usar (ej: "regex", "svm", "bert")
CURVES = [sys.argv[5]]   # Tipo de curva (ej: "LC" para Learning Curve / Active Learning)

# === CONFIGURACIÓN DE LOGGING Y ADVERTENCIAS ===
import warnings
warnings.filterwarnings("ignore") # Ignora warnings para mantener la consola limpia
import logging
logging.captureWarnings(True)
#logging.disable(sys.maxsize)

# Importamos las clases principales del proyecto
from cregex import *
from utils import *
from curves import *

# Fijamos la semilla para reproducibilidad global
seed_everything()

# === CARGA DEL DATASET ===
# Busca el archivo pickle preprocesado que contiene los textos y sus etiquetas.
data_path = os.path.join(os.getcwd(), "snippets_procesados_" + FILENAME + ".pkl")
if not os.path.exists(data_path):
    # Fallback: si no encuentra el específico, busca uno genérico
    data_path = os.path.join(os.getcwd(), "snippets_procesados.pkl")

print(f"[INFO] Cargando dataset desde: {data_path}")
with open(data_path, "rb") as f:
    data = pickle.load(f)

# === PREPARACIÓN DE DATOS ===
# Ordena y separa la lista de tuplas en dos arrays: DATA (X) y CLASSES (y)
data = sorted(data, key=lambda x: x[0], reverse=False)
DATA = np.array([snippet for snippet, classe in data])
CLASSES = np.array([classe for snippet, classe in data])

# === DETECCIÓN AUTOMÁTICA DE CLASES ===
# Identifica cuántas clases únicas existen en el dataset (binario vs multiclase).
unique_classes = np.unique(CLASSES)
N_CLASSES = len(unique_classes)
print(f"[INFO] Clases detectadas ({N_CLASSES}): {unique_classes}")

# === ACTUALIZACIÓN DINÁMICA DE HIPERPARÁMETROS ===
# Inyecta el número de clases detectado en la configuración de los modelos Deep Learning.
HYPERPARAMS["bert"]["n_classes"] = N_CLASSES
HYPERPARAMS["setfit"]["n_classes"] = N_CLASSES

# Crea la estructura de carpetas necesaria para guardar resultados (out/RESULTS, etc.)
create_paths(FILENAME)

print(f"[INFO] FILENAME: {FILENAME}")
RUNS = 1       # Número de ejecuciones completas (normalmente 1 si usamos K-Fold)
FOLDS = 5      # Número de particiones para Validación Cruzada
folds = KFold(n_splits=FOLDS, shuffle=False)
idxs = np.arange(0, len(DATA))

# === BUCLE PRINCIPAL DE EJECUCIÓN ===
for r in range(RUNS):
    # Mezcla los índices aleatoriamente antes de dividir
    idxs = shuffle(idxs, random_state=SEED)
    CLASSES = CLASSES[idxs]
    DATA = DATA[idxs]
    k = -1
    
    # === VALIDACIÓN CRUZADA (K-FOLD) ===
    # Itera sobre las particiones de Train/Test
    for train_index, test_index in folds.split(idxs):
        k += 1
        print("fold:", k + 1)
        
        # Filtro de ejecución: Permite correr solo ciertos folds (útil para paralelizar en clusters)
        if (k + 1) not in list(range(min_idx, max_idx + 1)):
            continue

        for CURVE in CURVES:
            for MODEL in MODELS:
                # Restricción: Pseudo-Labeling (PL) no se suele aplicar a modelos Regex/BERT en este script
                if "PL" in CURVE:
                    if "regex" in MODEL or "bert" in MODEL or "setfit" in MODEL:
                        continue

                print("CURVE:", CURVE)
                print("MODEL:", MODEL)

                # === DIVISIÓN DE DATOS (TRAIN / VAL / TEST) ===
                # 1. Obtenemos el Train y Test del K-Fold actual
                X_train = copy.deepcopy(DATA[train_index])
                y_train = copy.deepcopy(CLASSES[train_index])
                
                # 2. Sub-dividimos el Train para sacar un conjunto de Validación (20%)
                # Esto es crucial para ajustar umbrales o hiperparámetros sin tocar el Test set.
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=SEED
                )
                X_test = copy.deepcopy(DATA[test_index])
                y_test = copy.deepcopy(CLASSES[test_index])

                # === INSTANCIACIÓN Y EJECUCIÓN DE LA CURVA ===
                # La clase 'Curves' maneja el bucle de Active Learning (entrenar -> predecir -> seleccionar -> repetir)
                curve = Curves(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    N_CLASSES,
                    CURVE,
                    MODEL,
                    BATCH,       # Tamaño del lote a etiquetar en cada iteración de AL
                    FILENAME,
                )
                
                # Inicia el proceso de aprendizaje activo
                curve.learningCurve()
                
                # === RECOPILACIÓN DE RESULTADOS ===
                results = [
                    curve.results["scores"],       # Scores de confianza
                    curve.results["x"],            # Eje X: Cantidad de datos etiquetados
                    curve.results["y"],            # Eje Y: Performance (Accuracy/F1)
                    curve.results["y_u_dst"],      # Distribución real de lo no etiquetado (para análisis)
                    curve.results["y_clf"],        # Predicciones sobre lo no etiquetado
                    curve.results["dst_cregex"],   # Estadísticas específicas de CRegex
                    curve.HYPERPARAMS,             # Hiperparámetros usados
                    curve.N_FEATURES,              # Número de features (reglas o n-gramas)
                    y_test,                        # Etiquetas reales del test set (para calcular métricas finales)
                ]
                
                # === GUARDADO DE RESULTADOS EN DISCO ===
                out_path = os.path.join(
                    os.getcwd(),
                    "out",
                    "RESULTSLC",
                    CURVE,
                    FILENAME,
                    FILENAME + "_" + MODEL + "_" + CURVE + "_k" + str(k + 1) + ".pkl",
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as a:
                    pickle.dump(results, a, protocol=2) # Protocolo 2 para compatibilidad

                # Limpieza de memoria para evitar desbordamientos en ejecuciones largas
                del X_train, X_val, X_test, y_train, y_val, y_test
                gc.collect()

# !pip install gensim transformers seaborn openpyxl scikit-learn lingpy xlsxwriter editdistance xgboost