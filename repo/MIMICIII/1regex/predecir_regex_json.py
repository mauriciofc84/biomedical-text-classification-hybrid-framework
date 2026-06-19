import sys
import os
import json
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# SE AGREGO classification_report AQUI ABAJO
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import Counter, defaultdict
from sklearn.utils import shuffle
import datetime


# ==============================================================================
# [0] SISTEMA DE LOGS
# ==============================================================================
MODEL_NAME = "predecir_regex"
DIR_LOGS = f"LOGS_{MODEL_NAME}"

os.makedirs(DIR_LOGS, exist_ok=True)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(DIR_LOGS, f"log_ejecucion_{timestamp}.txt")
        self.log = open(self.log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout 

print(f"--- INICIO DEL PROCESO: {datetime.datetime.now()} ---")
print(f"📂 Log guardado en: {DIR_LOGS}")

# ================= CONFIGURACIÓN ORIGINAL =================
DATASET_NAME = "snippets_procesados.pkl" 
LIMIT_TRAIN = 1907
ARCHIVO_JSON_INDICES = "out/regex_rules/test_regexes.json"
ARCHIVO_CLASES_TXT = "out/CLASESX_test.txt"

CLASES_MAP = {
    0: "external_causes",
    1: "supplementary_factors",
}
# =================================================

print("\n--- SISTEMA DE RECONSTRUCCIÓN (MODIFICADO: PRIOR DINÁMICO + SIN FILTROS) ---")

# ==============================================================================
# [1] CARGAR TABLA DE VERDAD
# ==============================================================================
print(f"\n[1] Cargando clases maestras desde: {ARCHIVO_CLASES_TXT}")

if not os.path.exists(ARCHIVO_CLASES_TXT):
    if os.path.exists(f"out/{ARCHIVO_CLASES_TXT}"):
        ARCHIVO_CLASES_TXT = f"out/{ARCHIVO_CLASES_TXT}"
    else:
        print(f"❌ ERROR: No encuentro {ARCHIVO_CLASES_TXT}")
        sys.exit(1)

clases_verdad = []
with open(ARCHIVO_CLASES_TXT, "r") as f:
    for line in f:
        line = line.strip()
        if line.isdigit():
            clases_verdad.append(int(line))

print(f"    -> ¡ÉXITO! Se cargaron {len(clases_verdad)} etiquetas de documentos.")

# ==============================================================================
# [2] CARGAR ÍNDICE DE REGLAS
# ==============================================================================
print(f"\n[2] Cargando mapa de reglas: {ARCHIVO_JSON_INDICES}")
if not os.path.exists(ARCHIVO_JSON_INDICES):
    posible_ruta = "out/regex_rules/TEST_regexes.json"
    if os.path.exists(posible_ruta):
        ARCHIVO_JSON_INDICES = posible_ruta
    else:
        print("❌ ERROR: No encuentro el JSON de regexes.")
        sys.exit(1)

with open(ARCHIVO_JSON_INDICES, "r", encoding="utf-8") as f:
    json_indices = json.load(f)

# ==============================================================================
# [3] RECONSTRUCCIÓN DE INTELIGENCIA
# ==============================================================================
print(f"\n[3] Cruzando índices con clases...")

reglas_aprendidas = defaultdict(list) # Importar: from collections import defaultdict
conteo_reglas = 0
reglas_ignoradas = 0
reglas_filtradas = 0

STOP_EXTRA = []

for patron, data in json_indices.items():
    patron_limpio = re.sub(r'[^a-zA-Z]', '', patron).upper()
    
    if patron_limpio in STOP_EXTRA:
        reglas_filtradas += 1
        continue

    posiciones = data.get("positions", [])
    if not posiciones:
        continue
        
    votos = []
    for pos in posiciones:
        if pos < len(clases_verdad):
            votos.append(clases_verdad[pos])
            
    if not votos:
        reglas_ignoradas += 1
        continue
        
    clase_ganadora = Counter(votos).most_common(1)[0][0]
    
    try:
        reglas_aprendidas[clase_ganadora].append(re.compile(patron, re.IGNORECASE))
        conteo_reglas += 1
    except:
        pass

print(f"✅ RECONSTRUCCIÓN FINALIZADA:")
print(f"   - Reglas Útiles Recuperadas: {conteo_reglas}")
print(f"   - Reglas Descartadas: {reglas_filtradas + reglas_ignoradas}")


# ==============================================================================
# [4] CARGAR DATOS DE TEST Y CALCULAR PRIOR
# ==============================================================================
print(f"\n[4] Cargando datos y calculando clase por defecto...")
if not os.path.exists(DATASET_NAME):
    print(f"❌ ERROR: No encuentro {DATASET_NAME}")
    sys.exit(1)

with open(DATASET_NAME, "rb") as f:
    data_raw = pickle.load(f)

data_raw = shuffle(data_raw, random_state=42)

TODOS_LOS_TEXTOS = np.array([snippet for snippet, classe in data_raw])
TODAS_LAS_CLASES = np.array([classe for snippet, classe in data_raw])

X_nuevos = TODOS_LOS_TEXTOS[LIMIT_TRAIN:] 
y_nuevos_real = TODAS_LAS_CLASES[LIMIT_TRAIN:] 

y_entrenamiento = TODAS_LAS_CLASES[:LIMIT_TRAIN]
conteo_entrenamiento = Counter(y_entrenamiento)
CLASE_POR_DEFECTO = conteo_entrenamiento.most_common(1)[0][0]

nombre_clase_defecto = CLASES_MAP.get(CLASE_POR_DEFECTO, "Desconocida")
print(f"   -> [AUTO] Clase mayoritaria en entrenamiento: {CLASE_POR_DEFECTO} ({nombre_clase_defecto})")
print(f"   -> Evaluando {len(X_nuevos)} casos de prueba.")


# ==============================================================================
# [5] PREDICCIÓN
# ==============================================================================
print(f"\n[5] Clasificando...")
predicciones = []
filas_excel = []

for i, texto in enumerate(X_nuevos):
    # En lugar de manual, usa los IDs del mapeo
    scores = {cid: 0 for cid in CLASES_MAP.keys()}
    activadas = []
    
    for cid, lista_regex in reglas_aprendidas.items():
        for regex in lista_regex:
            if regex.search(texto):
                scores[cid] += 1
                activadas.append(regex.pattern)
    
    hits_totales = sum(scores.values())
    
    if hits_totales == 0:
        ganador = CLASE_POR_DEFECTO
        metodo = "FALLBACK (PRIOR)"
    else:
        ganador = max(scores, key=scores.get)
        metodo = "REGLAS"
    
    predicciones.append(ganador)
    
    real = int(y_nuevos_real[i])
    filas_excel.append({
        "Texto": texto,
        "Clase_REAL": CLASES_MAP.get(real, str(real)),
        "Prediccion_MODELO": CLASES_MAP.get(ganador, str(ganador)),
        "Acerto": "SI" if real == ganador else "NO",
        "Total_Hits": hits_totales,
        "Metodo_Decisión": metodo,
        "Reglas_Activadas": str(activadas[:10]) 
    })


# ==============================================================================
# [6] RESULTADOS (CON MÉTRICAS DETALLADAS)
# ==============================================================================
y_pred = np.array(predicciones)
# Convertimos y_nuevos_real a int por seguridad para el reporte
y_nuevos_real = y_nuevos_real.astype(int)

acc = accuracy_score(y_nuevos_real, y_pred)
f1 = f1_score(y_nuevos_real, y_pred, average='weighted')

print("\n" + "="*50)
print(f"RESULTADOS MODELO FINAL (MODIFICADO)")
print("="*50)
print(f"ACCURACY: {acc:.4f}")
print(f"F1 SCORE: {f1:.4f}")
print("="*50)

# --- AQUÍ ESTÁ EL REPORTE DETALLADO QUE PEDISTE ---
print("\n--- REPORTE DETALLADO POR CLASE ---")
print(classification_report(y_nuevos_real, y_pred, target_names=CLASES_MAP.values()))
print("="*50)

pd.DataFrame(filas_excel).to_excel("reporte_reconstruido_modificado.xlsx", index=False)
print("[EXITO] Reporte guardado: reporte_reconstruido_modificado.xlsx")

plt.figure(figsize=(8,6))
# Genera los nombres basados en las clases que realmente aparecieron en el test
labels_presentes = sorted(list(set(y_nuevos_real) | set(y_pred)))
nombres = [CLASES_MAP.get(i, str(i)) for i in labels_presentes]
cm = confusion_matrix(y_nuevos_real, y_pred, labels=labels_presentes)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nombres, yticklabels=nombres)
plt.title(f"Matriz de Confusión\n(Prior: {CLASE_POR_DEFECTO})")
plt.tight_layout()
plt.savefig("matriz_reconstruida_modificada.png")
print("--> Gráfico guardado.")

print(f"✅ FIN DEL PROCESO: {datetime.datetime.now()}")