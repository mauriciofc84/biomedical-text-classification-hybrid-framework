import sys
import os
import json
import re
import pickle
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import datetime

# ==============================================================================
# [0] CONFIGURACIÓN Y SISTEMA DE LOGS
# ==============================================================================
MODEL_NAME = "regexNB"
DATASET_NAME = "snippets_procesados.pkl" 
LIMIT_TRAIN = 1907
TOP_K_PLOT = 30    

DIR_EXCEL = f"REPORTES_EXCEL_{MODEL_NAME}"
DIR_IMG = f"GRAFICOS_{MODEL_NAME}"
DIR_LOGS = f"LOGS_{MODEL_NAME}"

os.makedirs(DIR_EXCEL, exist_ok=True)
os.makedirs(DIR_IMG, exist_ok=True)
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ ADVERTENCIA: La librería 'shap' no está instalada.")

ARCHIVO_CLASES_TXT = "CLASESX_TEST.txt"
ARCHIVO_JSON_REGLAS = "TEST_regexes.json"
ARCHIVO_JSON_MAPS = "TEST_maps.json"

CLASES_MAP = {
    0: "external_causes",
    1: "supplementary_factors"
}

plt.style.use('ggplot')

# [1] UTILERÍA DE ARCHIVOS
def encontrar_archivo(nombre):
    rutas = [nombre, os.path.join("out", nombre), os.path.join("out", "regex_rules", nombre)]
    for r in rutas:
        if os.path.exists(r): return r
    return None

ruta_clases = encontrar_archivo(ARCHIVO_CLASES_TXT)
ruta_reglas = encontrar_archivo(ARCHIVO_JSON_REGLAS)
ruta_maps = encontrar_archivo(ARCHIVO_JSON_MAPS)

if not ruta_reglas: 
    print("❌ Falta archivo de reglas.")
    sys.exit(1)

with open(ruta_reglas, "r", encoding="utf-8") as f: raw_rules = json.load(f)
with open(ruta_maps, "r", encoding="utf-8") as f: raw_maps = json.load(f)
mapa_tokens = raw_maps.get("pattern2token", {})

# ==============================================================================
# [2] PROCESAMIENTO DE REGLAS (PREPARACIÓN DEL CANAL 1)
# ==============================================================================
print("\n[1] Construyendo features de Regex...")
lista_reglas_obj = []
nombres_reglas = []

# Definimos qué patrones queremos ignorar por completo (copia aquí tus regex de stop)
PATRONES_A_BLOQUEAR = [
  #  r'.*trastorno.*', r'hiperten.*', r'\d+', r'.*stornos?', 
  #  r'\d+\s+\d+', r'\d+[^\S]+', r'\*\s*\d+', r'\*.*cardiopat.*',
  #  r'\*.*cardiopat.*', r'cardiopat.*', r'\d+.*\d+.*', r'\*.*\d+.*\d+.*'


]

for patron_str in raw_rules.keys():
    # --- NUEVO FILTRO DE SEGURIDAD ---
    # Si la regla del JSON coincide con algo que queremos mutear, la saltamos
    skip_regla = False
    for bloqueado in PATRONES_A_BLOQUEAR:
        if re.search(bloqueado, patron_str, re.IGNORECASE):
            skip_regla = True
            break
    if skip_regla: continue 
    # ---------------------------------

    token = mapa_tokens.get(patron_str, patron_str)
    if token == patron_str:
        s = re.sub(r'\(\?[:!=].*?\)', '', patron_str)
        s = re.sub(r'[\\[\\]\(\)\*\+\?\|\^]', '', s)
        token = s.strip()
    
    if len(token) < 2: continue
    
    try:
        regex = re.compile(patron_str, re.IGNORECASE)
        lista_reglas_obj.append(regex)
        nombres_reglas.append(f"R:{token}") 
    except: pass

print(f" -> Reglas compiladas (tras filtro): {len(lista_reglas_obj)}")

# [3] CARGAR DATASET
print("\n[2] Cargando datos...")
if not os.path.exists(DATASET_NAME): 
    print(f"❌ Falta {DATASET_NAME}")
    sys.exit(1)

with open(DATASET_NAME, "rb") as f: data_pkl = pickle.load(f)
from sklearn.utils import shuffle
data_pkl = shuffle(data_pkl, random_state=42)

textos = np.array([d[0] for d in data_pkl])
y = np.array([d[1] for d in data_pkl]).astype(int)

# ==============================================================================
# [4] INGENIERÍA DE CARACTERÍSTICAS (EL NÚCLEO CIENTÍFICO)
# ==============================================================================
print("\n[3] Generando Matrices Híbridas...")

# --- CANAL 1: REGEX ---
def vectorizar_regex(txts, reglas):
    m = np.zeros((len(txts), len(reglas)), dtype=int)
    for i, t in enumerate(txts):
        for j, r in enumerate(reglas):
            if r.search(t): m[i, j] = 1
    return m

X_regex = vectorizar_regex(textos, lista_reglas_obj)
X_regex_sparse = csr_matrix(X_regex)

# --- CANAL 2: BAG OF WORDS (CON ANALIZADOR REGEX PERSONALIZADO) ---

# Definimos las reglas de stopwords complejas
REGEX_STOPWORDS = [
#    r'.*trastorno.*', r'hiperten.*', r'\d+', 
#    r'.*stornos?'
    #r'\d+?[\s]\*\*[\s]*\d+?[\s]\*\*', r'\d+?[^\S]*[^\S]\*\*', r'\d+'
]
LITERAL_STOPWORDS = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 
                'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 
                'should', 'now'
}

def custom_analyzer(doc):
    # Normalización básica
    doc = ''.join(c for c in unicodedata.normalize('NFD', doc) if unicodedata.category(c) != 'Mn').lower()
    # Tokenización estándar de sklearn
    tokens = re.findall(r"(?u)\b\w\w+\b", doc)
    
    filtrados = []
    for t in tokens:
        if t in LITERAL_STOPWORDS: continue
        
        # Validación contra patrones Regex
        es_prohibida = False
        for patron in REGEX_STOPWORDS:
            try:
                if re.search(patron, t):
                    es_prohibida = True
                    break
            except: continue
        
        if not es_prohibida:
            filtrados.append(t)
    return filtrados

vectorizer_bow = CountVectorizer(max_features=500, analyzer=custom_analyzer)

X_bow = vectorizer_bow.fit_transform(textos)
nombres_bow = [f"W:{w}" for w in vectorizer_bow.get_feature_names_out()]

# --- FUSIÓN (STACKING) ---
X_final = hstack([X_regex_sparse, X_bow])
all_feature_names = np.array(nombres_reglas + nombres_bow)

X_train = X_final[:LIMIT_TRAIN]
y_train = y[:LIMIT_TRAIN]
X_test = X_final[LIMIT_TRAIN:]
y_test = y[LIMIT_TRAIN:]

# [5] ENTRENAMIENTO
print("\n[4] Entrenando Modelo Híbrido...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# [6] EVALUACIÓN
y_pred = nb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*50)
print(f"RESULTADOS: FEATURE STACKING")
print("="*50)
print(f"ACCURACY: {acc:.4f}")
print(f"F1 SCORE: {f1:.4f}")
print("="*50)
print(classification_report(y_test, y_pred, target_names=CLASES_MAP.values()))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASES_MAP.values(), yticklabels=CLASES_MAP.values())
plt.title(f"Matriz de Confusión Híbrida\n(Acc: {acc:.2%})")
plt.tight_layout()
ruta_matriz = os.path.join(DIR_IMG, f"grafico_matriz_confusion_{MODEL_NAME}.png")
plt.savefig(ruta_matriz)

# ==============================================================================
# [7] REPORTE DE PREDICCIONES (EXCEL DETALLADO)
# ==============================================================================
print("\n[5] Generando reporte de predicciones...")
filas = []
probs = nb_model.predict_proba(X_test)
X_test_csr = X_test.tocsr() 

for i in range(X_test.shape[0]):
    row = X_test_csr[i]
    feats = all_feature_names[row.indices]
    
    reglas = [f.replace("R:", "") for f in feats if f.startswith("R:")]
    palabras = [f.replace("W:", "") for f in feats if f.startswith("W:")]
    
    filas.append({
        "Texto": textos[LIMIT_TRAIN+i],
        "Clase_REAL": CLASES_MAP[y_test[i]],
        "Prediccion_MODELO": CLASES_MAP[y_pred[i]],
        "Confianza": round(np.max(probs[i]), 4),
        "Acerto": "SI" if y_test[i] == y_pred[i] else "NO",
        "Reglas_Activadas": str(reglas),
        "Palabras_Detectadas": str(palabras[:10])
    })

ruta_preds = os.path.join(DIR_EXCEL, f"reporte_predicciones_{MODEL_NAME}.xlsx")
pd.DataFrame(filas).to_excel(ruta_preds, index=False)

# ==============================================================================
# [8] ANÁLISIS SHAP QLIK-STYLE
# ==============================================================================
if SHAP_AVAILABLE:
    print(f"\n[6] Calculando SHAP para Ranking Qlik ({X_test.shape[0]} casos)...")
    
    try:
        X_test_explain = X_test.toarray() if hasattr(X_test, "toarray") else X_test
        background = np.zeros((1, X_test.shape[1]))
        explainer = shap.KernelExplainer(nb_model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_explain, nsamples=100)
        
        ranking_global = {k: {} for k in CLASES_MAP.keys()}
        
        es_lista = isinstance(shap_values, list)
        es_3d = not es_lista and len(shap_values.shape) == 3

        for i in range(X_test_explain.shape[0]):
            indices_activos = np.where(X_test_explain[i] > 0)[0]
            for idx in indices_activos:
                feat_raw = all_feature_names[idx]
                palabra_clean = feat_raw.split(":", 1)[1] if ":" in feat_raw else feat_raw
                clave = palabra_clean # Mantiene el texto procesado original
                
                for clase_idx in CLASES_MAP.keys():
                    if clave not in ranking_global[clase_idx]:
                        ranking_global[clase_idx][clave] = {"shap": 0.0, "count": 0}
                    
                    val_raw = 0.0
                    try:
                        if es_lista: val_raw = shap_values[clase_idx][i, idx]
                        elif es_3d: val_raw = shap_values[i, idx, clase_idx]
                        else: val_raw = shap_values[i, idx]
                        val_clean = abs(float(val_raw))
                    except: val_clean = 0.0

                    ranking_global[clase_idx][clave]["shap"] += val_clean
                    ranking_global[clase_idx][clave]["count"] += 1

        # Exportar Ranking Excel
        ruta_ranking = os.path.join(DIR_EXCEL, f"ranking_con_frecuencia_{MODEL_NAME}.xlsx")
        with pd.ExcelWriter(ruta_ranking) as writer:
            for clase_idx, nombre_clase in CLASES_MAP.items():
                df_r = pd.DataFrame(ranking_global[clase_idx]).T.reset_index()
                if not df_r.empty:
                    df_r.columns = ["Palabra", "Impacto_SHAP", "Frecuencia"]
                    df_r = df_r.sort_values("Impacto_SHAP", ascending=False)
                    df_r.to_excel(writer, sheet_name=nombre_clase[:30].replace("/","-"), index=False)

        # Gráfico Qlik
        todas_palabras = set()
        for c in ranking_global: todas_palabras.update(ranking_global[c].keys())
        data_viz = {nombre: [ranking_global[idx].get(p, {'shap': 0.0})['shap'] for p in todas_palabras] for idx, nombre in CLASES_MAP.items()}
        df_viz = pd.DataFrame(data_viz, index=list(todas_palabras))
        df_viz['TOTAL'] = df_viz.sum(axis=1)
        df_plot = df_viz.sort_values('TOTAL', ascending=False).head(TOP_K_PLOT).drop(columns=['TOTAL']).iloc[::-1]
        
        if not df_plot.empty:
            plt.figure(figsize=(12, 10))
            df_plot.plot(kind='barh', stacked=True, width=0.8, figsize=(12, 10), colormap='viridis', edgecolor='white')
            plt.title(f"Importancia Global por Palabra (SHAP) - {MODEL_NAME.upper()}")
            plt.tight_layout()
            plt.savefig(os.path.join(DIR_IMG, f"grafico_qlik_shap_{MODEL_NAME}.png"))
            plt.close()

    except Exception as e:
        print(f"\n❌ Error SHAP: {e}")

print("-" * 60)
print(f"✅ FIN DEL PROCESO: {datetime.datetime.now()}")

# ==============================================================================
# [9] EXPERIMENTO DE SENSIBILIDAD HÍBRIDA (CORREGIDO)
# ==============================================================================
if SHAP_AVAILABLE and 'ranking_global' in locals():
    print("\n[7] Iniciando Experimento de Sensibilidad Híbrida...")
    
    TOP_N_REGEX = 0
    TOP_N_WORDS = 30

    imp_reglas = {}
    imp_palabras = {}

    for clase in ranking_global:
        for feat, info in ranking_global[clase].items():
            # Identificamos si es Regla o Palabra buscando en el array original de nombres
            es_regex = any(f == f"R:{feat}" for f in all_feature_names)
            if es_regex:
                imp_reglas[feat] = imp_reglas.get(feat, 0) + info["shap"]
            else:
                imp_palabras[feat] = imp_palabras.get(feat, 0) + info["shap"]

    top_regex_to_mask = [f for f, s in sorted(imp_reglas.items(), key=lambda x: x[1], reverse=True)[:TOP_N_REGEX]]
    top_words_to_mask = [f for f, s in sorted(imp_palabras.items(), key=lambda x: x[1], reverse=True)[:TOP_N_WORDS]]

    print(f"--> Reglas clave: {top_regex_to_mask}")
    print(f"--> Palabras clave: {top_words_to_mask}")

    # 3. Función de Enmascaramiento Dual (Versión Segura)
    def mask_hybrid_text(text, rules_names, words):
        # A. Enmascarar Palabras (Canal BoW) - Usamos \b para palabras exactas
        for w in words:
            text = re.sub(rf'\b{w}\b', '[MASK_W]', text, flags=re.IGNORECASE)
        
        # B. Enmascarar Reglas (Canal Regex) 
        # IMPORTANTE: Aquí buscamos el patrón que la regla representa originalmente
        for r_name in rules_names:
            # Buscamos el objeto regex compilado que corresponde a este nombre
            for obj_reg in lista_reglas_obj:
                # Comparamos el nombre limpio de la regla
                # Si tu lista_reglas_obj y nombres_reglas están sincronizados:
                idx = nombres_reglas.index(f"R:{r_name}")
                regex_obj = lista_reglas_obj[idx]
                # Reemplazamos lo que encuentre la regex original por un tag neutro
                text = regex_obj.sub('[MASK_R]', text)
                break
        return text

    # 4. Crear Dataset Enmascarado
    textos_test_orig = textos[LIMIT_TRAIN:]
    y_test_labels = y[LIMIT_TRAIN:]
    
    print("-> Aplicando enmascaramiento sobre set de prueba...")
    textos_masked = [mask_hybrid_text(t, top_regex_to_mask, top_words_to_mask) for t in textos_test_orig]

    # 5. Re-vectorización (Crucial para modelos híbridos)
    print("-> Re-calculando matrices híbridas enmascaradas...")
    X_regex_masked = vectorizar_regex(textos_masked, lista_reglas_obj)
    X_regex_masked_sparse = csr_matrix(X_regex_masked)
    X_bow_masked = vectorizer_bow.transform(textos_masked)

    X_test_masked = hstack([X_regex_masked_sparse, X_bow_masked])

    # 6. Evaluación
    y_pred_masked = nb_model.predict(X_test_masked)
    acc_masked = accuracy_score(y_test_labels, y_pred_masked)
    f1_masked = f1_score(y_test_labels, y_pred_masked, average='weighted')

    # Reporte Final
    print("\n" + "="*60)
    print(f"SENSIBILIDAD HÍBRIDA: {MODEL_NAME}")
    print("="*60)
    print(f"ACCURACY ORIGINAL:   {acc:.4f}")
    print(f"ACCURACY MASKED:     {acc_masked:.4f}")
    print(f"CAÍDA DE PRECISIÓN:  {((acc - acc_masked) / acc) * 100:.2f}%")
    print("-" * 60)
    print(f"F1-SCORE ORIGINAL:   {f1:.4f}")
    print(f"F1-SCORE MASKED:     {f1_masked:.4f}")
    print("="*60)

    # Guardar en JSON
    res = {
        "top_regex": top_regex_to_mask,
        "top_words": top_words_to_mask,
        "metrics": {"acc_orig": float(acc), "acc_masked": float(acc_masked), "f1_orig": float(f1), "f1_masked": float(f1_masked)}
    }
    with open(os.path.join(DIR_EXCEL, f"experimento_sensibilidad_{MODEL_NAME}.json"), "w") as f:
        json.dump(res, f, indent=4)