import sys
import os
os.environ["WANDB_DISABLED"] = "true"
import json
import re
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import unicodedata

# --- PARCHE DE COMPATIBILIDAD SETFIT ---
from sentence_transformers import trainer as st_trainer_module
original_compute_loss = st_trainer_module.SentenceTransformerTrainer.compute_loss

def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    return original_compute_loss(self, model, inputs, return_outputs=return_outputs)

st_trainer_module.SentenceTransformerTrainer.compute_loss = patched_compute_loss
# --------------------------------------------

from setfit import SetFitModel, SetFitTrainer
from sentence_transformers import losses
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

# ==============================================================================
# [0] CONFIGURACIÓN
# ==============================================================================
MODEL_NAME = "regexSetFit"
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
DATASET_NAME = "snippets_procesados.pkl" 
ARCHIVO_JSON_REGLAS = "TEST_regexes.json"
ARCHIVO_JSON_MAPS = "TEST_maps.json"

LIMIT_TRAIN = 892
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIR_EXCEL = f"REPORTES_EXCEL_{MODEL_NAME}"
DIR_IMG = f"GRAFICOS_{MODEL_NAME}"
DIR_LOGS = f"LOGS_{MODEL_NAME}"

for d in [DIR_EXCEL, DIR_IMG, DIR_LOGS]: os.makedirs(d, exist_ok=True)

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
    def flush(self): self.terminal.flush()
    def isatty(self): return False

sys.stdout = Logger()
sys.stderr = sys.stdout 

print(f"--- INICIO PROCESO HÍBRIDO {MODEL_NAME}: {datetime.datetime.now()} ---")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

CLASES_MAP = {0: "Enfermedad cardiovascular", 
              1: "Enfermedad endocrina o metabólica", 
              2: "Trastorno mental o del comportamiento"}
plt.style.use('ggplot')

# ==============================================================================
# [1] UTILERÍA DE REGEX (CARGA Y EXTRACCIÓN)
# ==============================================================================
def encontrar_archivo(nombre):
    rutas = [nombre, os.path.join("out", nombre), os.path.join("out", "regex_rules", nombre)]
    for r in rutas:
        if os.path.exists(r): return r
    return None

ruta_reglas = encontrar_archivo(ARCHIVO_JSON_REGLAS)
if not ruta_reglas: 
    print("❌ Falta archivo de reglas."); sys.exit(1)

with open(ruta_reglas, "r", encoding="utf-8") as f: raw_rules = json.load(f)

lista_reglas_obj = []
nombres_reglas = []
for patron_str in raw_rules.keys():
    try:
        regex = re.compile(patron_str, re.IGNORECASE)
        lista_reglas_obj.append(regex)
        nombres_reglas.append(f"R:{patron_str}") 
    except: pass

def extraer_activaciones_regex(textos_list, reglas_obj):
    m = np.zeros((len(textos_list), len(reglas_obj)), dtype=np.float32)
    for i, t in enumerate(textos_list):
        for j, r in enumerate(reglas_obj):
            if r.search(str(t)): m[i, j] = 1.0
    return m

# ==============================================================================
# [2] CARGA Y ENTRENAMIENTO DE SETFIT (ADAPTACIÓN DE EMBEDDINGS)
# ==============================================================================
print("\n[1] Cargando datos y entrenando SetFit Base...")
with open(DATASET_NAME, "rb") as f: data_pkl = pickle.load(f)
data_pkl = shuffle(data_pkl, random_state=42)
textos = np.array([d[0] for d in data_pkl])
y = np.array([d[1] for d in data_pkl]).astype(int)

train_ds = Dataset.from_dict({"text": textos[:LIMIT_TRAIN], "label": y[:LIMIT_TRAIN]})
test_ds = Dataset.from_dict({"text": textos[LIMIT_TRAIN:], "label": y[LIMIT_TRAIN:]})

model_setfit = SetFitModel.from_pretrained(BASE_MODEL, device=DEVICE)
trainer = SetFitTrainer(
    model=model_setfit, train_dataset=train_ds, eval_dataset=test_ds,
    loss_class=losses.CosineSimilarityLoss, batch_size=16, num_epochs=1,
    column_mapping={"text": "text", "label": "label"}
)
trainer.train()

# ==============================================================================
# [3] CONSTRUCCIÓN DE LA CABEZA HÍBRIDA (EMBEDDINGS + REGEX)
# ==============================================================================
print("\n[2] Generando Características Híbridas...")
X_emb_all = model_setfit.model_body.encode(textos, show_progress_bar=True)
X_reg_all = extraer_activaciones_regex(textos, lista_reglas_obj)
X_final = np.hstack([X_emb_all, X_reg_all])

X_train, y_train = X_final[:LIMIT_TRAIN], y[:LIMIT_TRAIN]
X_test, y_test = X_final[LIMIT_TRAIN:], y[LIMIT_TRAIN:]

print("\n[3] Entrenando Clasificador Híbrido Final...")
hybrid_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
hybrid_clf.fit(X_train, y_train)

# ==============================================================================
# [4] EVALUACIÓN
# ==============================================================================
y_pred = hybrid_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*60)
print(f"REPORT CARD HÍBRIDO: {MODEL_NAME}")
print("="*60)
print(f"ACCURACY: {acc:.4f} | F1-SCORE: {f1:.4f}")
print("="*60)
print(classification_report(y_test, y_pred, target_names=CLASES_MAP.values()))

# --- GUARDADO MATRIZ (ORDEN CORREGIDO) ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=CLASES_MAP.values(), yticklabels=CLASES_MAP.values())
plt.title(f"Matriz {MODEL_NAME}\n(Acc: {acc:.2%})")
plt.tight_layout()

# Primero savefig, luego show
ruta_matriz = os.path.join(DIR_IMG, f"grafico_matriz_confusion_{MODEL_NAME}.png")
plt.savefig(ruta_matriz, dpi=300)
plt.show() 
print(f"✅ Matriz guardada en: {ruta_matriz}")

# ==============================================================================
# [5] SHAP: RANKING LIMPIO (SEPARACIÓN R: Y W:)
# ==============================================================================
if SHAP_AVAILABLE:
    print(f"\n[5] Generando Ranking SHAP (Identificando Reglas y Palabras)...")
    try:
        TOP_K_PLOT = 30
        STOPWORDS = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo'])

        def predict_func_hybrid(texts):
            # Función puente para SHAP: de texto crudo a probabilidad híbrida
            e = model_setfit.model_body.encode(texts, show_progress_bar=False)
            r = extraer_activaciones_regex(texts, lista_reglas_obj)
            return hybrid_clf.predict_proba(np.hstack([e, r]))

        masker = shap.maskers.Text(tokenizer=r"\W+")
        explainer = shap.Explainer(predict_func_hybrid, masker, output_names=list(CLASES_MAP.values()))
        shap_values = explainer(textos[LIMIT_TRAIN:LIMIT_TRAIN+40].tolist())

        ranking_global = {k: {} for k in CLASES_MAP.keys()}
        for i in range(len(shap_values)):
            for c_idx in range(len(CLASES_MAP)):
                for f_idx, palabra in enumerate(shap_values[i].data):
                    p_name = re.sub(r'[^a-záéíóúñ*\\?\[\]\s]', '', str(palabra).lower()).strip()
                    if p_name and p_name not in STOPWORDS and len(p_name) > 2:
                        # Lógica de BioBERT: ¿Es una regla o una palabra?
                        regla_encontrada = None
                        for name_r in nombres_reglas:
                            patron = name_r.replace("R:", "")
                            if p_name in patron or patron in p_name:
                                regla_encontrada = patron; break
                        
                        clave = f"R:{regla_encontrada}" if regla_encontrada else f"W:{p_name}"
                        impacto = float(np.abs(shap_values[i].values[f_idx, c_idx]))
                        
                        if clave not in ranking_global[c_idx]:
                            ranking_global[c_idx][clave] = {"shap": 0.0, "count": 0}
                        ranking_global[c_idx][clave]["shap"] += impacto
                        ranking_global[c_idx][clave]["count"] += 1
        
        # --- GUARDADO GRÁFICO QLIK ---
        all_unique_feats = list(set([t for c in ranking_global for t in ranking_global[c]]))
        data_viz = {CLASES_MAP[c]: [ranking_global[c].get(t, {'shap':0.0})['shap'] for t in all_unique_feats] for c in CLASES_MAP}
        df_viz = pd.DataFrame(data_viz, index=all_unique_feats)
        df_viz['TOTAL'] = df_viz.sum(axis=1)
        df_plot = df_viz.sort_values('TOTAL', ascending=False).head(TOP_K_PLOT).drop(columns=['TOTAL']).iloc[::-1]

        plt.figure(figsize=(12, 10))
        df_plot.plot(kind='barh', stacked=True, width=0.8, figsize=(12, 10), colormap='viridis', edgecolor='white')
        plt.title(f"Impacto Global SHAP - {MODEL_NAME}")
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_IMG, f"grafico_qlik_shap_{MODEL_NAME}.png"))
        plt.show()
        
        # Guardar Excel de Ranking
        ruta_ranking = os.path.join(DIR_EXCEL, f"ranking_shap_{MODEL_NAME}.xlsx")
        with pd.ExcelWriter(ruta_ranking) as writer:
            for c_idx, nombre in CLASES_MAP.items():
                df_r = pd.DataFrame(ranking_global[c_idx]).T.reset_index()
                df_r.columns = ["Feature", "Impacto_SHAP", "Frecuencia"]
                df_r.sort_values("Impacto_SHAP", ascending=False).to_excel(writer, sheet_name=nombre[:30], index=False)

    except Exception as e: print(f"❌ Error SHAP: {e}")

# ==============================================================================
# [6] PARTE 7: EXPERIMENTO DE SENSIBILIDAD HÍBRIDA (IMPLEMENTACIÓN SOLICITADA)
# ==============================================================================
if SHAP_AVAILABLE and 'ranking_global' in locals():
    print(f"\n[6] Iniciando Experimento de Sensibilidad Híbrida ({MODEL_NAME})...")
    
    # Parámetros solicitados
    TOP_N_REGEX = 0
    TOP_N_WORDS = 30

    imp_reglas, imp_palabras = {}, {}
    for clase in ranking_global:
        for feat_raw, info in ranking_global[clase].items():
            if feat_raw.startswith("R:"):
                nombre = feat_raw.replace("R:", "")
                imp_reglas[nombre] = imp_reglas.get(nombre, 0) + info["shap"]
            else:
                nombre = feat_raw.replace("W:", "")
                imp_palabras[nombre] = imp_palabras.get(nombre, 0) + info["shap"]

    top_regex_to_mask = [f for f, s in sorted(imp_reglas.items(), key=lambda x: x[1], reverse=True)[:TOP_N_REGEX]]
    top_words_to_mask = [f for f, s in sorted(imp_palabras.items(), key=lambda x: x[1], reverse=True)[:TOP_N_WORDS]]

    print(f"--> Reglas clave detectadas ({len(top_regex_to_mask)}): {top_regex_to_mask}")
    print(f"--> Palabras clave detectadas ({len(top_words_to_mask)}): {top_words_to_mask}")

    def safe_mask_hybrid(text, rules_list, words_list):
        # A. Enmascarar Palabras
        for w in words_list:
            text = re.sub(rf'\b{w}\b', '[MASK]', text, flags=re.IGNORECASE)
        # B. Enmascarar Reglas
        for r_patron in rules_list:
            try:
                # Buscamos el objeto regex que corresponde a este patrón
                for obj in lista_reglas_obj:
                    if obj.pattern == r_patron:
                        text = obj.sub('[MASK]', text)
                        break
            except: continue
        return text

    textos_test_orig = textos[LIMIT_TRAIN:]
    print("-> Aplicando enmascaramiento dual sobre test...")
    textos_masked = [safe_mask_hybrid(t, top_regex_to_mask, top_words_to_mask) for t in textos_test_orig]
    
    # Re-vectorización para el modelo híbrido
    X_emb_m = model_setfit.model_body.encode(textos_masked, show_progress_bar=False)
    X_reg_m = extraer_activaciones_regex(textos_masked, lista_reglas_obj)
    X_test_masked = np.hstack([X_emb_m, X_reg_m])

    y_pred_m = hybrid_clf.predict(X_test_masked)
    acc_m = accuracy_score(y_test, y_pred_m)
    f1_m = f1_score(y_test, y_pred_m, average='weighted')

    print("\n" + "="*60)
    print(f"SENSIBILIDAD HÍBRIDA: {MODEL_NAME}")
    print("="*60)
    print(f"ACCURACY ORIGINAL:   {acc:.4f}")
    print(f"ACCURACY MASKED:     {acc_m:.4f}")
    print(f"CAÍDA DE PRECISIÓN:  {((acc - acc_m) / acc):.2%}")
    print("-" * 60)
    print(f"F1-SCORE ORIGINAL:   {f1:.4f}")
    print(f"F1-SCORE MASKED:     {f1_m:.4f}")
    print("="*60)
    
    res_sens = {
        "top_regex": top_regex_to_mask, "top_words": top_words_to_mask,
        "metrics": {"acc_orig": float(acc), "acc_masked": float(acc_m), "drop": float((acc-acc_m)/acc)}
    }
    with open(os.path.join(DIR_EXCEL, "experimento_sensibilidad_hibrida.json"), "w") as f:
        json.dump(res_sens, f, indent=4)

print(f"\n--- FIN: {datetime.datetime.now()} ---")

# [1] LIMPIEZA TOTAL (Para evitar conflictos con lo que Colab trae por defecto)
#!pip uninstall -y transformers setfit peft sentence-transformers huggingface_hub -q

# [2] INSTALACIÓN DE VERSIONES COMPATIBLES (La combinación que nos dio el 98%)
#!pip install transformers==4.40.2 setfit==1.0.3 huggingface_hub==0.22.2 datasets -q

# [3] LIBRERÍAS DE SOPORTE
#!pip install pandas matplotlib seaborn scikit-learn openpyxl shap -q

# Reiniciar el entorno manualmente, volver a montar y volver a ejecutar.