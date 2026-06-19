import sys
import os
import json
import re
import pickle
import unicodedata
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning Profesional
import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup 

# Métricas Avanzadas
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import shuffle, class_weight

# ==============================================================================
# [0] CONFIGURACIÓN
# ==============================================================================
MODEL_NAME = "regexBioBERT"
BERT_CKPT = "dmis-lab/biobert-base-cased-v1.2"
DATASET_NAME = "snippets_procesados.pkl" 
ARCHIVO_JSON_REGLAS = "test_regexes.json"
ARCHIVO_JSON_MAPS = "test_maps.json"

LIMIT_TRAIN = 932
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 6 
LEARNING_RATE = 2e-5
TOP_K_PLOT = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas
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
    print("⚠️ SHAP no disponible.")

CLASES_MAP = {
    0: "Cardiovascular",
    1: "Digestive_system",
    2: "Neoplasms"
}
plt.style.use('ggplot')

# ==============================================================================
# [1] UTILERÍA DE BÚSQUEDA Y CARGA DE REGLAS
# ==============================================================================
def encontrar_archivo(nombre):
    rutas = [nombre, os.path.join("out", nombre), os.path.join("out", "regex_rules", nombre)]
    for r in rutas:
        if os.path.exists(r): return r
    return None

ruta_reglas = encontrar_archivo(ARCHIVO_JSON_REGLAS)
ruta_maps = encontrar_archivo(ARCHIVO_JSON_MAPS)

if not ruta_reglas or not ruta_maps: 
    print(f"❌ ERROR: No se encontraron los archivos JSON de reglas.")
    sys.exit(1)

with open(ruta_reglas, "r", encoding="utf-8") as f: raw_rules = json.load(f)
with open(ruta_maps, "r", encoding="utf-8") as f: raw_maps = json.load(f)
mapa_tokens = raw_maps.get("pattern2token", {})

lista_reglas_obj = []
nombres_reglas = []
for patron_str in raw_rules.keys():
    try:
        regex = re.compile(patron_str, re.IGNORECASE)
        lista_reglas_obj.append(regex)
        # GUARDAMOS EL PATRÓN ORIGINAL (esto es lo que pediste para que se vea como en SVM)
        nombres_reglas.append(f"R:{patron_str}") 
    except: pass

def extraer_activaciones_regex(textos_list, reglas_obj):
    activaciones = np.zeros((len(textos_list), len(reglas_obj)), dtype=np.float32)
    for i, t in enumerate(textos_list):
        for j, r in enumerate(reglas_obj):
            if r.search(str(t)): activaciones[i, j] = 1.0
    return activaciones

# ==============================================================================
# [2] PREPARACIÓN DE DATOS
# ==============================================================================
with open(DATASET_NAME, "rb") as f: data_pkl = pickle.load(f)
data_pkl = shuffle(data_pkl, random_state=42)
textos = np.array([d[0] for d in data_pkl])
y_labels = np.array([d[1] for d in data_pkl]).astype(int)

regex_feats_all = extraer_activaciones_regex(textos, lista_reglas_obj)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labels[:LIMIT_TRAIN]), y=y_labels[:LIMIT_TRAIN])
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, regex_feats, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.regex_feats = regex_feats
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, item):
        encoding = self.tokenizer(str(self.texts[item]), add_special_tokens=True, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'regex_features': torch.tensor(self.regex_feats[item], dtype=torch.float),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(BERT_CKPT)
train_loader = DataLoader(MedicalDataset(textos[:LIMIT_TRAIN], y_labels[:LIMIT_TRAIN], regex_feats_all[:LIMIT_TRAIN], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MedicalDataset(textos[LIMIT_TRAIN:], y_labels[LIMIT_TRAIN:], regex_feats_all[LIMIT_TRAIN:], tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# ==============================================================================
# [3] ARQUITECTURA HÍBRIDA
# ==============================================================================
class HybridRegexModel(nn.Module):
    def __init__(self, n_classes, n_regex_features):
        super().__init__()
        self.biobert = AutoModel.from_pretrained(BERT_CKPT)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + n_regex_features, n_classes)
        
    def forward(self, input_ids, attention_mask, regex_features):
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined = torch.cat((pooled_output, regex_features), dim=1) 
        return self.classifier(self.dropout(combined))

model = HybridRegexModel(len(CLASES_MAP), len(lista_reglas_obj)).to(DEVICE)

# ==============================================================================
# [4] ENTRENAMIENTO
# ==============================================================================
print("\n[3] Entrenando Modelo Híbrido...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        ids, mask, r_feats = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['regex_features'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(ids, mask, r_feats)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

# ==============================================================================
# [5] EVALUACIÓN INTEGRAL (FORMATO PROFESIONAL)
# ==============================================================================
print("\n[5] Evaluación Final con Métricas Avanzadas...")
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for batch in test_loader:
        ids, mask, r_feats = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['regex_features'].to(DEVICE)
        outputs = model(ids, mask, r_feats)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        actuals.extend(batch['labels'].numpy())

acc = accuracy_score(actuals, preds)
f1_w = f1_score(actuals, preds, average='weighted')

print("\n" + "="*60)
print(f"REPORT CARD: {MODEL_NAME}")
print("="*60)
print(f"ACCURACY: {acc:.4f}")
print(f"F1-SCORE: {f1_w:.4f}")
print("="*60)
print(classification_report(actuals, preds, target_names=CLASES_MAP.values()))

# Guardar Matriz de Confusión
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(actuals, preds), annot=True, fmt='d', cmap='RdPu', 
            xticklabels=CLASES_MAP.values(), yticklabels=CLASES_MAP.values())
plt.title(f"Matriz de Confusión: {MODEL_NAME}")
plt.ylabel('Clase Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig(os.path.join(DIR_IMG, f"confusion_matrix_{MODEL_NAME}.png"))
plt.close()

# ==============================================================================
# [6] SHAP: RANKING LIMPIO Y EXCEL MULTI-CLASE
# ==============================================================================
if SHAP_AVAILABLE:
    print(f"\n[6] Generando Ranking SHAP Limpio (Sin Símbolos)...")
    try:
        STOPWORDS = set(["de", "la", "el", "en", "que", "y", "los", "un", "con", "por", "una", "para", "es", "al", "lo", "como", "mas", "o", "sus", "se", "este", "del", "las", "su", "si", "no", "a", "sin"])

        def predict_func(texts):
            model.eval()
            r_feats_internal = extraer_activaciones_regex(texts, lista_reglas_obj)
            inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            r_feats_internal = torch.tensor(r_feats_internal, dtype=torch.float).to(DEVICE)
            with torch.no_grad():
                logits = model(inputs['input_ids'], inputs['attention_mask'], r_feats_internal)
            return torch.softmax(logits, dim=1).cpu().numpy()

        masker = shap.maskers.Text(tokenizer=r"\W+")
        explainer = shap.Explainer(predict_func, masker, output_names=list(CLASES_MAP.values()))
        test_samples = textos[LIMIT_TRAIN : LIMIT_TRAIN + 40].tolist()
        shap_values = explainer(test_samples)

        ranking_global = {k: {} for k in CLASES_MAP.keys()}
        for i in range(len(test_samples)):
            palabras_muestra = shap_values[i].data 
            vals_muestra = shap_values[i].values 
            for c_idx in range(len(CLASES_MAP)):
                for f_idx, palabra in enumerate(palabras_muestra):
                    p_name = re.sub(r'[^a-záéíóúñ*\\?\[\]\s]', '', str(palabra).lower()).strip()
                    
                    if p_name and p_name not in STOPWORDS and len(p_name) > 2:
                        # Buscamos si este término dispararía alguna de nuestras reglas originales
                        # Comparamos contra la lista de nombres_reglas que ahora tiene los patrones
                        regla_encontrada = None
                        for name_r in nombres_reglas:
                            # name_r tiene el formato "R:patron..."
                            patron_puro = name_r.replace("R:", "")
                            if p_name in patron_puro or patron_puro in p_name:
                                regla_encontrada = patron_puro
                                break
                        
                        # Si es regla, usamos el patrón original. Si no, es palabra normal.
                        clave_final = regla_encontrada if regla_encontrada else p_name
                        es_regla_bool = True if regla_encontrada else False
                        
                        # Guardamos en el ranking (usaremos un prefijo interno para separar luego)
                        label_sh = f"R:{clave_final}" if es_regla_bool else f"W:{clave_final}"

                        impacto = float(np.abs(vals_muestra[f_idx, c_idx]))
                        if label_sh not in ranking_global[c_idx]:
                            ranking_global[c_idx][label_sh] = {"shap": 0.0, "count": 0}
                        ranking_global[c_idx][label_sh]["shap"] += impacto
                        ranking_global[c_idx][label_sh]["count"] += 1

        # EXPORTACIÓN A EXCEL (Una hoja por clase)
        ruta_ranking = os.path.join(DIR_EXCEL, f"ranking_shap_{MODEL_NAME}.xlsx")
        with pd.ExcelWriter(ruta_ranking) as writer:
            for c_idx, nombre_clase in CLASES_MAP.items():
                df_r = pd.DataFrame(ranking_global[c_idx]).T.reset_index()
                df_r = df_r.rename(columns={"index":"Palabra"}).sort_values("shap", ascending=False)
                df_r.to_excel(writer, sheet_name=nombre_clase[:30].replace("/","-"), index=False)

        # GRÁFICO QLIK
        all_unique_words = list(set([t for c in ranking_global for t in ranking_global[c]]))
        data_viz = {CLASES_MAP[c]: [ranking_global[c].get(t, {'shap':0.0})['shap'] for t in all_unique_words] for c in CLASES_MAP}
        df_viz = pd.DataFrame(data_viz, index=all_unique_words)
        
        # Ordenamos y tomamos el Top K
        df_plot = df_viz.assign(TOTAL=df_viz.sum(axis=1)).sort_values('TOTAL', ascending=False).head(TOP_K_PLOT).drop(columns=['TOTAL']).iloc[::-1]

        # --- AJUSTE VISUAL: TRUNCAR REGLAS LARGAS ---
        # Si la etiqueta mide más de 40 caracteres, le ponemos "..."
        nuevos_indices = []
        for label in df_plot.index:
            if len(label) > 40:
                nuevos_indices.append(label[:37] + "...")
            else:
                nuevos_indices.append(label)
        df_plot.index = nuevos_indices
        # --------------------------------------------

        plt.figure(figsize=(12, 10))
        df_plot.plot(kind='barh', stacked=True, width=0.8, figsize=(12, 10), colormap='viridis', edgecolor='white')
        plt.title(f"Importancia Global SHAP - {MODEL_NAME}", fontsize=14)
        plt.xlabel("Impacto Acumulado")
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_IMG, f"grafico_shap_{MODEL_NAME}.png"), dpi=300)
        plt.close()
        print(f"✅ Excel y Gráfico generados correctamente.")
    except Exception as e: print(f"❌ Error SHAP: {e}")

# ==============================================================================
# [7] EXPERIMENTO DE SENSIBILIDAD HÍBRIDA (REGEX + BIOBERT)
# ==============================================================================
if SHAP_AVAILABLE and 'ranking_global' in locals():
    print(f"\n[7] Iniciando Experimento de Sensibilidad Híbrida ({MODEL_NAME})...")
    
    TOP_N_REGEX = 30
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

    # ESTO GENERARÁ LA SALIDA QUE BUSCAS
    print(f"--> Reglas clave detectadas: {top_regex_to_mask}")
    print(f"--> Palabras clave detectadas: {top_words_to_mask}")

    def safe_mask_hybrid(text, rules_list, words_list):
        for w in words_list:
            text = re.sub(rf'\b{w}\b', '[MASK]', text, flags=re.IGNORECASE)
        for r_name in rules_list:
            try:
                idx = nombres_reglas.index(f"R:{r_name}")
                text = lista_reglas_obj[idx].sub('[MASK]', text)
            except: continue
        return text

    textos_test_orig = textos[LIMIT_TRAIN:]
    textos_masked = [safe_mask_hybrid(t, top_regex_to_mask, top_words_to_mask) for t in textos_test_orig]
    regex_masked = extraer_activaciones_regex(textos_masked, lista_reglas_obj)

    masked_loader = DataLoader(MedicalDataset(textos_masked, y_labels[LIMIT_TRAIN:], regex_masked, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    model.eval()
    preds_m = []
    with torch.no_grad():
        for b in masked_loader:
            outputs = model(b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE), b['regex_features'].to(DEVICE))
            preds_m.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    acc_m = accuracy_score(y_labels[LIMIT_TRAIN:], preds_m)
    f1_m = f1_score(y_labels[LIMIT_TRAIN:], preds_m, average='weighted')

    print("\n" + "="*60)
    print("COMPARATIVA DE IMPACTO (ORIGINAL VS ENMASCARADO)")
    print("="*60)
    print(f"ACCURACY ORIGINAL:   {acc:.4f}")
    print(f"ACCURACY MASKED:     {acc_m:.4f}")
    print(f"CAÍDA DE PRECISIÓN:  {((acc - acc_m) / acc):.2%}")
    print("-" * 60)
    print(f"F1-SCORE ORIGINAL:   {f1_w:.4f}")
    print(f"F1-SCORE MASKED:     {f1_m:.4f}")
    print("="*60)
    
    # Guardar JSON
    res_sens = {"original_acc": float(acc), "masked_acc": float(acc_m), "drop": float((acc-acc_m)/acc)}
    with open(os.path.join(DIR_EXCEL, "resultado_sensibilidad.json"), "w") as f:
        json.dump(res_sens, f, indent=4)