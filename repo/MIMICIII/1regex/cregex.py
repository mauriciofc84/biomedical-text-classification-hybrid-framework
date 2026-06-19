import numpy as np
import copy
import gc
import re
import os
import json
from collections import defaultdict, Counter
from sklearn.model_selection import PredefinedSplit

# === IMPORTACIÓN DE MÓDULOS DEL PROYECTO ===
from utils import *
from fregex import FREGEX
from bert import BERT
from mysetfit import SETFIT

# === CLASE DE APOYO: MODELO DE VOTACIÓN CON RESPALDO (FALLBACK) ===
class RegexVotingModel:
    def __init__(self, n_classes, rule_to_class_map, tokens, priors=None):
        self.n_classes = n_classes
        self.W = np.zeros((len(tokens), n_classes))
        
        if priors is None:
            self.priors = np.ones(n_classes) / n_classes
        else:
            self.priors = np.array(priors)
        
        for i, token in enumerate(tokens):
            if token in rule_to_class_map:
                val = rule_to_class_map[token]
                classes_to_vote = []
                if isinstance(val, list) or isinstance(val, np.ndarray):
                    classes_to_vote = val
                else:
                    classes_to_vote = [val]
                
                for item in classes_to_vote:
                    try:
                        c = int(item)
                        if c < n_classes:
                            self.W[i, c] = 1
                    except: continue
    
    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        votes = X.dot(self.W)
        row_sums = votes.sum(axis=1)
        mask_no_votes = (row_sums == 0)
        
        probs = np.divide(votes, row_sums[:, None], 
                          out=np.zeros_like(votes), 
                          where=row_sums[:, None] != 0)
        
        if np.any(mask_no_votes):
            probs[mask_no_votes] = self.priors
            
        return probs

# === CLASE PRINCIPAL CREGEX ===
class CREGEX(object):
    def __init__(self, 
                 FILENAME, MODEL_NAMES, N_CLASSES, THR_CONF_CLF_opt=True,
                 PROBS_THR=np.arange(0.50, 1, 0.05), CLFS=1,
                 NGRAM_MIN=1, pnumbers=False,
                 gap_cmb=False, whitespaces=False, lexicon={}, HYPERPARAMS={}, SEED=42):
        
        self.FILENAME = FILENAME
        if '*' in MODEL_NAMES:
            clfs, _ = MODEL_NAMES.split('*')
        else:
            clfs = MODEL_NAMES
        self.MODEL_NAMES = clfs.split('.')
        
        self.N_CLASSES = N_CLASSES
        self.NGRAM_MIN = NGRAM_MIN
        self.lexicon = lexicon
        self.SEED = SEED
        self.HYPERPARAMS = HYPERPARAMS
        
        self.regexes = {}
        self.labeled_regexes = {}
        self.kw = []
        self.models = {}
        
        self.THR_CONF_CLF_opt = THR_CONF_CLF_opt
        self.THR_CONF_CLFS = {}
        self.PROBS_THR = PROBS_THR

    def _manual_vectorize(self, X_text, features_sorted):
        try:
            compiled_rules = [re.compile(p, re.IGNORECASE) for p in features_sorted]
        except Exception as e:
            print(f"[ERROR] Fallo al compilar regex: {e}")
            compiled_rules = []

        matrix = np.zeros((len(X_text), len(features_sorted)))
        for i, text in enumerate(X_text):
            text_str = str(text)
            for j, r in enumerate(compiled_rules):
                if r.search(text_str):
                    matrix[i, j] = 1
        return matrix

    def fit(self, X, y, X_val, y_val):
        self.regexes = {}
        self.labeled_regexes = {}
        self.models = {}

        print('CREGEX...fit: Minando Regex (FREGEX)')
        fregex = FREGEX(X, y, self.FILENAME)
        fregex.fit()
        self.regexes.update(fregex.transform())

        if self.FILENAME in self.lexicon:
            self.kw = copy.deepcopy(self.lexicon[self.FILENAME])
        else:
            self.kw = ["*"]

        self.pattern2token = copy.deepcopy(fregex.pattern2token)
        self.token2pattern = copy.deepcopy(fregex.token2pattern)
        self.tokens2pos = copy.deepcopy(fregex.tokens2pos) 
        
        self.regexes, self.regex2class = get_classes_regexes(self.regexes, y, self.tokens2pos)

        # ==============================================================================
        # [MODIFICACIÓN CRÍTICA] FILTRADO SEMÁNTICO Y ACTUALIZACIÓN DE DISCO
        # ==============================================================================
        if self.kw == ["*"]:
            self.labeled_regexes = {}
            print(f"[INFO] Filtrando reglas por confianza (THR_CONF) >= {THR_CONF}...")
            
            # --- MODIFICADO: SE COMENTÓ LA LISTA DE STOPWORDS EXTRA ---
            # STOP_EXTRA = ["OTROS", "OTRAS", "GENERAL", "TOTAL", "PARCIAL", "DE", "LA", "EL", "EN", "QUE"]
            
            count_kept = 0
            
            for key in self.regexes:
                label, conf = self.regex2class[key]
                token_origin = self.pattern2token.get(key, "").strip()
                
                # --- FILTROS DE CALIDAD ---
                # A. Filtro de Confianza (Estadístico)
                if conf < THR_CONF:
                    continue
                    
                # B. Filtro de Números (Mata reglas como "\d+")
#                if re.search(r'\d', token_origin): 
#                    continue
                    
                # C. Filtro de Longitud (Mata reglas cortas o letras sueltas)
#                if len(token_origin) < 4:
#                    continue
                    
                # --- MODIFICADO: SE ELIMINÓ EL FILTRO DE STOPWORDS EXTRA ---
                # D. Filtro de Stopwords Extra
                # if token_origin.upper() in STOP_EXTRA:
                #     continue
                # ---------------------------

                self.labeled_regexes[key] = self.regexes[key]
                count_kept += 1
            
            print(f"[INFO] Reglas sobrevivientes: {count_kept} (Descartadas: {len(self.regexes) - count_kept})")

            # --- [NUEVO] SOBRESCRIBIR ARCHIVOS EN DISCO CON LA VERSIÓN LIMPIA ---
            try:
                out_dir = os.path.join(os.getcwd(), "out", "regex_rules")
                
                # 1. Generar JSON Limpio
                rules_simple_clean = {}
                for rgx, payload in self.labeled_regexes.items():
                    pos_arr = payload[0]
                    num_arr = payload[1]
                    rules_simple_clean[rgx] = {
                        "positions": np.asarray(pos_arr, dtype=int).tolist(),
                        "numbers": num_arr.tolist() if isinstance(num_arr, np.ndarray) and num_arr.size > 0 else []
                    }

                # Sobrescribir JSON
                rules_path = os.path.join(out_dir, f"{self.FILENAME}_regexes.json")
                with open(rules_path, "w", encoding="utf-8") as f:
                    json.dump(rules_simple_clean, f, ensure_ascii=False, indent=2)

                # 2. Generar TXT Limpio
                txt_path = os.path.join(out_dir, f"{self.FILENAME}_regexes.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    # Ordenar por cobertura (número de apariciones)
                    sorted_rules = sorted(self.labeled_regexes.items(), key=lambda x: len(x[1][0]), reverse=True)
                    for rgx, payload in sorted_rules:
                        coverage = len(payload[0])
                        f.write(f"{coverage:5d}  {rgx}\n")

                print(f"[INFO] ✅ ARCHIVOS EN DISCO ACTUALIZADOS: Se guardaron solo las reglas limpias.")

            except Exception as e:
                print(f"[WARN] No se pudo actualizar el archivo en disco: {e}")
            
        else:
            labeled_regexes, _, _ = get_filtered_regexes(
                self.regexes, y, self.kw, self.pattern2token, self.regex2class)
            self.labeled_regexes.update(labeled_regexes)
        
        gc.collect()

        keys = copy.deepcopy(list(self.regexes.keys()))
        for key in keys:
            if key not in self.labeled_regexes:
                self.regexes.pop(key)
                if key in self.regex2class:
                    self.regex2class.pop(key)

        # === ENTRENAMIENTO MODELOS ===
        for MODEL_NAME in self.MODEL_NAMES:
            print(MODEL_NAME + '...fit')
            tokens = None
            opt = None
            regexes_aux = None
            model = None
            
            if 'random' not in MODEL_NAME:
                seed_everything()
                
                regexes_aux = copy.deepcopy(self.labeled_regexes)
                
                if MODEL_NAME.startswith("regex"):
                    print("[INFO] Generando matriz manualmente para RegexVotingModel...")
                    
                    all_features = sorted(list(regexes_aux.keys()))
                    tokens = all_features 
                    opt = 'manual'        
                    
                    X_l_aux = self._manual_vectorize(X, all_features)
                    X_val_aux = self._manual_vectorize(X_val, all_features)
                    y_l_aux = copy.deepcopy(y)
                    y_val_aux = copy.deepcopy(y_val)
                    
                    counts = Counter(y)
                    total_samples = len(y)
                    priors = np.zeros(self.N_CLASSES)
                    for c in range(self.N_CLASSES):
                        priors[c] = counts.get(c, 0) / total_samples
                    
                    print(f"[INFO] Priors (Clase por defecto): {priors}")

                    model = RegexVotingModel(self.N_CLASSES, self.regex2class, all_features, priors)
                    
                elif 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME:
                    m = re.search(r'(\d+)$', MODEL_NAME)
                    NGRAM_SIZE = int(m.group(1)) if m else self.NGRAM_MIN
                    tokens = n_grams(X, NGRAM_SIZE)
                    opt = False
                    
                    X_l_aux = copy.deepcopy(get_matrix(tokens, X, regexes_aux, opt))
                    X_val_aux = copy.deepcopy(get_matrix(tokens, X_val, regexes_aux, opt))
                    y_l_aux = copy.deepcopy(y)
                    y_val_aux = copy.deepcopy(y_val)
                    
                    X_train_val = np.vstack((X_l_aux, X_val_aux))
                    y_train_val = np.hstack((y, y_val_aux))
                    ps = PredefinedSplit(np.array([0]*len(y)+[-1]*len(y_val)))
                    
                    HYPERPARAMS = best_model(MODEL_NAME, ps, X_train_val, y_train_val)
                    model = select_trad_model(MODEL_NAME, HYPERPARAMS)
                    self.HYPERPARAMS[MODEL_NAME] = copy.deepcopy(HYPERPARAMS)
                    model.fit(X_l_aux, y_l_aux)

                else:
                    X_l_aux = copy.deepcopy(X)
                    y_l_aux = copy.deepcopy(y)
                    X_val_aux = copy.deepcopy(X_val)
                    y_val_aux = copy.deepcopy(y_val)
                    if 'bert' in MODEL_NAME:
                        model = BERT(n_classes=self.N_CLASSES, **self.HYPERPARAMS.get('bert', {}))
                        model.fit(X_l_aux, y_l_aux)
                    elif 'setfit' in MODEL_NAME:
                        model = SETFIT(**self.HYPERPARAMS.get('setfit', {}))
                        model.fit(X_l_aux, y_l_aux, X_val_aux, y_val_aux)

                if 'random' not in MODEL_NAME:
                    pred_val = model.predict_proba(X_val_aux)
                else:
                    pred_val = np.random.rand(len(X_val), self.N_CLASSES)
            
            else: 
                X_val_aux = X_val
                y_val_aux = y_val
                pred_val = np.random.rand(len(X_val), self.N_CLASSES)

            if self.THR_CONF_CLF_opt:
                precision, recall, thresholds, weights = prec_rec_curves(
                    y_val_aux, pred_val, self.PROBS_THR, self.N_CLASSES)
                
                if self.N_CLASSES < 3:
                    precision[np.isnan(precision)] = 0
                    recall[np.isnan(recall)] = 0
                    fscore = (2 * precision * recall) / (precision + recall)
                    fscore[np.isnan(fscore)] = 0
                    idx = np.argmax(fscore)
                    self.THR_CONF_CLFS[MODEL_NAME] = thresholds[idx]
                else:
                    aux_thr = []
                    for c in range(self.N_CLASSES):
                        p_c = precision[c]; p_c[np.isnan(p_c)] = 0
                        r_c = recall[c]; r_c[np.isnan(r_c)] = 0
                        fscore = (2 * p_c * r_c) / (p_c + r_c); fscore[np.isnan(fscore)] = 0
                        idx = np.argmax(fscore)
                        aux_thr.append(thresholds[c][idx])
                    self.THR_CONF_CLFS[MODEL_NAME] = aux_thr
                
                if MODEL_NAME.startswith("regex"):
                      self.THR_CONF_CLFS[MODEL_NAME] = [0.0] * self.N_CLASSES

            self.models[MODEL_NAME] = [tokens, opt, regexes_aux, model, X_l_aux, y_l_aux]

    def predict_proba(self, X):
        all_probs = []
        for MODEL_NAME in self.MODEL_NAMES:
            base_model = MODEL_NAME if MODEL_NAME in self.models else "regex"
            if base_model not in self.models:
                return np.ones((len(X), self.N_CLASSES)) / self.N_CLASSES

            tokens, opt, regexes_aux, model, _, _ = self.models[base_model]
            
            if opt == 'manual':
                X_test_aux = self._manual_vectorize(X, tokens)
            elif 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME:
                X_test_aux = copy.deepcopy(get_matrix(tokens, X, regexes_aux, opt))
            else:
                X_test_aux = copy.deepcopy(X)
            
            probs = model.predict_proba(X_test_aux)
            all_probs.append(probs)
            
        if len(all_probs) == 1:
            return all_probs[0]
        else:
            return all_probs

    def predict(self, X):
        self.y = defaultdict(list)
        for MODEL_NAME in self.MODEL_NAMES:
            print(MODEL_NAME + '...predict (voting with confidence)')
            base_model = MODEL_NAME if MODEL_NAME in self.models else "regex"
            tokens, opt, regexes_aux, model, _, _ = self.models[base_model]
            
            if opt == 'manual':
                X_test_aux = self._manual_vectorize(X, tokens)
            elif 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME:
                X_test_aux = copy.deepcopy(get_matrix(tokens, X, regexes_aux, opt))
            else:
                X_test_aux = copy.deepcopy(X)
                
            probs = model.predict_proba(X_test_aux)
            final_predictions = list(np.argmax(probs, axis=1))
            self.y[MODEL_NAME] = final_predictions
        
        if len(self.MODEL_NAMES) == 1:
            return self.y[self.MODEL_NAMES[0]]
        return self.y