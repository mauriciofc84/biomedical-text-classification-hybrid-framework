import numpy as np
import gc
import copy
from sklearn.model_selection import PredefinedSplit
from sklearn.utils import shuffle

# Importa tus módulos
from fregex import FREGEX
from utils import *
from bert import BERT
from cregex import CREGEX
from mysetfit import SETFIT


def get_tokens(X, y, N, FILENAME):
    regexes = {}
    opt = False
    tokens = []
    if type(N) == int:
        tokens = n_grams(X, N)
    elif 'regex' in str(N): 
        mode = N.split('-')[0]
        fregex = FREGEX(X, y, FILENAME) 
        fregex.fit()
        regexes = copy.deepcopy(fregex.transform())
        opt = True
        tokens = list(regexes.keys())
    
    # === CORRECCIÓN AQUÍ ===
    # Usamos len() > 0 para que funcione tanto con Listas como con Numpy Arrays
    if len(tokens) > 0:
        tokens = sorted(tokens)
    else:
        tokens = ["dummy_token"]
        
    return regexes, opt, tokens


class Curves(object):
    def __init__(self,
                 X_train, y_train,
                 X_val, y_val,
                 X_test,
                 N_CLASSES, CURVE, MODEL,
                 BATCH, FILENAME,
                 HYPERPARAMS={}, SEED=42):
        
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)
        self.X_val = copy.deepcopy(X_val)
        self.y_val = copy.deepcopy(y_val)
        self.X_test = copy.deepcopy(X_test)
        self.N_CLASSES = N_CLASSES
        self.CURVE = CURVE
        self.MODEL = MODEL.split('-')[0]  # clf
        self.BATCH = BATCH  # lc
        self.FILENAME = FILENAME
        self.SEED = SEED
        self.results = {}
        self.dst_cregex = []
        self.N_FEATURES = []
        self.X_u = []
        
        # Copia de hiperparámetros por seguridad
        self.HYPERPARAMS_BASE = copy.deepcopy(HYPERPARAMS)

        # Configuración de NGRAM_SIZE
        if 'cregex' in MODEL:
            split_aux = MODEL.split('-')
            if len(split_aux) > 2:  # clf-nx-cregex
                self.NGRAM_SIZE = '-' + MODEL.split('-')[1] + '*cregex'
            else:  # clf-cregex
                self.NGRAM_SIZE = '*cregex'
        elif 'fregex' in MODEL:
            self.NGRAM_SIZE = 'fregex'
        elif 'bert' not in MODEL and 'setfit' not in MODEL:
            try:
                self.NGRAM_SIZE = int(MODEL[-1])
            except:
                self.NGRAM_SIZE = 1 # Default
        elif 'bert' in MODEL:
            self.NGRAM_SIZE = 'bert'
        elif 'setfit' in MODEL:
            self.NGRAM_SIZE = 'setfit'

        # Selección de hiperparámetros
        if 'bert' not in self.MODEL and 'setfit' not in self.MODEL:
            # Búsqueda inicial
            self.HYPERPARAMS = self.search_hyperparams(self.X_train, self.y_train, self.X_val, self.y_val)
        else:
            self.HYPERPARAMS = copy.deepcopy(self.HYPERPARAMS_BASE.get(self.MODEL, {}))

        print('[INFO] Curves Init - NGRAM_SIZE:', self.NGRAM_SIZE)

    def search_hyperparams(self, X_l, y_l, X_val, y_val):
        ps = PredefinedSplit(np.array([0] * len(y_l) + [-1] * len(y_val)))
        y_l_val = copy.deepcopy(np.hstack((y_l, y_val)))
        
        regexes, opt, tokens = get_tokens(X_l, y_l, self.NGRAM_SIZE, self.FILENAME)
        
        X_l_aux = copy.deepcopy(get_matrix(tokens, X_l, regexes, opt))
        X_val_aux = copy.deepcopy(get_matrix(tokens, X_val, regexes, opt))
        X_l_val = copy.deepcopy(np.vstack((X_l_aux, X_val_aux)))
        
        return best_model(self.MODEL, ps, X_l_val, y_l_val)

    def start(self):
        # Inicialización de Learning Curve (Split inicial Labeled/Unlabeled)
        X_l = np.array([])
        y_l = np.array([])
        
        # Aseguramos tener al menos un ejemplo de cada clase en el set inicial
        classes_ = copy.deepcopy(self.y_train[:self.BATCH])
        attempts = 0
        while len(set(classes_)) != self.N_CLASSES and attempts < 100:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.SEED + attempts)
            classes_ = copy.deepcopy(self.y_train[:self.BATCH])
            attempts += 1
            
        del classes_
        gc.collect()
        
        X_l = self.X_train[:self.BATCH]
        y_l = self.y_train[:self.BATCH]
        X_u = self.X_train[self.BATCH:]
        y_u = self.y_train[self.BATCH:]
        
        return X_l, y_l, X_u, y_u, [], []

    def model_selection(self, X_train, y_train, X_test, X_u=[], results=False, return_model=False):
        model = None
        pred = None
        pred_u = None
        scores_u = None

        # === OPCIÓN 1: CREGEX (Ahora con Active Learning Real) ===
        if "regex" in self.MODEL.lower():
            # print("[INFO] Entrenando ciclo CREGEX...")
            model = CREGEX(
                self.FILENAME,
                self.MODEL,
                self.N_CLASSES,
                HYPERPARAMS=self.HYPERPARAMS,
                SEED=self.SEED
            )

            # 1. Entrenar
            model.fit(X_train, y_train, self.X_val, self.y_val)
            
            # 2. Predecir en Test (Evaluación final, usa 'predict' con umbrales)
            pred = model.predict(X_test)

            # 3. Predecir en Unlabeled (Para selección inteligente)
            if len(X_u) > 0:
                # Usamos predict_proba para obtener confianza
                probs_u = model.predict_proba(X_u)
                
                # --- LÓGICA DE ACTIVE LEARNING ---
                # scores_u: Confianza del modelo.
                # argsort ordenará de menor a mayor. Queremos etiquetar lo que tiene MENOR confianza.
                scores_u = np.max(probs_u, axis=1) 
                
                # pred_u: La clase que el modelo cree que es (aunque no esté seguro)
                pred_u = np.argmax(probs_u, axis=1)
            else:
                pred_u, scores_u = None, None

            if return_model:
                return pred, pred_u, scores_u, model
            return pred, pred_u, scores_u

        # === OPCIÓN 2: Modelos Tradicionales (SVM, RF, etc.) ===
        elif any(m in self.MODEL.lower() for m in ["svm", "rf", "lr", "knn", "nb"]):
            # print("[INFO] Entrenando ML tradicional:", self.MODEL)
            model = select_trad_model(self.MODEL, self.HYPERPARAMS)
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)
            
            if return_model:
                # Nota: Para modelos trad, el cálculo de X_u se hace usualmente fuera 
                # o el objeto model se usa después. Aquí seguimos tu patrón original.
                return pred, None, None, model
            return pred, None, None

        # === OPCIÓN 3: Deep Learning (BERT / SetFit) ===
        elif "bert" in self.MODEL.lower():
            model = BERT(**self.HYPERPARAMS.get("bert", {}))
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)
            if return_model:
                return pred, None, None, model
            return pred, None, None

        elif "setfit" in self.MODEL.lower():
            model = SETFIT(**self.HYPERPARAMS.get("setfit", {}))
            model.fit(X_train, y_train, self.X_val, self.y_val)
            pred = model.predict_proba(X_test)
            if return_model:
                return pred, None, None, model
            return pred, None, None

        else:
            print(f"[WARN] Modelo no reconocido: {self.MODEL}")
            return None, None, None

    def learningCurve(self):
        MIN_X = 1
        scores = []
        y_clf = []
        y_u_dst = []
        
        # Iniciar splits
        X_l, y_l, X_u, y_u, x, y = self.start()
        
        matrix_U = []
        iteration = 0

        # === BUCLE DE APRENDIZAJE ACTIVO ===
        while len(X_u) > 0:
            print(f"Iteración LC: Labeled={len(X_l)}, Unlabeled={len(X_u)}")
            
            # Entrenar y obtener predicciones (incluyendo scores para X_u en caso de CREGEX)
            pred, pred_u, scores_u, clf = self.model_selection(X_l, y_l, self.X_test, X_u, False, True)
            
            x.append(len(y_l))
            y.append(pred)

            if len(X_u) == 0:
                break

            # === ESTRATEGIA DE SELECCIÓN (SAMPLING) ===
            indexes = np.array([], dtype=int)
            
            if self.CURVE == 'PL': # Pseudo-Labeling (No implementado full aquí, asumo random/secuencial)
                indexes = np.arange(len(X_u))
            else:
                # Active Learning: Ordenar por Score (de menor a mayor confianza)
                if scores_u is not None:
                    indexes = np.argsort(scores_u)
                else:
                    # Fallback para modelos que devuelven el objeto clf sin scores directos (ej. SVM arriba)
                    # Si clf existe, predecimos aquí:
                    if hasattr(clf, 'predict_proba'):
                        # Lógica para modelos tradicionales que no sea CREGEX
                        # (Dependiendo de si tu utils maneja raw text o matriz, esto puede variar. 
                        #  Asumimos que CREGEX ya lo manejó internamente en model_selection)
                        
                        # Si es ML Tradicional, necesitamos features. 
                        # Simplificación: Si scores_u es None y no es CREGEX, usamos random por seguridad
                        indexes = np.arange(len(X_u))
                    else:
                        indexes = np.arange(len(X_u))

            if len(indexes) == 0:
                print("[WARN] No hay índices válidos.")
                break

            # Convertir a numpy arrays si no lo son
            if not isinstance(self.X_u, np.ndarray):
                self.X_u = np.array(self.X_u, dtype=object)
            if not isinstance(y_u, np.ndarray):
                y_u = np.array(y_u)

            # Guardar estado de la iteración
            try:
                # Guardamos qué muestras se seleccionaron
                matrix_U.append([self.X_u[indexes], y_u[indexes]])
            except:
                pass

            # Guardar predicciones sobre lo no etiquetado (para análisis posterior)
            if pred_u is None:
                y_clf.append(np.zeros(len(indexes)))
            else:
                y_clf.append(pred_u[indexes])

            if scores_u is None:
                scores.append(np.zeros(len(indexes)))
            else:
                scores.append(scores_u[indexes])

            # === ACTUALIZACIÓN DE DATASETS ===
            # Mover las mejores (o peores) muestras de U a L
            # Tomamos los primeros BATCH índices (los de menor confianza si usamos argsort)
            
            idx_to_move = indexes[:self.BATCH]
            
            X_new = np.array(X_u)[idx_to_move]
            y_new = np.array(y_u)[idx_to_move]
            
            X_l = np.concatenate((X_l, X_new))
            y_l = np.concatenate((y_l, y_new))
            
            # Eliminar de Unlabeled
            X_u = np.delete(np.array(X_u), idx_to_move, axis=0)
            y_u = np.delete(np.array(y_u), idx_to_move, axis=0)

            # === CAMBIO CRÍTICO: SE ELIMINÓ EL BREAK ===
            # Antes había un 'if regex break' aquí. Se eliminó para permitir el ciclo.
            
            del clf
            gc.collect()

        # === Guardar resultados ===
        self.results['x'] = np.array(x, dtype=object)
        self.results['y'] = np.array(y, dtype=object)
        self.results['scores'] = np.array(scores, dtype=object)
        self.results['y_u_dst'] = np.array(y_u_dst, dtype=object)
        self.results['y_clf'] = np.array(y_clf, dtype=object)
        self.results['dst_cregex'] = np.array(self.dst_cregex, dtype=object)
        self.results['X_u-y_u'] = np.array(matrix_U, dtype=object)

        # Cleanup
        del X_l, y_l, X_u, y_u
        gc.collect()