import copy
import re
import os
import sys
import pickle
import numpy as np #1.20.0 pip install numba==0.53 --user
import pandas as pd
from matplotlib import pylab
from sklearn.preprocessing import label_binarize
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
from gensim.models import TfidfModel, FastText
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from nltk.util import ngrams
import time
import platform
import random
import editdistance
from lingpy.align.multiple import mult_align
from nltk.stem import SnowballStemmer
import shutil
from scipy.stats import entropy
import ast
import math
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier as DTC
from xgboost import XGBClassifier as XGB
import itertools
from tqdm import tqdm
from itertools import combinations
import unicodedata
import ast
from os.path import dirname as up
#from limer.lime.lime_text import LimeTextExplainer as LIMER
#from lime.lime.lime_text import LimeTextExplainer as LIME
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
#import shap
from scipy.stats import kendalltau
from sklearn.linear_model import RidgeClassifier
from scipy.spatial.distance import cosine

# === IMPORTACIÓN DEEP LEARNING (OPCIONAL) ===
# Se envuelve en try-except para que el código funcione aunque no se tenga GPU/Torch instalado.
try:
    import transformers
    from transformers import get_linear_schedule_with_warmup
    from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
    from transformers import BertModel, DistilBertModel, AlbertModel    
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import SGD, Adam, lr_scheduler, AdamW
    import torch.nn.functional as F
    
    from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
    from sentence_transformers.losses import CosineSimilarityLoss
    from transformers.trainer_callback import PrinterCallback
    from datasets import Dataset
    from transformers import set_seed
except:
    pass

# === CONSTANTES GLOBALES ===
SEED = 42
F_OUTLIERS = 1.5      # Factor para detectar outliers en números (Rango Intercuartil)
F_EXT_OUTLIERS = 2.5
NGRAM_MIN = 3         # Tamaño mínimo de n-grama para considerar una "palabra" válida
WINDOW_MIN = 5        # Ventana para cálculo estadístico de números
BATCH = 64            # Tamaño del batch para Active Learning
#THR_PRB = 0.9 #learning curve
THR_CLASS = 2/3       # Umbral de pureza para asignar una regex a una clase
THR_CONF = 0.90 #0.90 regexes
THR_CONF_CLF = 0.90 #0.90 #classifiers

# === EXPRESIONES REGULARES BÁSICAS ===
# Se usan como "bloques de construcción" para las reglas minadas
pnumbers = r'\d+(?:[\.\,]\d+)?'        # Detecta números decimales o enteros
punctuation = r'[^a-zA-Z\d\s\+\-]'     # Todo lo que no sea letra, número o espacio
gap = r'(?:\w)?'                       # Gap corto (una letra opcional)
gaps =  r'(?:\w+)?'                    # Gap largo (palabra opcional)
nonalpha =  r'[^a-zA-Z\d\s]'           # No alfanuméricos
words = r'[a-zA-Z]{3,}'
whitespaces = r'[\s]*'                 # Espacios flexibles
gap_cmb = r'[\s\S]*'                   # Gap combinado (cualquier cosa)
ptimes = r'[^\S]*'
digit_mask = 'DIGIT'                   # Máscara para normalizar números
gap_mask = 'GAP'                       # Máscara para alineamiento
gap_sw = r'XYZ'                        # Separador especial para comunicación con C++

lexicon = {} # Diccionario opcional para filtrar reglas por dominio

# === HIPERPARÁMETROS DE MODELOS ===
# Repositorio central de configuraciones para todos los clasificadores
HYPERPARAMS = defaultdict(dict)
HYPERPARAMS['bert']  = {
            'scheduler_opt': True,
            'early_stopping': False,
            'validation_split': 0.0,
            'val_loss_min': None,
            'patience': None,
            'batch_size': 16,
            'epochs': 4,
            'dropout': 0.2,
            'MAX_SENT_LEN': 128, #64,
            'lr': 2e-5,
            'RUNS': 1,
            #'bert_type': 'albert'
            'bert_type': 'biobert' # Modelo pre-entrenado específico
}

HYPERPARAMS['setfit']  = {
            'batch_size': 8, 
            'num_epochs': 1, 
            'learning_rate': 2e-5
}

# --- AGREGAR ESTO ---
HYPERPARAMS['svm'] = {
    'C': 1.0,
    'kernel': 'linear',       
    'class_weight': 'balanced', # Clave para arreglar el desbalance de clases
    'probability': True,        # Necesario para Active Learning (necesitamos confianza)
    'random_state': SEED
}

HYPERPARAMS['rf'] = {
    'n_estimators': 300,      # Aumentado para estabilidad (más árboles = menos varianza)
    'class_weight': 'balanced', # Clave para el recall bajo
    'max_depth': None,        
    'min_samples_split': 5,   
    'n_jobs': -1,             
    'random_state': SEED
}
# --------------------

# Función para garantizar reproducibilidad en todos los frameworks
def seed_everything(seed=SEED):
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)   
    except:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)   
  
# Gestión de directorios de salida
def create_paths(FILENAME, root=os.getcwd()):
    if 'out' not in os.listdir( os.path.join( root ) ):
        os.mkdir( os.path.join( root, 'out' ) )
    if 'RESULTS' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTS' ) )
    # ... (Creación de subcarpetas para tablas, figuras y logs) ...
    if 'Tables' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'Tables' ) )
    if 'Figures' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'Figures' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTS') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTS', FILENAME ) )
    if 'RESULTSLC' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC' ) )
    # ... (Carpetas para Pseudo-Labeling y Active Learning) ...
    if 'PL' not in os.listdir( os.path.join( root, 'out', 'RESULTSLC') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'PL' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTSLC', 'PL') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'PL', FILENAME ) )
    if 'AL' not in os.listdir( os.path.join( root, 'out', 'RESULTSLC') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'AL' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTSLC', 'AL') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'AL', FILENAME ) )
    if 'SSLAL' not in os.listdir( os.path.join( root, 'out', 'RESULTSLC') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'SSLAL' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTSLC', 'SSLAL') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'SSLAL', FILENAME ) )
    # Copia el código C++ necesario para la ejecución
    shutil.copy( os.path.join( root, 'sw_cpp.cpp' ), os.path.join( os.getcwd(), 'sw_cpp_%s.cpp' %FILENAME ) )
    shutil.copy( os.path.join( root, 'sw_cpp_score.cpp' ), os.path.join( os.getcwd(), 'sw_cpp_score_%s.cpp' %FILENAME ) )

def remove(path, filename):
    # Función segura para borrar archivos (reintenta si está ocupado)
    #if filename in os.listdir( path ):
    while filename in os.listdir( path ):
        print(filename, 'was removed')
        os.remove( os.path.join( path, filename ) )

# Determina automáticamente dónde "cortar" el árbol de clustering (dendrograma)
# para obtener los grupos de palabras más coherentes.
def get_thr_clustering(X, Z, metric='cosine', iterations=10,seed=SEED):
    d = dendrogram(Z)
    h = [y[1] for y in d['dcoord']] # Alturas del dendrograma
    min_thr, max_thr = get_min_max_thr(Z, min(h), max(h))
    silhouettes_x = np.linspace(min_thr, max_thr, iterations)
    silhouettes_y = []
    silhouette_max = 0
    # Prueba varios cortes y se queda con el que maximiza el Silhouette Score
    for h in silhouettes_x:
        clusters_aux = fcluster(Z, t=h, criterion='distance')
        score  = silhouette_score(X, clusters_aux, metric=metric, random_state=seed)
        silhouettes_y.append( score )
        if score>silhouette_max:
            silhouette_max = score
            t = h
    return t

# Generador para dividir tokens en grupos basados en prefijos (n-grams)
def split_tokens(tokens, N=NGRAM_MIN):
    tokens = list(sorted(tokens))   
    visited = []
    for i in range(len(tokens)):
        visited_aux = []
        if i not in visited:
            visited_aux = [ tokens[i]  ]
            ngramA = tokens[i][:N]
            visited.append(i)
            for j in range(len(tokens)):
                ngramB = tokens[j][:N]
                if j not in visited:
                    # Agrupa si comparten prefijo y no son números
                    if not re.findall(r'%s' %pnumbers, tokens[j]):
                        if ngramA[0] == ngramB[0]:
                            visited_aux.append( tokens[j] )
                            visited.append(j)
        if visited_aux:
            yield visited_aux

# Filtra clústeres ruidosos o poco frecuentes
def filtering_clusters(tokens_clusters, tokens_freq):
    tokens_aux = list( split_tokens(tokens_clusters) )
    bases, filters = [], []
    for tokens in tokens_aux:
        tokens = list(sorted(tokens))
        max_ = -1
        base = ''
        # Elige la palabra más frecuente como representante del grupo
        for token in tokens:
            if tokens_freq[token]>max_:
                max_ = tokens_freq[token]
                base = token
        bases.append(base)
        filters.append(tokens)
    del tokens_aux
    gc.collect()
    return bases, filters

# Función CORE de regex: busca un patrón en un texto
def match(regex, text, pos=False):
    if not pos:
        # Devuelve los strings encontrados
        f = [ m.strip() if type(m)==str else m for m in re.findall(  r'\s%s\s' %regex,  ' '+text+' ' ) ] 
    else:
        # Devuelve los índices (posiciones) donde se encontró
        f = set() 
        for m in re.findall(  r'\s%s\s' %regex,  ' '+text+' ' ):
            if type(m)==str:
                f.add( text.index(m.strip())) 
            else:
                for elem in m:
                    f.add( text.index(elem.strip()) )             
    return f
    
# Busca regex validando rangos numéricos
# Ej: Si la regla es "IMC {NUM}" y el rango aprendido es [20-30], 
# "IMC 45" no activará la regla.
def findall(regex, pos_aux, numbers_aux, text, return_numbers=False, pnumbers=pnumbers):
    if len(numbers_aux)==0:
        return match(regex, text)
    else:
        # Crea regex temporal capturando el número
        regex_numbers = r'%s' %regex.replace(pnumbers,'('+pnumbers+')')
        find = match(regex_numbers, text)
        findings = []
        if find:
            flag = True
            for f in find:
                if type(f)==str:
                    f = [f]
                f = list(filter(None, f))
                findings.append(f)
                if flag:
                    count = 0
                    for i in range(len(f)):
                        number = float(f[i].replace(',', '.'))
                        # Valida que el número esté dentro del rango aprendido (min/max)
                        min_aux = min(numbers_aux[:,i])
                        max_aux = max(numbers_aux[:,i])
                        if number>=min_aux and number<=max_aux:
                            count+=1                        
                    if count==numbers_aux.shape[1]:
                        flag = False
                        break
            # Si todos los números coinciden con el rango esperado
            if count==numbers_aux.shape[1]:
                if return_numbers:
                    findings = np.array(findings)
                    return [findings, np.round(np.mean(numbers_aux, axis=0),0,).astype(int) ]
                else:
                    return match(regex, text)
            else:
                return []
        else:
            return []

# Genera N-gramas (secuencias de N palabras)
def n_grams(texts, N):
    tokens_aux = []
    for text in texts:
        tokens = re.split(r'\s+', text)
        for token in list(ngrams(tokens, N)):
            tokens_aux.append(" ".join(token))
    tokens_aux = np.array( sorted( list(set(tokens_aux)) ) )
    return tokens_aux

# Utilidad para guardar listas/arrays a texto plano
def save_txt(data, path, filename):
    remove( path, filename )
    with open(os.path.join(path, filename), 'w', encoding='utf-8', newline='\n') as a:
        for c in range(len(data)):
            if type(data[c])==list:
                a.write(' '.join( data[c]) )
            elif type(data[c]) in [int, float]:
                a.write( str( data[c] ) )
            else:
                a.write( data[c] )
            if c<len(data)-1:
                a.write('\n')

# Wrapper para entrenar FastText (embeddings de palabras)
def fasttext( VECTOR_SIZE, NGRAM_SIZE, min_count, sg, corpus, CORPUS_SIZE, epochs,seed=SEED  ):
    model = FastText(vector_size=VECTOR_SIZE, window=NGRAM_SIZE, min_count=min_count, sg=sg, 
                     seed=seed, workers=1, max_vocab_size=None, hashfxn=hashfxn, sorted_vocab=1)  
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus_iterable=corpus, total_examples=CORPUS_SIZE, epochs=epochs)     
    return model

# Limpieza estadística de números extraídos (reemplaza outliers por la mediana)
def replace_outliers (numbers, WINDOW_MIN=WINDOW_MIN, THR=F_OUTLIERS):
    numbers_aux = np.array( sorted(numbers, reverse=False) ).astype(float)
    median = int( np.median(numbers_aux) )
    # ... (Lógica de detección de outliers basada en Z-score local) ...
    i = 0
    while i<len(numbers_aux):
        mean = np.mean(numbers_aux[:i+WINDOW_MIN])
        std = np.std(numbers_aux[:i+WINDOW_MIN])
        z_score = (numbers_aux[i]-mean)/std
        i+=1
        if np.abs(z_score)>THR:
            break
    if i==len(numbers_aux):
        i = 0
    j = len(numbers_aux)
    while j>0:
        mean = np.mean(numbers_aux[-WINDOW_MIN+j:j])
        std = np.std(numbers_aux[-WINDOW_MIN+j:j])
        z_score = (numbers_aux[j-1]-mean)/std
        j-=1
        if np.abs(z_score)>THR:
            break
    if i!=0:
        numbers_aux[:i+1] = median
    if j==0:
        j = len(numbers_aux)      
    else:
        numbers_aux[j:] = median
    return numbers_aux

# Preprocesamiento crítico para el algoritmo Smith-Waterman (C++)
# Reemplaza caracteres especiales y números por máscaras para que el alineamiento
# se centre en la estructura de las palabras.
def sw_pre_processing(x, 
                      regexes,
                      token2pattern,
                      stopwords,
                      replace_numbers = False,
                      stop_words = False,
                      mask_numbers = True,
                      pnumbers=pnumbers, 
                      digit_mask=digit_mask,
                      nonalpha=nonalpha,
                      punctuation=punctuation,
                      whitespaces=whitespaces, gap_cmb=gap_cmb
                      ):
    
    text_aux = ' '+ x +' '
    
    if replace_numbers:
        # Lógica para normalizar números encontrados por regex existentes
        keys = sorted( list(regexes.keys()), 
                    key = lambda x: len( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x) ),
                    reverse = True )
        visited = []
        for regex in keys:          
            _, numbers_aux, _, _, _ = regexes[regex]
            f = findall(regex, [], numbers_aux, text_aux, True)
            if len(f)>0 and len(numbers_aux)>0:
                f_matches, f_mean = copy.deepcopy( f )
                for i in range(f_matches.shape[0]):
                    for j in range(f_matches.shape[1]):
                        if f_matches[i][j] not in visited:
                            text_aux = re.sub(' '+f_matches[i][j]+' ',' '+ str(f_mean[j])+' ', ' '+text_aux+' ').strip()
                            visited.append( f_matches[i][j] )

    if mask_numbers:
        text_aux = re.sub(pnumbers, digit_mask, text_aux)         
    
    # Normalización agresiva de puntuación y espacios
    text_aux = re.sub(r'(%s\s*)\1+' %nonalpha, r'\1', text_aux)
    text_aux = re.sub(r'(%s)\s*' %nonalpha, r'(?:\\\1\\s*)+ ', text_aux)
    text_aux = re.sub(r'(\(\?\:\\%s\\s\*\))\+' %punctuation, r'%s' %punctuation.replace('\\', '\\\\'), text_aux)  
    text_aux = re.sub(r'(%s\s*)\1+' %re.escape(punctuation), r'\1', text_aux)  
    text_aux = re.sub(r'(%s)' %re.escape(punctuation), r'(?:\1\\s*)*', text_aux)  
    
    if mask_numbers:
        text_aux = re.sub(digit_mask, pnumbers.replace('\\', '\\\\'), text_aux) 
        
    text_aux = text_aux.strip()
    corpus_aux = text_aux.split(' ')
    for t in range(len(corpus_aux)):
        if stop_words:
            if corpus_aux[t] in stopwords:
                corpus_aux[t] = r'(?:%s)?' %corpus_aux[t]
        if corpus_aux[t] in token2pattern:
            corpus_aux[t] = token2pattern[corpus_aux[t]]     
    return ' '.join( corpus_aux )

# Elimina secuencias que son subsecuencias de otras
def reduce_sequences(sequences, gap=r' '):
    #no comb yet: gap_cmb=gap_cmb
    sequences = [re.split(r'%s' %gap, seq) for seq in sequences]
    #print(sequences)
    sequences = sorted(sequences, key=lambda x:len(x), reverse=True)
    descartar = []
    filtrados = []
    for seqA in sequences:
        for seqB in sequences:
            if gap.join(seqA) != gap.join(seqB) and len(set(seqB).difference(set(seqA)))==0:
                descartar.append(gap.join(seqB))
        if gap.join(seqA) not in descartar:
            filtrados.append(gap.join(seqA))
    return filtrados



# === ASIGNACIÓN DE CLASES A REGEX ===
# Evalúa cada regex candidata para decidir a qué clase pertenece y con qué confianza.
def get_classes_regexes(regexes, y, tokens2pos, gap_cmb=gap_cmb, whitespaces=whitespaces, THR_CLASS=THR_CLASS):
    keys = sorted( list(regexes.keys()), 
                    key = lambda x: len( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x) ),
                    reverse = False
     )
    regex2class = {}
    regexes_aux = {}
    
    for indexA in range(len(keys)):
        # Obtiene metadatos de la regex
        posA, numbersA, pattern2token, pattern2tokens, model = regexes[keys[indexA]]
        
        pos = tokens2pos[keys[indexA]] # índices de documentos que hacen match
        pos = np.array(pos)

        labels_texts = y[pos]
        
        # Evitar índices no válidos
        try:
            posA_int = np.array(posA, dtype=int)
            labels_training = y[posA_int]
        except Exception as e:
            print(f"[WARN] Índices inválidos detectados en get_classes_regexes() ({e}). Se omiten estos casos.")
            continue

        # Calcula la clase mayoritaria (Voto Mayoritario)
        label_aux, f_aux = Counter(labels_texts).most_common()[0]
        
        # Si la pureza supera el umbral (THR_CLASS), se acepta la regla
        if f_aux/len(labels_texts) >THR_CLASS: #>= THR_CLASS:
            # Calcula precisión/confianza
            ypred =  np.ones(len(labels_training), dtype=int)
            ytrue = np.where(labels_training==label_aux,1,0)
            
            conf = precision_score(ytrue, ypred)
            
            regexes_aux[keys[indexA]] = [posA, numbersA, pattern2token, pattern2tokens, model]
            regex2class[keys[indexA]] = [label_aux, conf]

    return regexes_aux, regex2class


# Filtra las regex finales basándose en:
# 1. Palabras clave (Lexicon)
# 2. Umbral de confianza calculado anteriormente
def get_filtered_regexes(regexes, y, kw, pattern2token, regex2class, THR_CONF=THR_CONF, whitespaces=whitespaces, gap_cmb=gap_cmb): 
    keys_regexes = list( regexes.keys() )
    labeled_regexes = {}
    labeled_regexes_filtered = {}
    labeled_regexes_all = {}
    i = 0
    while i<len(keys_regexes):
        label = -1
        conf = -1
        key_i = keys_regexes[i]
        label, conf = regex2class[key_i]
        flag = False
        
        # Verifica si alguna palabra de la regex está en el lexicon (kw)
        key_i_aux = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), key_i)
        for token in key_i_aux:
            if token in pattern2token:
                tokenA = pattern2token[token]
            else:
                tokenA = copy.deepcopy(token)
            for tokenB in kw:
                if tokenB in tokenA:
                    flag = True
                    break
            if flag:
                break
        
        # Lógica de filtrado
        if flag: # Contiene keyword
            if label != -1 and conf>THR_CONF: 
                labeled_regexes[key_i] = [label, conf]
                labeled_regexes_filtered[key_i] = [label, conf]
                labeled_regexes_all[key_i] = [label, conf]
            else:
                labeled_regexes_filtered[key_i] = [label, conf]
                labeled_regexes_all[key_i] = [label, conf]
        else: # No contiene keyword
            labeled_regexes_all[key_i] = [label, conf]
        i+=1

    return labeled_regexes, labeled_regexes_filtered, labeled_regexes_all

# Construye la matriz de características (Bag-of-Regexes)
# Filas = Documentos, Columnas = Regex
# Valores = Conteos o TF-IDF de ocurrencias de la regex en el documento
def get_matrix(tokens, X, regexes, opt=False, idf=True, return_idf=False):
    n_x, n_t = len(X), len(tokens)
    matrix = np.zeros((n_x,n_t))
    idf_vector = np.zeros(n_t)
    for t in range(n_t):
        d = 0
        for x in range(n_x):
            if opt:
                pos_aux, numbers_aux, _, __, ___ = regexes[tokens[t]]
                f = len( findall(tokens[t], pos_aux, numbers_aux, X[x]) )
            else:
                f = len( match( re.escape(tokens[t]), X[x]) )
            matrix[x,t] = f
            if f>0:
                d += 1
        if d==0:
            idf_vector[t] = 0
        else:
            idf_vector[t] = np.log10(n_x/d) # Cálculo IDF
    if idf:
        if return_idf:
            return matrix*idf_vector, idf_vector
        else:
            return matrix*idf_vector
    else:
        return matrix

# Wrapper para búsqueda de hiperparámetros (GridSearch)
def best_model(MODEL, ps, X_train_val, y_train_val, scoring='accuracy', SEED=SEED):
    seed_everything()

    # --- NUEVO: caso regex ---
    if 'regex' in MODEL:
        print("[INFO] Modelo 'regex' detectado: no se aplica GridSearchCV.")
        # Devolvemos parámetros vacíos para mantener compatibilidad
        return {"model": "regex", "params": {}, "score": None}

    # --- Modelos tradicionales ---
    # Define la grilla de búsqueda según el modelo elegido
    if 'svm' in MODEL:
        best_params = {'random_state': SEED, 'probability': True}
        param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
        model = SVC(random_state=SEED)
    elif 'nb' in MODEL:
        best_params = {}
        param_grid = {'alpha': [0, 0.25, 0.75, 1]}
        model = MNB()
    elif 'rf' in MODEL:
        best_params = {'random_state': SEED}
        param_grid = {'criterion': ('entropy', 'gini'), 'n_estimators': [10, 100, 500, 1000]}
        model = RFC(random_state=SEED)
    elif 'gbc' in MODEL:
        best_params = {'random_state': SEED}
        param_grid = {'n_estimators': [5, 50, 250, 500], 'max_depth': [1, 3, 5, 7, 9], 'learning_rate': [0.01, 0.1, 1, 10, 100]}
        model = GBC(random_state=SEED)
    elif 'xgb' in MODEL:
        best_params = {'random_state': SEED}
        param_grid = {'gamma': [0, 0.5, 1, 10], 'learning_rate': [0.1, 0.3, 0.8, 1.0], 'n_estimators': [10, 20, 50, 200, 400]}
        model = XGB()
    else:
        print(f"[WARN] Modelo no reconocido: {MODEL}. Se omite búsqueda de hiperparámetros.")
        return {"model": MODEL, "params": {}, "score": None}

    # --- Búsqueda de hiperparámetros ---
    clf = GridSearchCV(model, param_grid=param_grid, cv=ps, scoring=scoring)
    clf.fit(X_train_val, y_train_val)
    best_params.update(clf.best_params_)

    del clf
    del model
    gc.collect()

    return best_params

# Factory Pattern para instanciar modelos de scikit-learn con params ya optimizados
def select_trad_model(MODEL, HYPERPARAMS):
    seed_everything()
    
    # Seleccionamos solo los parámetros específicos del modelo actual
    params = HYPERPARAMS.get(MODEL, {}) if isinstance(HYPERPARAMS, dict) and MODEL in HYPERPARAMS else HYPERPARAMS

    if 'svm' in MODEL:
        if 'probability' not in params: params['probability'] = True
        model = SVC(**params)            
    elif 'nb' in MODEL:
        model = MNB(**params)
    elif 'rf' in MODEL:
        model = RFC(**params)
    elif 'gbc' in MODEL:
        model = GBC(**params)
    elif 'xgb' in MODEL:
        model = XGB(**params)
    else:
        print(f"[WARN] Modelo no reconocido en select_trad_model(): {MODEL}. Retornando None.")
        model = None
    return model

# Calcula curvas P-R para encontrar el umbral óptimo de clasificación
def prec_rec_curves(y_test, y_pred, probs, N_CLASSES):
    y_pred = copy.deepcopy(y_pred)
    precision = []
    recall = []
    thresholds = []
    weights = []
    # Lógica binaria vs multiclase
    if N_CLASSES<3: #binary
        y_pred = y_pred[:,1] 
        for p in probs:
            preds = np.where( y_pred>=p, 1, 0 )
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            precision.append(prec)
            recall.append(rec)
            thresholds.append(p)
        precision = np.array(precision)
        recall = np.array(recall)
        thresholds = np.array(thresholds)
    else: #multiclass
        precision = []
        recall = []
        thresholds = []
        for c in range(N_CLASSES):
            precision_aux = []
            recall_aux = []
            thresholds_aux = []
            y_pred_multi = y_pred[:,c]
            y_test_multi = np.where(y_test==c, 1, 0)
            for p in probs:
                preds = np.where( y_pred_multi>=p, 1, 0 )
                tn, fp, fn, tp = confusion_matrix(y_test_multi, preds).ravel()
                prec = tp/(tp+fp)
                rec = tp/(tp+fn)
                precision_aux.append(prec)
                recall_aux.append(rec)
                thresholds_aux.append(p)
            w = list(y_test).count(c)/len(y_test)
            precision.append( np.array(precision_aux) )
            recall.append( np.array(recall_aux) )
            thresholds.append( np.array(thresholds_aux) )
            weights.append( np.array(w) )
        precision = np.array(precision)
        recall = np.array(recall)
        thresholds = np.array(thresholds)

    return precision, recall, thresholds, weights

# Calcula el Area Under Learning Curve (AULC) para evaluar Active Learning
def AULC(x,y):
    suma = 0
    for i in range(1, len(x)):
        suma += (y[i] + y[i-1])
    suma = (1/2)*suma
    return suma/(len(x)-1)

# Eficiencia de datos (Data Efficiency)
def deff(PL, AL, n):
   return  ( (PL[n]-AL[:n]).sum() )/ ( (PL[n]-PL[:n]).sum() )

# Criterio de parada (Stopping Criterion)
def SC(v):
    max_ = len(v)-1
    for i in range(1, len(v)-2):
        # Detecta pico local en performance para detener el entrenamiento
        if v[i]>v[i-1] and v[i]>v[i+1]:
            max_ = i+1
            break
    return max_