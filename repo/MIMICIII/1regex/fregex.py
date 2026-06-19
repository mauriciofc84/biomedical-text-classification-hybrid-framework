from utils import *
import unicodedata # Aseguramos que esté disponible explícitamente

class FREGEX(object):
    def __init__(self, X, y, filename, verbs_opt=True, mode='automatic', 
                 pnumbers=pnumbers, punctuation=punctuation, gap=gap, gaps=gaps, nonalpha=nonalpha, 
                 ptimes=ptimes, whitespaces=whitespaces,
                 digit_mask=digit_mask, gap_mask=gap_mask, gap_sw = gap_sw,
                 stem=False, lev=True, NGRAM_SIZE=NGRAM_MIN, VECTOR_SIZE=100, seed=SEED):
        
        # === 1. CONFIGURACIÓN E INICIALIZACIÓN ===
        self.__metaclass__ = 'FREGEX'
        self.X = copy.deepcopy(X)
        
        # Tokenización: Rompe el texto en palabras usando espacios como separador
        self.corpus = [ re.split(r'\s+', str(text)) for text in self.X ]
        self.all_tokens = sum(self.corpus, [])
        
        # Construcción del Vocabulario Inicial (Gensim Dictionary)
        self.dct = Dictionary(self.corpus) 
        self.tokens = np.array( list(sorted([key for key in self.dct.token2id])) ) 
        
        # Carga de Verbos (para ignorarlos si verbs_opt=True)
        if verbs_opt:
            self.verbs = set(pd.read_csv('verbos-ingles-conjugaciones.txt').to_numpy().reshape(-1,))
        else:
            self.verbs = []

        # Carga de Stopwords (Palabras vacías: el, la, paciente, aps, etc.)
        # (Se mantienen cargadas por si se usan en otros métodos, pero no se filtrarán en el vocabulario)
        with open(os.path.join(os.getcwd(), 'stop.txt'), 'r', encoding='utf-8', newline='\n') as a:
            self.stopwords = a.read().split('\n')[:-1]
            self.stopwords = [unicodedata.normalize('NFD', w.lower()).encode('ascii', 'ignore').decode('utf-8').strip() for w in self.stopwords]
            self.stopwords = sorted(self.stopwords)

        self.NGRAM_SIZE = NGRAM_SIZE
        stemmer = SnowballStemmer('english') 
        
        # ==============================================================================
        # [MODIFICACIÓN 1] FILTRADO ROBUSTO DE TOKENS
        # ==============================================================================
        self.tokens_filtered = []
        
        # a) Preparamos un SET de stopwords normalizadas para búsqueda rápida (O(1))
        stopwords_set = set()
        for w in self.stopwords:
            # Normalización: minúsculas + quitar tildes (NFD)
            w_clean = unicodedata.normalize('NFD', w.lower()).encode('ascii', 'ignore').decode('utf-8')
            stopwords_set.add(w_clean)
            
        print(f"[DEBUG] FREGEX: Stopwords activas y normalizadas: {len(stopwords_set)}")

        # b) Filtramos el vocabulario
        for token in self.tokens:
            # Normalizamos el token actual igual que las stopwords para comparar
            # token_clean = unicodedata.normalize('NFD', token.lower()).encode('ascii', 'ignore').decode('utf-8')
            
            # Condiciones de aceptación:
            # 1. Longitud mínima
            # 2. No es un verbo conjugado
            # 3. No es un número puro
            # 4. NO está en la lista negra (Stopwords) - [MODIFICADO: DESACTIVADO]
            if (len(token) >= self.NGRAM_SIZE 
                and token not in self.verbs 
                and not re.findall(r'%s' % pnumbers, token)):
                # and token_clean not in stopwords_set): # <--- FILTRO DE STOPWORDS DESACTIVADO
                
                self.tokens_filtered.append(token)
        
        # c) SOBRESCRITURA DEL VOCABULARIO
        # ¡CRÍTICO! Reemplazamos self.tokens con la lista limpia.
        # Ahora 'get_clusters' solo verá palabras útiles.
        self.tokens = np.array(self.tokens_filtered)
        self.TOKENS_SIZE = len(self.tokens) # Actualizamos el tamaño
        # ==============================================================================
        
        # Configuración de métricas
        self.stem = stem
        self.lev = lev
        # Mapeo de stemming si está activo
        if self.stem:
            self.token2stem = dict([(token, stemmer.stem(token)) if token in self.tokens_filtered else (token, token) for token in self.tokens])        
        
        # Inicialización de variables para guardar reglas
        self.keywords = []
        self.clusters = defaultdict(list)
        self.token2pattern = {}
        self.pattern2token = {}
        self.pattern2tokens = defaultdict(list)
        self.y = copy.deepcopy(y)
        
        # Configuración de Regex y Máscaras
        self.pnumbers = pnumbers
        self.ptimes = ptimes
        self.digit_mask = digit_mask
        self.gap_mask = gap_mask
        self.punctuation = punctuation
        self.nonalpha = nonalpha
        self.gap = gap
        self.gaps = gaps
        self.whitespaces = whitespaces
        self.regexes = {}
        self.FILENAME = filename
        self.CLASSES_SIZE = len(set(self.y))
        self.VECTOR_SIZE = VECTOR_SIZE
        self.CORPUS_SIZE = len(self.corpus)
        self.SEED = seed
        self.mode = mode
        self.tokens2pos = defaultdict(list)
        self.gap_sw = gap_sw
        
    def fit(self):
        # Orquestador del proceso de minería
        self.get_clusters()           # 1. Agrupar palabras
        self.get_global_alignments()  # 2. Generar regex visual
        self.get_local_alignments()   # 3. Buscar ocurrencias en el texto
        
    def transform(self):
        return self.regexes

    def lev_metric(self, x, y):
        i, j = int(x[0]), int(y[0])  
        if self.stem:
            return editdistance.eval(self.token2stem[self.tokens[i]], self.token2stem[self.tokens[j]])
        else:
            return editdistance.eval(self.tokens[i], self.tokens[j])
    
    def get_clusters(self, min_count=1, epochs=100, sg=1):        
        # === FASE 1: CLUSTERING ===
        if self.lev:
            X_aux = np.arange(len(self.tokens)).reshape(-1, 1)
            Z = linkage(X_aux, method='average', metric=self.lev_metric, optimal_ordering=True)
            th = self.NGRAM_SIZE
            self.model = None
        else:
            X_aux = np.zeros((self.TOKENS_SIZE, self.VECTOR_SIZE))
            if self.stem:
                corpus_stem = []
                for c in self.corpus:
                    corpus_aux = []
                    for t in c:
                        corpus_aux.append(self.token2stem[t])
                    corpus_stem.append(corpus_aux)
                self.model = fasttext(self.VECTOR_SIZE, self.NGRAM_SIZE, min_count, sg, corpus_stem, self.CORPUS_SIZE, epochs)
                for t in range(self.TOKENS_SIZE):
                    X_aux[t] = self.model.wv.get_vector(self.token2stem[self.tokens[t]])
            else:
                self.model = fasttext(self.VECTOR_SIZE, self.NGRAM_SIZE, min_count, sg, self.corpus, self.CORPUS_SIZE, epochs)
                for t in range(self.TOKENS_SIZE):
                    X_aux[t] = self.model.wv.get_vector(self.tokens[t])
            
            Z = linkage(X_aux, method='average', metric='cosine', optimal_ordering=True)
            th = get_thr_clustering(X_aux, Z)  
        
        tokens_freq = {token: self.all_tokens.count(token) for token in self.tokens}
        all_clusters = fcluster(Z, t=th, criterion='distance')        
        
        for c in np.unique(all_clusters):
            tokens_aux = self.tokens[np.where(all_clusters == c)[0]]
            tokens_aux = set(tokens_aux)
            elements = tokens_aux.intersection(self.verbs)
            tokens_aux.difference_update(elements)
            tokens_aux = [token for token in tokens_aux if len(token) >= self.NGRAM_SIZE]
            
            if tokens_aux:
                k_values, v_values = filtering_clusters(tokens_aux, tokens_freq)
                for k, v in zip(k_values, v_values):
                    if len(k) >= self.NGRAM_SIZE and len(v) > 1:
                        v.remove(k)
                        self.clusters[k] = v

    def get_global_alignments(self):
        # === FASE 2: ALINEAMIENTO GLOBAL ===
        for A in self.clusters:
            aux_alignments = []
            clean = lambda s: re.sub(r'[^A-Za-z ]', '', s)
            base = clean(A)
            cluster_clean = [clean(B) for B in self.clusters[A] if B.strip() != ""]
            if not base or not cluster_clean:
                continue
            try:
                aux_alignments = mult_align([' '.join(list(base))] + [' '.join(list(B)) for B in cluster_clean])
            except Exception as e:
                # print(f"[WARN] Skipping cluster '{A}' due to bad alignment: {e}")
                continue

            aux_alignments = np.array(aux_alignments)
            MAX_SIZE = aux_alignments.shape[1]
            pattern = self.gap
            count = 0
            
            for j in range(MAX_SIZE):
                chrs = np.unique(aux_alignments[:, j])
                if len(chrs) == 1 and chrs[0] != '-':
                    pattern += chrs[0]
                    count += 1
                else:
                    pattern += self.gap_mask
            
            if count >= self.NGRAM_SIZE:
                pattern = re.sub(r'(?:%s){2,}' % self.gap_mask, self.gaps.replace('\\', '\\\\'), pattern)
                pattern = re.sub(r'(?:%s){1}' % self.gap_mask, self.gap.replace('\\', '\\\\'), pattern)
                
                self.token2pattern[A] = pattern
                self.pattern2token[pattern] = A
                self.pattern2tokens[pattern].append(A)
                for B in self.clusters[A]:
                    self.token2pattern[B] = pattern
                    self.pattern2tokens[pattern].append(B)

    def get_local_alignments(self, expand_numbers=False, min_count=1, epochs=100, sg=0):
        # === FASE 3: BÚSQUEDA LOCAL ===
        corpus = []
        for c in range(len(self.corpus)):
            text_aux = ' ' + ' '.join(self.corpus[c]) + ' '
            text_aux = sw_pre_processing(text_aux, self.regexes, self.token2pattern, self.stopwords)
            corpus_aux = text_aux.split(' ')
            corpus.append(corpus_aux)
            
        if self.mode == 'automatic':
            print('PATH-fregex...', os.getcwd())
            remove(os.path.join(os.getcwd(), 'out'), 'TOKENS_' + self.FILENAME + '.txt')
            remove(os.getcwd(), 'sw_cpp_%s' % self.FILENAME)
            remove(os.getcwd(), 'sw_cpp_%s.exe' % self.FILENAME)
            
            # ==============================================================================
            # [MODIFICACIÓN 2] LIMPIEZA DE FRASES BASURA ANTES DE ENVIAR A C++
            # ==============================================================================
            # Evita que el alineador encuentre frases comunes que no queremos
            corpus_limpio = []
            frases_basura = [
#                "Fundamento Clinico APS", 
#                "Fundamento Clinico",
#                "Fundamento", # <--- Matar la palabra raíz
#                "Clinico APS",
#                "APS Paciente", 
#                "Paciente de",
#                "Paciente con",
#                "derivado para",
#                "solicito evaluacion",
#                "Trastorno de la refraccion no especificado",
#                "OTRAS",
#                "OTROS"
            ]
            
            for doc in corpus:
                texto_doc = " ".join(doc)
                for basura in frases_basura:
                    # Borrado insensible a mayúsculas/minúsculas
                    texto_doc = re.sub(r'(?i)' + re.escape(basura), " ", texto_doc)
                corpus_limpio.append(texto_doc.split())

            # Guardamos la versión LIMPIA
            save_txt(corpus_limpio, os.path.join(os.getcwd(), 'out'), 'DATOSX_' + self.FILENAME + '.txt')
            # ==============================================================================
            
            save_txt(self.corpus, os.path.join(os.getcwd(), 'out'), 'CORPUSX_' + self.FILENAME + '.txt')
            save_txt(self.y.astype(str), os.path.join(os.getcwd(), 'out'), 'CLASESX_' + self.FILENAME + '.txt')
            
            tokens_aux = []
            exclude_tokens = set()
            
            while len(tokens_aux) == 0:
                if platform.system() == 'Windows':
                    os.system("g++ sw_cpp_%s.cpp -o sw_cpp_%s.exe" % (self.FILENAME, self.FILENAME))
                    os.system("sw_cpp_%s.exe %s" % (self.FILENAME, self.FILENAME))
                    remove(os.getcwd(), 'sw_cpp_%s.exe' % self.FILENAME)
                elif platform.system() == 'Linux':
                    os.system("g++ sw_cpp_%s.cpp -o sw_cpp_%s" % (self.FILENAME, self.FILENAME))
                    os.system("./sw_cpp_%s %s" % (self.FILENAME, self.FILENAME))
                    remove(os.getcwd(), 'sw_cpp_%s' % self.FILENAME)
                
                with open(os.path.join(os.getcwd(), 'out', 'TOKENS_' + self.FILENAME + '.txt'), 'r', encoding='utf-8', newline='\n') as a:
                    tokens_aux = []
                    read_aux = a.read().split("\n")[:-1]
                    read_aux = list(set(read_aux))
                    tokens2pos = defaultdict(list)
                    
                    for i in range(len(self.y)):
                        tokens_r = set()
                        for aux in read_aux:
                            pos, token_aux = re.split(self.gap_sw, aux)
                            posA, posB = pos.split('-')
                            posA = int(posA); posB = int(posB)
                            token_aux = token_aux.strip()
                            if i == posA or i == posB:
                                if re.findall(re.escape(self.pnumbers), token_aux):
                                    token_aux = re.sub(r'%s' % re.escape(self.pnumbers),
                                                       r'%s%s' % (self.pnumbers.replace('\\', '\\\\'),
                                                                  self.ptimes.replace('\\', '\\\\') * (self.y[i])),
                                                       token_aux)
                                tokens_r = tokens_r.union([token_aux])
                        
                        tokens_p = np.array(list(tokens_r), dtype=object)
                        tokens_r = reduce_sequences(tokens_p)
                        tokens_aux.extend(tokens_r)

                        for seq in tokens_r:
                            tokens2pos[seq].append(i)

                        exclude_tokens = exclude_tokens.union(set(tokens_p).difference(tokens_r))

                    tokens_aux = sorted(list(filter(None, list(tokens_aux))))
        else:
            N = int(self.mode)
            for c in range(len(corpus)):
                corpus[c] = " ".join(corpus[c])
            tokens_aux = n_grams(corpus, N)
            tokens_aux = sorted(list(filter(None, list(set(tokens_aux)))))
        
        for token_aux in tokens_aux:
            pos_aux = []
            numbers_aux = []
            regex = re.sub(r'\s+', r'%s' % self.whitespaces.replace('\\', '\\\\'), token_aux)
            regex_numbers = r'%s' % regex.replace(self.pnumbers, '(' + self.pnumbers + ')')
            
            if re.findall(re.escape(self.pnumbers), token_aux):
                idxs_aux = tokens2pos[token_aux]
            else:
                idxs_aux = list(range(len(self.corpus)))

            for c in idxs_aux:
                text_aux = ' '.join(self.corpus[c])
                f_match = match(regex, text_aux, True)
                if len(f_match) > 0:
                    f_numbers = match(regex_numbers, text_aux)
                    if regex != regex_numbers and len(f_numbers) > 0:
                        for number in f_numbers:
                            if type(number) == tuple:
                                numbers_aux.append(list(filter(None, list(number))))
                            elif type(number) == str:
                                numbers_aux.append(list(set(filter(None, list([number])))))
                            pos_aux.append(c)
                    else:
                        pos_aux.append(c)

            pos_aux = np.array(pos_aux)
            numbers_aux = np.array(numbers_aux)
            
            if len(numbers_aux) > 0:
                for i in range(numbers_aux.shape[1]):
                    numbers_aux[:, i] = np.array([re.sub(r'\,', '.', str(number)) for number in numbers_aux[:, i]])
                    numbers_aux[:, i] = replace_outliers(numbers_aux[:, i])
                numbers_aux = numbers_aux.astype(float)

            self.regexes[regex] = [pos_aux, numbers_aux, self.pattern2token, self.pattern2tokens, self.model]
            self.tokens2pos[regex] = tokens2pos[token_aux]

        try:
            import json
            out_dir = os.path.join(os.getcwd(), "out", "regex_rules")
            os.makedirs(out_dir, exist_ok=True)

            rules_simple = {}
            for rgx, payload in self.regexes.items():
                pos_arr = payload[0]
                num_arr = payload[1]
                rules_simple[rgx] = {
                    "positions": np.asarray(pos_arr, dtype=int).tolist(),
                    "numbers": num_arr.tolist() if isinstance(num_arr, np.ndarray) and num_arr.size > 0 else []
                }

            rules_path = os.path.join(out_dir, f"{self.FILENAME}_regexes.json")
            with open(rules_path, "w", encoding="utf-8") as f:
                json.dump(rules_simple, f, ensure_ascii=False, indent=2)

            maps_payload = {
                "pattern2token": dict(self.pattern2token),
                "pattern2tokens": {k: list(v) for k, v in self.pattern2tokens.items()},
                "tokens2pos": {k: list(v) for k, v in self.tokens2pos.items()},
            }
            maps_path = os.path.join(out_dir, f"{self.FILENAME}_maps.json")
            with open(maps_path, "w", encoding="utf-8") as f:
                json.dump(maps_payload, f, ensure_ascii=False, indent=2)

            txt_path = os.path.join(out_dir, f"{self.FILENAME}_regexes.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for rgx, payload in self.regexes.items():
                    coverage = int(payload[0].shape[0]) if isinstance(payload[0], np.ndarray) else 0
                    f.write(f"{coverage:5d}  {rgx}\n")

            print(f"[INFO] Se exportaron {len(rules_simple)} regex a: {rules_path}")
            print(f"[INFO] Mapas de patrones exportados a: {maps_path}")
            print(f"[INFO] Listado plano exportado a: {txt_path}")
        except Exception as e:
            print(f"[WARN] No se pudieron exportar las reglas a disco: {e}")

        del corpus
        gc.collect()