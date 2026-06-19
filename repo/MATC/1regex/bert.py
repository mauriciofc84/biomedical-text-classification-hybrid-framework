import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import dirname as up
from utils import SEED

# --- IMPORTS DIRECTOS (SIN TRY/EXCEPT) ---
# Esto hará que si falta una librería, el error salga AQUÍ y no después.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam, lr_scheduler, AdamW
import torch.nn.functional as F
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
from transformers import BertModel, DistilBertModel, AlbertModel
from transformers import get_linear_schedule_with_warmup
from transformers import set_seed
# -----------------------------------------

class Texts(Dataset):
    def __init__(self, texts, labels, max_len, path_model, cased, bert_type):
        self.texts = copy.deepcopy(texts)
        self.labels = copy.deepcopy(labels)
        
        # --- CORRECCIÓN AQUÍ: Agregamos 'biobert' ---
        if bert_type=='bert' or bert_type=='biobert': 
            self.tokenizer = BertTokenizer.from_pretrained(
                path_model, do_lower_case=cased
            )
        # --------------------------------------------
        
        elif bert_type=='distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                path_model, do_lower_case=cased
            )
        elif bert_type=='albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(
                path_model, do_lower_case=cased
            )
        self.max_len = max_len
         
    def __len__(self):
        return (len(self.texts))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = str(self.texts[idx]) 
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
            )

        return {
            'label': torch.tensor(label, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten()
        }


class BertClassifier(nn.Module):
    def __init__(self, n_classes, dropout, path_model, bert_type):
        super(BertClassifier, self).__init__()

        self.bert_type = bert_type
        if bert_type=='bert':
            self.bert = BertModel.from_pretrained(path_model)  
            hdim = 768
        elif bert_type=='biobert':
            self.bert = BertModel.from_pretrained(path_model)  
            hdim = 768
        elif bert_type=='distilbert':
            self.bert = DistilBertModel.from_pretrained(path_model)
            hdim = 768
        elif bert_type=='albert':
            self.bert = AlbertModel.from_pretrained(path_model)  
            hdim = 312
        
        self.fc = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(hdim, n_classes)
                               )
    def forward(self, ids, mask):

        if self.bert_type == 'bert' or self.bert_type == 'biobert':
            # return_dict=False para compatibilidad
            _, pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)
        elif self.bert_type == 'distilbert':
            pooled_output = self.bert(ids, attention_mask = mask)
            pooled_output = pooled_output[0][:,0]
        elif self.bert_type == 'albert':
            _, pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)

        output = self.fc(pooled_output)
        return output

class BERT(object):
    def __init__(
        self,
        n_classes,
        scheduler_opt,
        early_stopping,
        validation_split,
        val_loss_min,
        patience,
        batch_size,
        epochs,
        dropout,
        MAX_SENT_LEN,
        lr,
        RUNS,
        bert_type,
        SEED=SEED
    ):

        # --- DETECCIÓN INTELIGENTE DE HARDWARE ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.gpu = 'cuda:0'
            print(f"[INFO] BERT usará GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.gpu = 'cpu'
            print("[INFO] BERT usará CPU (Advertencia: Será lento).")

        self.scheduler_opt = scheduler_opt

        # Seleccionamos nombres de modelos oficiales de HuggingFace
        if bert_type == 'albert':
            model_bert = "albert-base-v2" 
        elif bert_type == 'distilbert':
            model_bert = "distilbert-base-multilingual-cased"
        elif bert_type == 'bert':
            model_bert = "bert-base-multilingual-uncased"
        elif bert_type == 'biobert':
            model_bert = "dmis-lab/biobert-base-cased-v1.2"

        # Usamos el nombre directo para descargar
        self.path_model = model_bert 
        self.cased = "uncased" not in model_bert
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.val_loss_min = val_loss_min
        self.patience = patience
        self.dropout = dropout
        self.MAX_SENT_LEN = MAX_SENT_LEN
        self.lr = lr
        self.RUNS = RUNS
        self.n_classes = n_classes
        self.SEED = SEED
        self.bert_type = bert_type

    def reset_linear(self, m):
        if type(m) == nn.Linear:
            m.reset_parameters()

    def fit(self, X, y):
        
        # Limpieza de memoria segura
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        if self.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_split,
                shuffle=False,
                random_state=self.SEED,
            )
        else:
            X_train = copy.deepcopy(X)
            y_train = copy.deepcopy(y)

        train_data = Texts(X_train, y_train, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        if self.validation_split > 0:
            val_data = Texts(X_val, y_val, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
            val_sampler = RandomSampler(val_data)
            val_dataloader = DataLoader(
                val_data,
                sampler=val_sampler,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            )

        self.clf = BertClassifier(self.n_classes, self.dropout, self.path_model, self.bert_type)
        optimizer = Adam(self.clf.parameters(), lr=self.lr)
        self.clf.to(self.device)

        if self.scheduler_opt:
            total_steps = len(train_dataloader) * self.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )

        epochs_stop = 0
        self.loss_training = []
        self.loss_val = []
        fcn = nn.CrossEntropyLoss()

        print(f"[INFO] Iniciando entrenamiento por {self.epochs} épocas...")

        for epoch_i in range(0, self.epochs):
            train_loss = 0
            self.clf.train()
            
            for step, batch in enumerate(train_dataloader):
                
                # Limpieza de memoria solo si es GPU
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.gpu):
                        torch.cuda.empty_cache()
                
                optimizer.zero_grad()

                # Mover tensores al dispositivo (GPU o CPU)
                labels = batch['label'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                input_masks = batch['attention_mask'].to(self.device)
                
                # Forward pass sin autocast (más estable en CPU)
                logits = self.clf(input_ids, input_masks)
                batch_loss = fcn(logits.view(-1, self.n_classes), labels.view(-1))

                del input_ids, input_masks, labels

                train_loss += batch_loss.item()
                batch_loss.backward()
                
                if self.scheduler_opt:
                    clip_grad_norm_(parameters=self.clf.parameters(), max_norm=1.0)
                
                optimizer.step()

                if self.scheduler_opt:
                    scheduler.step()

            train_loss /= len(train_dataloader.dataset)
            self.loss_training.append(train_loss)
            print(f"Epoch {epoch_i+1}/{self.epochs} - Loss: {train_loss:.4f}")

            if self.validation_split > 0:
                val_loss = 0
                self.clf.eval()
                for step, batch in enumerate(val_dataloader):
                    
                    if self.device.type == 'cuda':
                        with torch.cuda.device(self.gpu):
                            torch.cuda.empty_cache()

                    labels = batch['label'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    input_masks = batch['attention_mask'].to(self.device)

                    with torch.no_grad():
                        logits = self.clf(input_ids, input_masks)
                        batch_loss = fcn(logits.view(-1, self.n_classes), labels.view(-1))

                    del input_ids, input_masks, labels
                    val_loss += batch_loss.item()

                val_loss /= len(val_dataloader.dataset)
                self.loss_val.append(val_loss)

                if self.early_stopping:
                    if val_loss < self.val_loss_min:
                        self.val_loss_min = val_loss
                        epochs_stop = 0
                        params_model = copy.deepcopy(self.clf.state_dict())
                    else:
                        epochs_stop += 1
                    if epochs_stop >= self.patience:
                        self.clf.load_state_dict(params_model)
                        break

        del train_dataloader
        if self.validation_split > 0:
            del val_dataloader

    def predict(self, X_test):
        print('predicting...')
        y = np.zeros(len(X_test))
        prediction_data = Texts(X_test, y, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
        prediction_sampler = SequentialSampler(prediction_data)
        test_dataloader = DataLoader(
            prediction_data,
            sampler=prediction_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        self.clf.eval()
        predictions = []
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.gpu):
                        torch.cuda.empty_cache()

                # No necesitamos labels para predecir, pero el dataloader los devuelve
                input_ids = batch['input_ids'].to(self.device)
                input_masks = batch['attention_mask'].to(self.device)
                
                logits = self.clf(input_ids, input_masks)

                del input_ids, input_masks
                
                logits = F.softmax(logits, dim=1)
                logits = logits.detach().cpu().numpy()
                predictions += list(np.argmax(logits, axis=1))

        return np.array(predictions, dtype=int)

    def apply_dropout(self, m):
        if type(m) == nn.Dropout:
            m.train()

    def predict_proba(self, X_u):
        y = np.zeros(len(X_u))
        prediction_data = Texts(X_u, y, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
        prediction_sampler = SequentialSampler(prediction_data)
        test_dataloader = DataLoader(
            prediction_data,
            sampler=prediction_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        self.clf.eval()
        self.clf.apply(self.apply_dropout)
        probs = []
        for times in range(self.RUNS):
            logits_sum = np.array([])
            with torch.no_grad():
                for step, batch in enumerate(test_dataloader):
                    
                    if self.device.type == 'cuda':
                        with torch.cuda.device(self.gpu):
                            torch.cuda.empty_cache()

                    input_ids = batch['input_ids'].to(self.device)
                    input_masks = batch['attention_mask'].to(self.device)

                    logits = self.clf(input_ids, input_masks)

                    del input_ids, input_masks
                    
                    logits = F.softmax(logits, dim=1)
                    logits = logits.detach().cpu().numpy()
                    if len(logits_sum) == 0:
                        logits_sum = copy.deepcopy(logits)
                    else:
                        logits_sum = np.vstack((logits_sum, logits))
            probs.append(logits_sum)
        probs = np.mean(probs, axis=0)
        return probs
