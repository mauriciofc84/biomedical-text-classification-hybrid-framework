import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ================= CONFIGURACIÓN =================
DATASET_NAME = "TEST"  # Asegúrate que sea el mismo nombre usado en main.py
MODEL_NAME = "regex"
CURVE_NAME = "LC"
FOLDS = 5

CLASES_MAP = {
    0: "Enfermedad cardiovascular",
    1: "Enfermedad endocrina o metabólica",
    2: "Trastorno mental o del comportamiento"
}

# Ruta base
BASE_DIR = os.path.join(os.getcwd(), "out", "RESULTSLC", CURVE_NAME, DATASET_NAME)

print(f"--- ANALIZANDO RESULTADOS DE: {DATASET_NAME} ({MODEL_NAME}) ---")

y_true_total = []
y_pred_total = []
f1_scores_folds = []
total_uncertain = 0
total_samples = 0

for k in range(1, FOLDS + 1):
    filename = f"{DATASET_NAME}_{MODEL_NAME}_{CURVE_NAME}_k{k}.pkl"
    filepath = os.path.join(BASE_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"[WARN] No se encontró el archivo del Fold {k}")
        continue
        
    print(f"Procesando Fold {k}...", end=" ")
    
    with open(filepath, "rb") as f:
        results = pickle.load(f)
    
    # results[2] es el historial de predicciones del Active Learning
    preds_history = results[2]
    # results[8] son las etiquetas reales
    y_test_fold = results[8]
    
    if len(preds_history) == 0:
        print("-> Vacío.")
        continue

    # 1. OBTENER LA ÚLTIMA PREDICCIÓN
    raw_preds = preds_history[-1]
    
    # 2. LIMPIEZA EXTREMA DE DATOS (AQUÍ ESTÁ LA SOLUCIÓN AL ERROR)
    # Convertimos a numpy array
    y_pred_fold = np.array(raw_preds)
    y_true_fold = np.array(y_test_fold)

    # Si y_pred es una matriz de probabilidades (N, Clases), hacemos argmax
    if y_pred_fold.ndim > 1 and y_pred_fold.shape[1] > 1:
        y_pred_fold = np.argmax(y_pred_fold, axis=1)
    
    # APLANAR: Aseguramos que sean vectores 1D (N,) y no matrices (N, 1)
    y_pred_fold = y_pred_fold.ravel()
    y_true_fold = y_true_fold.ravel()

    # FORZAR ENTEROS: Evitamos floats que confundan a sklearn
    y_pred_fold = y_pred_fold.astype(int)
    y_true_fold = y_true_fold.astype(int)

    # 3. FILTRADO DE INCERTIDUMBRE (-1)
    # Creamos la máscara booleana
    mask_confident = (y_pred_fold != -1)
    
    # Contabilizamos estadísticas
    n_uncertain = np.sum(~mask_confident)
    n_total_fold = len(y_true_fold)
    
    total_uncertain += n_uncertain
    total_samples += n_total_fold
    
    # Aplicamos la máscara
    if np.sum(mask_confident) > 0:
        y_true_filtered = y_true_fold[mask_confident]
        y_pred_filtered = y_pred_fold[mask_confident]
        
        # DEBUG: Verificar que no queden cosas raras
        # print(f"Unique pred: {np.unique(y_pred_filtered)}") 
        
        f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted')
        f1_scores_folds.append(f1)
        
        # Acumular para reporte global
        y_true_total.extend(y_true_filtered)
        y_pred_total.extend(y_pred_filtered)
        
        print(f"-> F1: {f1:.4f} (Ignorados: {n_uncertain}/{n_total_fold})")
    else:
        print(f"-> F1: N/A (Todo ignorado por baja confianza)")

# ================= RESULTADOS GLOBALES =================
y_true_total = np.array(y_true_total)
y_pred_total = np.array(y_pred_total)

if len(y_true_total) == 0:
    print("\n❌ ERROR CRÍTICO: El modelo no tiene predicciones confiables.")
else:
    acc_global = accuracy_score(y_true_total, y_pred_total)
    f1_global = f1_score(y_true_total, y_pred_total, average='weighted')
    coverage = 100 * (1 - (total_uncertain / total_samples)) if total_samples > 0 else 0

    print("\n" + "="*50)
    print(f"RESULTADOS FINALES (Promedio 5 Folds)")
    print("="*50)
    print(f"Muestras Totales Evaluadas: {total_samples}")
    print(f"Muestras 'No sé' (-1):     {total_uncertain}")
    print(f"COBERTURA DEL MODELO:       {coverage:.2f}%")
    print("-" * 50)
    print(f"Accuracy (sobre respuestas): {acc_global:.4f}")
    print(f"F1-Score (sobre respuestas): {f1_global:.4f}")
    if f1_scores_folds:
        print(f"Promedio F1 Folds:           {np.mean(f1_scores_folds):.4f}")
    print("="*50)

    # ================= GRÁFICOS =================
    plt.figure(figsize=(10, 8))
    
    # Obtener etiquetas únicas presentes para evitar errores en el gráfico
    unique_labels = sorted(list(set(y_true_total) | set(y_pred_total)))
    target_names = [CLASES_MAP.get(i, str(i)) for i in unique_labels]

    cm = confusion_matrix(y_true_total, y_pred_total, labels=unique_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    
    plt.title(f"Matriz de Confusión Global\n(Cobertura: {coverage:.1f}% - Accuracy: {acc_global:.1%})")
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("grafico_matriz_global.png")
    print(" -> Gráfico guardado: grafico_matriz_global.png")

    # Reporte de texto
    print("\nREPORTE DETALLADO:")
    print(classification_report(y_true_total, y_pred_total, target_names=target_names))
    print("\n[PROCESO TERMINADO CON ÉXITO]")