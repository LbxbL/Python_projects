# Installation des librairies nécessaires
#!pip install numpy pandas scikit-learn matplotlib


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import pandas as pd

# Charger le fichier d'entraînement en utilisant l'engine 'openpyxl'
train_df = pd.read_excel(r"C:\Users\leand\Desktop\PYTHON\Python projet Angele\entrainement.xlsx", engine='openpyxl') 
# Afficher le contenu du fichier d'entraînement
print("Données d'entraînement:")
print(train_df.head())  # Affiche les premières lignes pour vérifier le chargement

# Charger le fichier de test en utilisant l'engine 'openpyxl'
test_df = pd.read_excel(r"C:\Users\leand\Desktop\PYTHON\Python projet Angele\test.xlsx", engine='openpyxl')
# Afficher le contenu du fichier de test
print("Données de test:")
print(test_df.head())  # Affiche les premières lignes pour vérifier le chargement

# Charger les fichiers d'entraînement et de test
train_data = pd.read_excel(r"C:\Users\leand\Desktop\PYTHON\Python projet Angele\entrainement.xlsx")
test_data = pd.read_excel(r"C:\Users\leand\Desktop\PYTHON\Python projet Angele\test.xlsx")

# Supprimer les lignes avec des valeurs manquantes dans les deux ensembles
train_data = train_data.dropna()
test_data = test_data.dropna()

print("PHASE 1")


# Convertir les variables catégorielles en variables indicatrices pour les deux ensembles
# Assurez-vous d'inclure toutes les variables catégorielles et de les traiter uniformément entre les ensembles
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# Assurer la même structure de colonnes dans les deux ensembles, au cas où certaines catégories sont manquantes dans l'un des ensembles
train_data, test_data = train_data.align(test_data, join='outer', axis=1, fill_value=0)

# Sélectionner la variable cible et les variables explicatives pour l'entraînement
X_train = train_data.drop('Target', axis=1)
y_train = train_data['Target']

# Sélectionner la variable cible et les variables explicatives pour le test
X_test = test_data.drop('Target', axis=1)
y_test = test_data['Target']

# Les ensembles sont maintenant prêts à être utilisés pour entraîner et tester des modèles de machine learning
# Créer une instance du modèle
model = LogisticRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

print("PHASE 2")

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Afficher la matrice de confusion et le rapport de classification
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Extraire les évaluations manuelles
eval_manuelle = test_data['EvaMan']


print("TESTING PHASE")

# Calculer la précision par rapport à l'évaluation manuelle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Comparer les prédictions avec les évaluations manuelles
precision_manuelle = accuracy_score(eval_manuelle, y_pred)
print(f"Précision par rapport à l'évaluation manuelle: {precision_manuelle:.2f}")

# Rapport de classification pour la comparaison
print("Rapport de classification par rapport à l'évaluation manuelle:")
print(classification_report(eval_manuelle, y_pred))

# Matrice de confusion pour la comparaison
print("Matrice de confusion par rapport à l'évaluation manuelle:")
print(confusion_matrix(eval_manuelle, y_pred))