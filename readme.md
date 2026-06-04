# Prédiction de Défaut de Prêt Immobilier

## Description
Application de Machine Learning pour prédire si un emprunteur sera en défaut 
de paiement sur son prêt immobilier.

## Dataset
- Source : Kaggle - Loan Default Dataset
- Taille : 148 670 lignes, 34 variables
- Target : Status (0 = remboursement normal, 1 = défaut)

## Modèle
- Algorithme : Random Forest Classifier
- AUC-ROC : 0.997
- F1-score classe 1 : 0.96

## Application
🔗 [LIEN]([URL_STREAMLIT_ICI](https://projet-loan-default-rania.streamlit.app/))

## Structure du projet
- `app.py` : Application Streamlit
- `model_rf.pkl` : Modèle sérialisé
- `scaler.pkl` : Scaler sérialisé
- `notebook.ipynb` : Notebook complet
- `rapport.pdf` : Rapport du projet
