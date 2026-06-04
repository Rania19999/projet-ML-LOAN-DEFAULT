import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Chargement du modèle et du scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# Titre de l'application
st.title("🏦 Prédiction de Défaut de Prêt Immobilier")
st.markdown("---")

# Saisie des features
st.header("📋 Informations sur le prêt")

col1, col2 = st.columns(2)

with col1:
    Upfront_charges = st.number_input("Frais initiaux (Upfront charges)", min_value=0.0)
    term = st.number_input("Durée du prêt (mois)", min_value=96, max_value=360, value=360)
    property_value = st.number_input("Valeur du bien (€)", min_value=0.0)
    income = st.number_input("Revenu annuel (€)", min_value=0.0)
    LTV = st.number_input("LTV (%)", min_value=0.0, max_value=120.0)
    dtir1 = st.number_input("Ratio dette/revenu", min_value=0.0)

with col2:
    loan_limit = st.selectbox("Limite du prêt", ['cf', 'ncf'])
    Gender = st.selectbox("Genre", ['Male', 'Female', 'Joint', 'Sex Not Available'])
    approv_in_adv = st.selectbox("Pré-approuvé", ['pre', 'nopre'])
    loan_type = st.selectbox("Type de prêt", ['type1', 'type2', 'type3'])
    Credit_Worthiness = st.selectbox("Solvabilité", ['l1', 'l2'])
    open_credit = st.selectbox("Crédit ouvert", ['opc', 'nopc'])

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    business_or_commercial = st.selectbox("Usage", ['nob/c', 'b/c'])
    Neg_ammortization = st.selectbox("Amortissement négatif", ['not_neg', 'neg_amm'])
    interest_only = st.selectbox("Intérêts seulement", ['not_int', 'int_only'])
    lump_sum_payment = st.selectbox("Paiement forfaitaire", ['not_lpsm', 'lpsm'])
    construction_type = st.selectbox("Type de construction", ['sb', 'mh'])

with col4:
    occupancy_type = st.selectbox("Type d'occupation", ['pr', 'sr', 'ir'])
    Secured_by = st.selectbox("Garantie", ['home', 'land'])
    total_units = st.selectbox("Nombre d'unités", ['1U', '2U', '3U', '4U'])
    credit_type = st.selectbox("Bureau de crédit", ['CIB', 'CRIF', 'EQUI', 'EXP'])
    co_applicant_credit_type = st.selectbox("Bureau co-emprunteur", ['CIB', 'EXP'])
    submission = st.selectbox("Soumission", ['to_inst', 'not_inst'])
    Region = st.selectbox("Région", ['North', 'south', 'central', 'North-East'])
    Security_Type = st.selectbox("Type de sécurité", ['direct', 'Indriect'])
# Bouton de prédiction
# Bouton de prédiction
st.markdown("---")
if st.button("🔍 Prédire"):

    # Affichage des données saisies
    st.header("📊 Données saisies")
    input_data = {
        'Upfront_charges': Upfront_charges,
        'term': term,
        'property_value': property_value,
        'income': income,
        'LTV': LTV,
        'dtir1': dtir1,
        'loan_limit': loan_limit,
        'Gender': Gender,
        'approv_in_adv': approv_in_adv,
        'loan_type': loan_type,
        'Credit_Worthiness': Credit_Worthiness,
        'open_credit': open_credit,
        'business_or_commercial': business_or_commercial,
        'Neg_ammortization': Neg_ammortization,
        'interest_only': interest_only,
        'lump_sum_payment': lump_sum_payment,
        'construction_type': construction_type,
        'occupancy_type': occupancy_type,
        'Secured_by': Secured_by,
        'total_units': total_units,
        'credit_type': credit_type,
        'co_applicant_credit_type': co_applicant_credit_type,
        'submission': submission,
        'Region': Region,
        'Security_Type': Security_Type
    }
    st.dataframe(pd.DataFrame([input_data]))

    # Preprocessing
    # One-Hot Encoding manuel
    features = {
        'Upfront_charges': Upfront_charges,
        'term': term,
        'property_value': property_value,
        'income': income,
        'LTV': LTV,
        'dtir1': dtir1,
        'loan_limit_cf': 1 if loan_limit == 'cf' else 0,
        'Gender_Joint': 1 if Gender == 'Joint' else 0,
        'approv_in_adv_nopre': 1 if approv_in_adv == 'nopre' else 0,
        'loan_type_type1': 1 if loan_type == 'type1' else 0,
        'Credit_Worthiness_l1': 1 if Credit_Worthiness == 'l1' else 0,
        'open_credit_nopc': 1 if open_credit == 'nopc' else 0,
        'business_or_commercial_nob/c': 1 if business_or_commercial == 'nob/c' else 0,
        'Neg_ammortization_neg_amm': 1 if Neg_ammortization == 'neg_amm' else 0,
        'Neg_ammortization_not_neg': 1 if Neg_ammortization == 'not_neg' else 0,
        'interest_only_not_int': 1 if interest_only == 'not_int' else 0,
        'lump_sum_payment_lpsm': 1 if lump_sum_payment == 'lpsm' else 0,
        'lump_sum_payment_not_lpsm': 1 if lump_sum_payment == 'not_lpsm' else 0,
        'construction_type_sb': 1 if construction_type == 'sb' else 0,
        'occupancy_type_pr': 1 if occupancy_type == 'pr' else 0,
        'Secured_by_home': 1 if Secured_by == 'home' else 0,
        'total_units_1U': 1 if total_units == '1U' else 0,
        'credit_type_CIB': 1 if credit_type == 'CIB' else 0,
        'credit_type_CRIF': 1 if credit_type == 'CRIF' else 0,
        'credit_type_EQUI': 1 if credit_type == 'EQUI' else 0,
        'credit_type_EXP': 1 if credit_type == 'EXP' else 0,
        'co-applicant_credit_type_CIB': 1 if co_applicant_credit_type == 'CIB' else 0,
        'co-applicant_credit_type_EXP': 1 if co_applicant_credit_type == 'EXP' else 0,
        'submission_of_application_not_inst': 1 if submission == 'not_inst' else 0,
        'submission_of_application_to_inst': 1 if submission == 'to_inst' else 0,
        'Region_North': 1 if Region == 'North' else 0,
        'Region_south': 1 if Region == 'south' else 0,
        'Security_Type_direct': 1 if Security_Type == 'direct' else 0,
    }

    # Créer dataframe
    df_input = pd.DataFrame([features])

    # Normaliser les colonnes numériques
    num_cols = ['Upfront_charges', 'property_value', 'income', 'LTV', 'dtir1']
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    # Prédiction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    # Affichage résultat
    st.markdown("---")
    st.header("🎯 Résultat de la prédiction")

    if prediction == 1:
        st.error(f"⚠️ **DÉFAUT DE PAIEMENT PROBABLE**")
    else:
        st.success(f"✅ **REMBOURSEMENT NORMAL PROBABLE**")

    st.metric("Niveau de confiance (probabilité de défaut)", 
              f"{probability:.1%}")