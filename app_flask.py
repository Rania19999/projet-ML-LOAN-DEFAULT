from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Chargement du modèle et du scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données du formulaire
    Upfront_charges = float(request.form['Upfront_charges'])
    term = float(request.form['term'])
    property_value = float(request.form['property_value'])
    income = float(request.form['income'])
    LTV = float(request.form['LTV'])
    dtir1 = float(request.form['dtir1'])
    loan_limit = request.form['loan_limit']
    Gender = request.form['Gender']
    approv_in_adv = request.form['approv_in_adv']
    loan_type = request.form['loan_type']
    Credit_Worthiness = request.form['Credit_Worthiness']
    open_credit = request.form['open_credit']
    business_or_commercial = request.form['business_or_commercial']
    Neg_ammortization = request.form['Neg_ammortization']
    interest_only = request.form['interest_only']
    lump_sum_payment = request.form['lump_sum_payment']
    construction_type = request.form['construction_type']
    occupancy_type = request.form['occupancy_type']
    Secured_by = request.form['Secured_by']
    total_units = request.form['total_units']
    credit_type = request.form['credit_type']
    co_applicant_credit_type = request.form['co_applicant_credit_type']
    submission = request.form['submission']
    Region = request.form['Region']
    Security_Type = request.form['Security_Type']

    # Preprocessing - One-Hot Encoding manuel
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

    df_input = pd.DataFrame([features])

    # Normalisation
    num_cols = ['Upfront_charges', 'property_value', 'income', 'LTV', 'dtir1']
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    # Prédiction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    result = "DÉFAUT DE PAIEMENT PROBABLE" if prediction == 1 else "REMBOURSEMENT NORMAL PROBABLE"
    color = "red" if prediction == 1 else "green"

    return render_template('index.html', 
                         result=result, 
                         probability=f"{probability:.1%}",
                         color=color)

if __name__ == '__main__':
    app.run(debug=True)