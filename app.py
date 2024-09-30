from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tensorflow import keras
import traceback

app = Flask(__name__)

# Load data from Excel file
df = pd.read_excel('steel_data.xlsx', sheet_name='LOCAL-Behaviour-Non Composite')

# Rename columns if necessary
column_mapping = {
    'hb': 'hb',
    'tbw': 'tw',
    'bbf': 'bf',
    'tbf': 'tf',
    'Lb': 'Lb',
    'do': 'do',
    'Se': 'So',
    'fyn': 'fyb'
}
df = df.rename(columns=column_mapping)

# Calculate new columns
df['hb/tbw'] = df['hb'] / df['tw']
df['bbf/2tbf'] = df['bf'] / (2 * df['tf'])
df['Lb/hb'] = df['Lb'] / df['hb']
df['do/hb'] = df['do'] / df['hb']
df['se/hb'] = df['So'] / df['hb']

app.logger.info(f"Columns in DataFrame after renaming and calculations: {df.columns.tolist()}")
app.logger.info(f"First few rows of DataFrame after renaming and calculations:\n{df.head()}")

# Load scaler
scaler_X = joblib.load('scaler_X.joblib')

input_vars = ['hb/tbw', 'bbf/2tbf', 'Lb/hb', 'do/hb', 'se/hb', 'fyn']
output_params = ['K0', 'as-Plus', 'as-Neg', 'My-Plus', 'My-Neg', 'Lamda-S', 'Lamda-C', 'Lamda_A', 'Lamda_K', 'c_S', 'c_C', 'c_A', 'c_K', 'theta_p_Plus', 'theta_p_Neg', 'theta_pc_Plus', 'theta_pc_Neg', 'Res_Pos', 'Res-Neg', 'theta_u_Plus', 'theta_u_Neg', 'D_Plus', 'D_Neg', 'nFactor']

def calculate_ratios(bf, tf, hb, tw, Lb, do, So, fyb):
    return {
        'hb/tbw': hb / tw,
        'bbf/2tbf': bf / (2 * tf),
        'Lb/hb': Lb / hb,
        'do/hb': do / hb,
        'se/hb': So / hb,
        'fyn': fyb
    }

def check_in_database(input_data):
    epsilon = 1e-6
    for _, row in df.iterrows():
        if all(abs(row[col] - input_data[col]) < epsilon for col in input_data.keys() if col in row.index):
            return True, row
    return False, None

def get_database_ranges(df, input_vars):
    ranges = {}
    for var in input_vars:
        if var in df.columns:
            ranges[var] = (df[var].min(), df[var].max())
        else:
            app.logger.warning(f"Column '{var}' not found in the DataFrame")
            ranges[var] = (None, None)
    return ranges

def predict_params(input_data):
    predictions = {param: {} for param in output_params}
    best_models = {}

    X_scaled = scaler_X.transform(input_data)

    input_dict = {var: input_data[0, i] for i, var in enumerate(input_vars)}

    in_database, db_row = check_in_database(input_dict)

    db_ranges = get_database_ranges(df, input_vars)

    hb_tw_in_range = db_ranges['hb/tbw'][0] <= input_dict['hb/tbw'] <= db_ranges['hb/tbw'][1]

    do_hb_ratio = input_dict['do/hb']

    for param in output_params:
        scaler_y = joblib.load(f'scaler_y_{param}.joblib')
        
        try:
            if os.path.exists(f'{param}_best_model.keras'):
                model = keras.models.load_model(f'{param}_best_model.keras')
                y_scaled_pred = model.predict(X_scaled, verbose=0).flatten()
            else:
                model = joblib.load(f'{param}_best_model.joblib')
                y_scaled_pred = model.predict(X_scaled).reshape(-1, 1)
            
            predictions[param] = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
            
            if not in_database and not hb_tw_in_range:
                if param == 'K0':
                    correction_factor = 1
                    predictions[param] *= correction_factor
                elif param in ['My-Plus', 'My-Neg']:
                    correction_factor = 1
                    predictions[param] *= correction_factor
            elif not in_database:
                if param == 'K0':
                    correction_factor = 1
                    predictions[param] *= correction_factor
                elif param in ['My-Plus', 'My-Neg']:
                    correction_factor = 1
                    predictions[param] *= correction_factor
            
        except Exception as e:
            app.logger.error(f"Error predicting {param}: {str(e)}")
            predictions[param] = None

    return predictions, in_database, db_row, not hb_tw_in_range

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_values = {param: float(data[param]) for param in ['bf', 'tf', 'hb', 'tw', 'Lb', 'do', 'So', 'fyb']}
        ratios = calculate_ratios(**input_values)
        model_input = np.array([[ratios[var] for var in input_vars]])
        predictions, in_database, db_row, hb_tw_out_of_range = predict_params(model_input)

        results = []
        for param in output_params:
            if predictions[param] is not None:
                value = predictions[param][0]
                result = {
                    'param': param,
                    'predicted_value': f"{value:.4f}",
                    'in_database': in_database,
                    'hb_tw_out_of_range': hb_tw_out_of_range
                }
                if in_database:
                    db_value = db_row[param]
                    error_percentage = abs(value - db_value) / db_value * 100
                    result['db_value'] = f"{db_value:.4f}"
                    result['error_percentage'] = f"{error_percentage:.2f}"
                results.append(result)

        return jsonify(results)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test route is working"}), 200

if __name__ == '__main__':
    app.run(debug=True)