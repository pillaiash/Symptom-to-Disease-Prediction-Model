from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and the column names for symptoms
best_model = joblib.load('best_model.pkl')
symptoms_columns = joblib.load('symptoms_columns.pkl')

# Function to predict disease based on user input considering probabilities
def predict_disease(symptoms_input, days):
    input_data = pd.Series(symptoms_input).str.get_dummies().reindex(columns=symptoms_columns, fill_value=0).sum()
    input_data['days'] = days  # Include days as a feature
    input_data = pd.DataFrame([input_data])

    # Get probabilities for each disease
    probas = best_model.predict_proba(input_data)

    # Get the disease with the highest probability
    disease_index = probas.argmax(axis=1)
    prediction = best_model.classes_[disease_index][0]

    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data['symptoms']
    days = int(data['days'])
    prediction = predict_disease(symptoms, days)
    return jsonify({'disease': prediction})

@app.route('/search_symptoms', methods=['GET'])
def search_symptoms():
    query = request.args.get('query', '').lower()
    matching_symptoms = [col for col in symptoms_columns if query in col.lower()]
    return jsonify(matching_symptoms)

if __name__ == '__main__':
    app.run(debug=True)

