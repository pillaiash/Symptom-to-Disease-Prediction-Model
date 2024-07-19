# Symptom-to-Disease Prediction Model

This repository contains a Symptom-to-Disease Prediction Model designed to predict medical conditions based on user-reported symptoms. The project includes Python scripts for training and deploying the model, as well as an HTML file for the web-based user interface.

## Project Structure
├── main.py # Main script for running the application.

├── version_2.py # Additional script with extended functionality.

├── best_model.pkl # Trained model in pickle format.

├── symptoms_columns.pkl # Pickle file with the list of symptoms columns.

├── templates/

  └── index.html # HTML file for the web interface

└── README.md # Project README file

## Features

- Predicts diseases based on user-reported symptoms.
- Provides a user-friendly web interface for interaction.

## Installation

1. Clone the repository:
   git clone https://github.com/pillaiash/Symptom-to-Disease-Prediction-Model.git
   
   cd symptom-to-disease-prediction

2. Create and activate a virtual environment:
   python -m venv env
   
# On Windows
.\env\Scripts\activate

# On macOS and Linux
source env/bin/activate

3. Install the required dependencies:
   pip install -r requirements.txt

# Usage (Run)
python app.py

# Access the web interface:
Open your web browser and navigate to http://127.0.0.1:5000.




