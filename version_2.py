import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load the dataset from CSV file
data = pd.read_csv('symptom_disease_dataset.csv')

# Convert symptoms to a suitable format (one-hot encoding)
symptoms = pd.get_dummies(data[['symptom_1', 'symptom_2', 'symptom_3']].stack()).groupby(level=0).sum()

# Combine the one-hot encoded symptoms with the disease labels and days
dataset = pd.concat([symptoms, data['days'], data['disease']], axis=1)

# Split data into features (X) and target (y)
X = dataset.drop('disease', axis=1)
y = dataset['disease']

# Oversample the data to balance the classes
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Split resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                    stratify=y_resampled)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Use StratifiedKFold to ensure the splits have the same proportion of classes as the dataset
cv = StratifiedKFold(n_splits=3)

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Function to predict disease based on user input considering probabilities
def predict_disease(symptom_input, days):
    input_symptoms = symptom_input.split(',')
    input_data = pd.Series(input_symptoms).str.get_dummies().reindex(columns=X.columns[:-1], fill_value=0).sum()
    input_data['days'] = days  # Include days as a feature
    input_data = pd.DataFrame([input_data])

    # Get probabilities for each disease
    probas = best_model.predict_proba(input_data)

    # Get the disease with the highest probability
    disease_index = probas.argmax(axis=1)
    prediction = best_model.classes_[disease_index][0]

    return prediction


# Example usage
#user_symptoms = "skin_rash,dry_skin,itching"
#days = 6
#print(predict_disease(user_symptoms, days))


# Assuming 'best_model' is your trained RandomForestClassifier model
 joblib.dump(best_model, 'best_model.pkl')

# # Save the columns of the one-hot encoded symptoms
 joblib.dump(list(X.columns[:-1]), 'symptoms_columns.pkl')
