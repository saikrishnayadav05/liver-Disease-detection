# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Create a Flask app
app = Flask(__name__)

# Load the dataset
df_liver = pd.read_csv('Liver_data.csv')

# Check for null values and drop them
df_liver = df_liver.dropna()

# Replace gender column with integer values
df_liver['Gender'] = df_liver['Gender'].replace({'Male': 1, 'Female': 0})

# Split the dataset into features and target variable
X = df_liver.drop(columns='Dataset', axis=1)
Y = df_liver['Dataset']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Train a Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define a function to make predictions
def make_prediction(input_data):
    input_data = pd.DataFrame(input_data, columns=X.columns)
    prediction = model.predict(input_data)
    return prediction[0]

# Create a frontend to input patient data and get predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        prediction = make_prediction(input_data)
        return jsonify({'prediction': prediction})
    return '''
        <h1>Liver Disease Detection</h1>
        <form action="" method="post">
            <label for="Age">Age:</label><br>
            <input type="number" id="Age" name="Age" required><br>
            <label for="Gender">Gender:</label><br>
            <select id="Gender" name="Gender" required>
                <option value="0">Female</option>
                <option value="1">Male</option>
            </select><br>
            <label for="Total_Bilirubin">Total Bilirubin:</label><br>
            <input type="number" id="Total_Bilirubin" name="Total_Bilirubin" required><br>
            <label for="Direct_Bilirubin">Direct Bilirubin:</label><br>
            <input type="number" id="Direct_Bilirubin" name="Direct_Bilirubin" required><br>
            <label for="Alkaline_Phosphotase">Alkaline Phosphotase:</label><br>
            <input type="number" id="Alkaline_Phosphotase" name="Alkaline_Phosphotase" required><br>
            <label for="Alamine_Aminotransferase">Alamine Aminotransferase:</label><br>
            <input type="number" id="Alamine_Aminotransferase" name="Alamine_Aminotransferase" required><br>
            <label for="Aspartate_Aminotransferase">Aspartate Aminotransferase:</label><br>
            <input type="number" id="Aspartate_Aminotransferase" name="Aspartate_Aminotransferase" required><br>
            <label for="Total_Proteins">Total Proteins:</label><br>
            <input type="number" id="Total_Proteins" name="Total_Proteins" required><br>
            <label for="Albumin">Albumin:</label><br>
            <input type="number" id="Albumin" name="Albumin" required><br>
            <input type="submit" value="Predict">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)