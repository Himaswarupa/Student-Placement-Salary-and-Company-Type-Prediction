from flask import Flask, request, render_template
import numpy as np
import joblib
from functools import lru_cache

app = Flask(__name__)

# company mapping logic
COMPANY_TYPES = {
    (400000, 600000): "Core Engineering",
    (600000, 800000): "Startup",
    (800000, 1000000): "IT",
    (1000000, 1300000): "MNC"
}

# loading saved models
@lru_cache(maxsize=1)
def load_models():
    placement_model = joblib.load('models/placement_model.pkl')
    salary_model = joblib.load('models/salary_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return placement_model, salary_model, scaler

def get_company_type(salary):
    """Determine company type based on salary range"""
    for (min_sal, max_sal), company_type in COMPANY_TYPES.items():
        if min_sal <= salary < max_sal:
            return company_type
    return "Unknown"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict_form')
def predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        placement_model, salary_model, scaler = load_models()

        features = np.array([[
            float(request.form['cgpa']),
            float(request.form['major_projects']),
            float(request.form['certifications']),
            float(request.form['mini_projects']),
            float(request.form['skills']),
            float(request.form['communication']),
            float(request.form['internship']),
            float(request.form['hackathon']),
            float(request.form['twelve_percentage']),
            float(request.form['ten_percentage']),
            float(request.form['backlogs']),
        ]])

        features_scaled = scaler.transform(features)

        placement_prediction = placement_model.predict(features_scaled)[0]

        name = request.form['name']
        course = request.form['course']
        sap_id = request.form['sap_id']

        if placement_prediction == 1:
            predicted_salary = salary_model.predict(features_scaled)[0]
            company_type = get_company_type(predicted_salary)

            return render_template('result.html',
                                name=name,
                                sap_id=sap_id,
                                course=course,
                                placed=True,
                                salary=f"{predicted_salary:,.2f}",
                                company_type=company_type)
        else:
            return render_template('result.html',
                                name=name,
                                sap_id=sap_id,
                                course=course,
                                placed=False,
                                salary=None,
                                company_type=None)
    
    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
# runs on all devices connected to same internet 
