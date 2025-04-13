# Student Placement, Salary, and Company-Type Prediction System

### Overview
This system uses machine learning to predict students' placement status, expected salary, and suitable company types based on academic performance, skills, and other metrics. Our goal is to help students make informed career decisions and optimize their placement preparation.

### Problem Statement
Students often lack insight into their placement potential and suitable career paths. This leads to:
- Suboptimal preparation for placement processes
- Poor matching between students and employers
- Missed opportunities and salary negotiations below potential
- Lack of personalized guidance for career planning

### Solution
Our web application leverages machine learning to provide personalized placement insights:
1. **Placement Status Prediction**: Predicts whether a student will be placed based on their profile
2. **Salary Prediction**: Estimates expected salary for students predicted to be placed
3. **Company Type Recommendation**: Suggests suitable company categories based on predicted salary ranges

### Final Project Progress 

#### Completed Tasks
- Data collection and preprocessing pipeline
- Implementation of Random Forest Classifier for placement prediction
- Implementation of Random Forest Regressor for salary prediction
- Company type mapping logic based on salary brackets
- Fully functional Flask web application
- Integrated HTML/CSS UI for user input and result display
- Visualization components for results
- Final documentation and testing

### Technology Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

### How to Run the Project
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the models: `python Model_Training.py`
4. Run the application: `python app.py`
5. Access the web interface at `http://localhost:5000`

### Preliminary Results
- Placement Prediction Accuracy: 93.50%
- Placement Prediction AUC Score: 98.56%
- Placemnt Prediction Confusion matrix: [ [ 1098 74 ]
                                          [ 56 772 ] ]
- Salary Prediction R² Score: 93.32%
- Mean Absolute Error for Salary: 54617.33
- Mean Squared Error for Salary: 68732.63

### Future Advancements
- Resume Parsing Integration
- Real-Time Data Integration
- Skill Gap Analysis and Recommendations
- User Dashboard with Progress Tracking
- Administrative Portal for T&P Cells
- Mobile Application Development
- Advanced Machine Learning Models
