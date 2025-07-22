# Employee Salary Prediction (ML + Streamlit)

This project predicts the salary of an employee based on experience, test score, and interview score using a Linear Regression model.

## Features
- Built using Streamlit for interactive UI
- CSV upload support for batch predictions
- Actual vs Predicted salary visualization
- Trained using dummy data

## How to Run
1. Clone this repository
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the app:  
   `streamlit run app.py`

## Files
- `app.py`: Main Streamlit app
- `employee_salary.ipynb`: Model training and visualization
- `train_model.py`: Script to train and save model
- `salary_model.pkl`: Serialized trained model
