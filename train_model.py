# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Dummy data (experience, test score, interview score)
data = {
    'experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'test_score': [6, 7, 8, 9, 6.5, 7.5, 8.5, 7.2, 8.3, 9.1],
    'interview_score': [7, 8, 9, 9.5, 6, 7.5, 8, 9, 7.5, 8.2],
    'salary': [30000, 35000, 40000, 45000, 32000, 37000, 42000, 43000, 41000, 46000]
}

df = pd.DataFrame(data)

X = df[['experience', 'test_score', 'interview_score']]
y = df['salary']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('salary_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl")
