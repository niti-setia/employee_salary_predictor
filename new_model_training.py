# new_model_training.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Dummy data
data = {
    'experience': [1, 3, 5, 7, 9],
    'test_score': [80, 70, 88, 92, 85],
    'interview_score': [70, 65, 90, 95, 85],
    'salary': [40000, 50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

X = df[['experience', 'test_score', 'interview_score']]
y = df['salary']

model = LinearRegression()
model.fit(X, y)

# Save this new model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained & saved!")
