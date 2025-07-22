import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load trained model
model = pickle.load(open('salary_model.pkl', 'rb'))

# Set bright page theme
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f2f6fc;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict salary based on experience, test score, and interview score")

# Input section
st.header("🎯 Predict Salary for a Single Employee")
experience = st.slider("Years of Experience", 0, 30, 1)
test_score = st.slider("Test Score (out of 100)", 0, 100, 50)
interview_score = st.slider("Interview Score (out of 100)", 0, 100, 50)

if st.button("🔮 Predict Salary"):
    input_data = np.array([[experience, test_score, interview_score]])
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Salary: ₹{int(prediction[0])}")

    # Add prediction to graph
    st.subheader("📈 Graph: Actual vs Predicted + Your Prediction")
    training_data = {
        'experience': [1, 3, 5, 7, 9],
        'test_score': [80, 70, 88, 92, 85],
        'interview_score': [70, 65, 90, 95, 85],
        'salary': [40000, 50000, 60000, 70000, 80000]
    }
    df_train = pd.DataFrame(training_data)
    X_train = df_train[['experience', 'test_score', 'interview_score']]
    y_actual = df_train['salary']
    y_pred = model.predict(X_train)

    fig, ax = plt.subplots()
    ax.plot(y_actual, label='Actual Salary', marker='o')
    ax.plot(y_pred, label='Predicted Salary', marker='x')

    # Add user point
    user_pred = prediction[0]
    ax.scatter(len(y_pred), user_pred, color='red', label='Your Prediction', zorder=5)

    ax.set_title('Actual vs Predicted Salary')
    ax.set_xlabel('Employee Index')
    ax.set_ylabel('Salary')
    ax.legend()
    st.pyplot(fig)

# CSV Upload
st.header("📁 Upload CSV for Bulk Salary Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with columns: experience, test_score, interview_score", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.isnull().values.any():
            st.warning("⚠️ Warning: Your file contains missing values. They will be dropped.")
            df = df.dropna()

        if not {'experience', 'test_score', 'interview_score'}.issubset(df.columns):
            st.error("❌ Error: CSV must contain 'experience', 'test_score', and 'interview_score' columns.")
        else:
            X = df[['experience', 'test_score', 'interview_score']]
            y_pred_csv = model.predict(X)
            df['predicted_salary'] = y_pred_csv
            st.success("✅ Prediction completed! Here's a preview:")
            st.dataframe(df)

            # Plot for CSV
            st.subheader("📊 Predicted Salaries from CSV")
            fig2, ax2 = plt.subplots()
            ax2.plot(df['predicted_salary'], marker='o', label='Predicted Salary (CSV)', color='purple')
            ax2.set_title('Predicted Salaries from CSV')
            ax2.set_xlabel('Employee Index')
            ax2.set_ylabel('Salary')
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
