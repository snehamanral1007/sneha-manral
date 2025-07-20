import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("enhanced_salary_data.csv")

data = load_data()

# Separate features and label
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Define categorical and numerical features
categorical_features = ["EducationLevel", "JobRole", "CompanyType", "Location"]
numerical_features = ["YearsExperience"]

# Create the preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"  # Keep numerical columns as-is
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train the model
model.fit(X, y)

# --- Streamlit UI ---
st.title("💼 Advanced Salary Prediction App")
st.write("Enter your details to predict your estimated salary.")

# User inputs
years_exp = st.slider("Years of Experience", 0.0, 15.0, 3.0, 0.1)
education = st.selectbox("Education Level", sorted(data["EducationLevel"].unique()))
job_role = st.selectbox("Job Role", sorted(data["JobRole"].unique()))
company = st.selectbox("Company Type", sorted(data["CompanyType"].unique()))
location = st.selectbox("Location", sorted(data["Location"].unique()))

# Predict
input_data = pd.DataFrame({
    "YearsExperience": [years_exp],
    "EducationLevel": [education],
    "JobRole": [job_role],
    "CompanyType": [company],
    "Location": [location]
})

predicted_salary = model.predict(input_data)[0]

# Show result
st.subheader("Predicted Salary")
st.success(f"₹ {predicted_salary:,.2f}")

# Optional: Show dataset
with st.expander("📊 Show Training Data"):
    st.dataframe(data)
