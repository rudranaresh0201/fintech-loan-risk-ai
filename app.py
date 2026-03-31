import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# LOAD MODELS
# ---------------------------
clf = joblib.load("classification_model.pkl")      # Eligibility model
reg = joblib.load("regression_model.pkl")          # EMI amount model
scaler = joblib.load("scaler.pkl")                 # StandardScaler
feature_cols = joblib.load("feature_columns.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# Remove incorrectly included target columns
bad_cols = [
    "emi_eligibility_High_Risk",
    "emi_eligibility_Not_Eligible",
    "emi_eligibility_label"
]

feature_cols = [c for c in feature_cols if c not in bad_cols]

  # Input column list

st.title("EMI Predict AI – Financial Risk Assessment Platform")
st.write("Enter your financial details to predict EMI eligibility and maximum EMI limit.")

# ---------------------------
# USER INPUTS
# ---------------------------
monthly_salary = st.number_input("Monthly Salary", min_value=0, value=50000)
years_of_employment = st.number_input("Years of Employment", min_value=0, value=2)
monthly_rent = st.number_input("Monthly Rent", min_value=0, value=10000)
family_size = st.number_input("Family Size", min_value=1, value=3)
dependents = st.number_input("Dependents", min_value=0, value=1)
school_fees = st.number_input("School Fees", min_value=0, value=0)
college_fees = st.number_input("College Fees", min_value=0, value=0)
travel_expenses = st.number_input("Travel Expenses", min_value=0, value=3000)
groceries = st.number_input("Groceries & Utilities", min_value=0, value=5000)
other_expenses = st.number_input("Other Monthly Expenses", min_value=0, value=2000)
current_emi_amount = st.number_input("Current EMI Amount", min_value=0, value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750)
bank_balance = st.number_input("Bank Balance", min_value=0, value=20000)
emergency_fund = st.number_input("Emergency Fund", min_value=0, value=5000)
requested_amount = st.number_input("Requested Loan Amount", min_value=0, value=200000)
requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, value=24)

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education", ["High School", "Post Graduate", "Professional"])
employment_type = st.selectbox("Employment Type", ["Private", "Self-employed"])
company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Small", "Startup"])
house_type = st.selectbox("House Type", ["Own", "Rented"])
existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
emi_scenario = st.selectbox("EMI Scenario", ["Education EMI", "Home Appliances EMI", 
                                             "Personal Loan EMI", "Vehicle EMI"])

# ---------------------------
# BASE NUMERIC INPUT DF
# ---------------------------
df_input = pd.DataFrame([{
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries,
    "other_monthly_expenses": other_expenses,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure
}])

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
df_input["total_expenses"] = (
    df_input["monthly_rent"] +
    df_input["school_fees"] +
    df_input["college_fees"] +
    df_input["travel_expenses"] +
    df_input["groceries_utilities"] +
    df_input["other_monthly_expenses"]
)

df_input["disposable_income"] = df_input["monthly_salary"] - df_input["total_expenses"]
df_input["debt_to_income"] = df_input["current_emi_amount"] / df_input["monthly_salary"]
df_input["expense_to_income"] = df_input["total_expenses"] / df_input["monthly_salary"]
df_input["financial_buffer"] = df_input["bank_balance"] + df_input["emergency_fund"]

# ---------------------------
#
# ---------------------------
# DUMMY ENCODING (1 for selected option)
# ---------------------------
dummy_cols = {
    f"gender_{gender}": 1,
    f"marital_status_{marital_status}": 1,
    f"education_{education}": 1,
    f"employment_type_{employment_type}": 1,
    f"company_type_{company_type}": 1,
    f"house_type_{house_type}": 1,
    f"existing_loans_{existing_loans}": 1,
    f"emi_scenario_{emi_scenario}": 1
}

# Add dummy columns to input dataframe
for col in dummy_cols:
    df_input[col] = dummy_cols[col]

# Add missing columns that were present during training
for col in feature_cols:
    if col not in df_input.columns:
        df_input[col] = 0

# Ensure df_input matches EXACT training columns order
df_input = df_input.reindex(columns=feature_cols, fill_value=0)

# ---------------------------
# SCALE INPUT
# ---------------------------
st.write(" df_input columns:", list(df_input.columns))
st.write(" feature_cols:", feature_cols)
scaled_input = scaler.transform(df_input)


# ---------------------------
# PREDICT
# ---------------------------
class_pred = clf.predict(scaled_input)[0]
reg_pred = reg.predict(scaled_input)[0]

# ---------------------------
# DISPLAY OUTPUT
# ---------------------------
if st.button("Predict EMI Eligibility & EMI Amount"):
    st.subheader("📌 Prediction Results")

    if class_pred == 1:
        st.error("❌ Not Eligible for EMI")
    else:
        st.success("✅ Eligible for EMI")

    st.info(f"💸 Maximum EMI You Can Pay: **₹{reg_pred:.2f}**")
