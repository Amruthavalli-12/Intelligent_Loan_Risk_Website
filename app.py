import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Loan Risk Assessment",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL + PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model():
    preprocessor = joblib.load("preprocessor.pkl")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgb_model.json")

    return preprocessor, xgb_model


try:
    preprocessor, xgb_model = load_model()

except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Intelligent Loan Risk Assessment System")

st.write(
    "Enter the applicant's loan and financial information "
    "to assess the loan approval risk."
)

st.divider()


# ============================================================
# LOAN APPLICATION DETAILS
# ============================================================

st.header("📋 Loan Application Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )


with col2:
    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0
    )


with col3:
    loan_amount_term = st.number_input(
        "Loan Amount Term",
        min_value=1.0,
        value=360.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


# ============================================================
# FINANCIAL DETAILS
# ============================================================

st.header("💰 Financial Details")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=0.0,
        max_value=1000.0,
        value=700.0
    )

    previous_loans = st.number_input(
        "Previous Loans",
        min_value=0.0,
        value=1.0
    )

    previous_defaults = st.number_input(
        "Previous Defaults",
        min_value=0.0,
        value=0.0
    )


with col2:
    total_debt = st.number_input(
        "Total Debt",
        min_value=0.0,
        value=100000.0
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=600000.0
    )

    monthly_expenses = st.number_input(
        "Monthly Expenses",
        min_value=0.0,
        value=20000.0
    )


with col3:
    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=100000.0
    )

    payment_history = st.number_input(
        "Payment History",
        min_value=0.0,
        value=90.0
    )

    credit_utilization = st.number_input(
        "Credit Utilization",
        min_value=0.0,
        value=30.0
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Assess Loan Risk",
    use_container_width=True
):

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # Same features used during model training
    # --------------------------------------------------------

    total_income = (
        applicant_income +
        coapplicant_income
    )

    dti = (
        total_debt /
        (annual_income + 1)
    )

    loan_income_ratio = (
        loan_amount /
        (total_income + 1)
    )

    disposable_income = (
        annual_income -
        monthly_expenses * 12
    )

    debt_per_previous_loan = (
        total_debt /
        (previous_loans + 1)
    )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],

        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area],

        "Credit_Score": [credit_score],
        "Previous_Loans": [previous_loans],
        "Previous_Defaults": [previous_defaults],
        "Total_Debt": [total_debt],
        "Annual_Income": [annual_income],
        "Monthly_Expenses": [monthly_expenses],
        "Savings": [savings],
        "Payment_History": [payment_history],
        "Credit_Utilization": [credit_utilization],

        "TotalIncome": [total_income],
        "DTI": [dti],
        "LoanIncomeRatio": [loan_income_ratio],
        "DisposableIncome": [disposable_income],
        "DebtPerPreviousLoan": [debt_per_previous_loan]
    })


    # --------------------------------------------------------
    # PREPROCESS INPUT
    # --------------------------------------------------------

    try:
        processed_input = preprocessor.transform(input_data)

        # Convert sparse matrix if required
        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = xgb_model.predict(processed_input)[0]

        probability = xgb_model.predict_proba(
            processed_input
        )[0][1]


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("📊 Loan Risk Assessment Result")

        if prediction == 1:

            st.success(
                "✅ Loan Application: APPROVED"
            )

        else:

            st.error(
                "❌ Loan Application: REJECTED"
            )


        approval_percentage = probability * 100

        st.metric(
            "Approval Probability",
            f"{approval_percentage:.2f}%"
        )

        st.progress(
            float(probability)
        )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader("Applicant Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.write(
                f"**Credit Score:** {credit_score}"
            )
            st.write(
                f"**Annual Income:** ₹{annual_income:,.2f}"
            )

        with summary_col2:
            st.write(
                f"**Loan Amount:** ₹{loan_amount:,.2f}"
            )
            st.write(
                f"**Total Debt:** ₹{total_debt:,.2f}"
            )

        with summary_col3:
            st.write(
                f"**Previous Loans:** {previous_loans}"
            )
            st.write(
                f"**Previous Defaults:** {previous_defaults}"
            )


    except Exception as e:

        st.error(
            "An error occurred while processing the application."
        )

        st.exception(e)