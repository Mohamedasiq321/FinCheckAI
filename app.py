from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load("loan_default_xgboost_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get values from form
        gender = request.form["gender"]
        married = request.form["married"]
        dependents = request.form["dependents"]
        education = request.form["education"]
        self_employed = request.form["self_employed"]
        property_area = request.form["property_area"]
        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        credit_history = float(request.form["credit_history"])
        sanity_threshold = float(request.form["sanity_threshold"])

        total_income = applicant_income + coapplicant_income
        income_to_loan_ratio = loan_amount / (total_income + 1)

        input_data = {
            'cat__Gender_Female': 1 if gender == 'Female' else 0,
            'cat__Gender_Male': 1 if gender == 'Male' else 0,
            'cat__Gender_nan': 0,
            'cat__Married_No': 1 if married == 'No' else 0,
            'cat__Married_Yes': 1 if married == 'Yes' else 0,
            'cat__Married_nan': 0,
            'cat__Dependents_0': 1 if dependents == '0' else 0,
            'cat__Dependents_1': 1 if dependents == '1' else 0,
            'cat__Dependents_2': 1 if dependents == '2' else 0,
            'cat__Dependents_3+': 1 if dependents == '3+' else 0,
            'cat__Dependents_nan': 0,
            'cat__Education_Graduate': 1 if education == 'Graduate' else 0,
            'cat__Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
            'cat__Self_Employed_No': 1 if self_employed == 'No' else 0,
            'cat__Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
            'cat__Self_Employed_nan': 0,
            'cat__Property_Area_Rural': 1 if property_area == 'Rural' else 0,
            'cat__Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
            'cat__Property_Area_Urban': 1 if property_area == 'Urban' else 0,
            'remainder__Loan_ID': 123456,
            'remainder__ApplicantIncome': applicant_income,
            'remainder__CoapplicantIncome': coapplicant_income,
            'remainder__LoanAmount': loan_amount,
            'remainder__Loan_Amount_Term': loan_term,
            'remainder__Credit_History': credit_history
        }

        df_input = pd.DataFrame([input_data])

        warnings = []
        suggestions = []

        try:
            emi = loan_amount / loan_term
            if total_income < 3 * emi:
                warnings.append(f"⚠ Income ₹{total_income} is less than 3× EMI ₹{round(emi, 2)}")
        except:
            warnings.append("⚠ EMI calculation error (check loan amount and term)")

        if credit_history == 1.0:
            max_allowed = 0.85 * total_income
        else:
            max_allowed = 0.75 * total_income

        if loan_amount > max_allowed:
            warnings.append(f"⚠ Loan ₹{loan_amount} exceeds safe limit ₹{int(max_allowed)}")

        sanity_flag = income_to_loan_ratio > sanity_threshold

        prediction = model.predict(df_input)[0]
        status = "Approved" if prediction == 1 else "Rejected"

        if prediction == 0:
            # Try reducing loan amount
            for reduced_amount in range(int(loan_amount), 0, -5):
                input_data['remainder__LoanAmount'] = reduced_amount
                df_try = pd.DataFrame([input_data])
                if model.predict(df_try)[0] == 1:
                    suggestions.append(f"📉 Try reducing loan amount to ₹{reduced_amount} (Approved)")
                    break

            # Try increasing loan term
            for extended_term in range(int(loan_term + 12), int(loan_term + 121), 12):
                input_data['remainder__LoanAmount'] = loan_amount
                input_data['remainder__Loan_Amount_Term'] = extended_term
                df_try = pd.DataFrame([input_data])
                if model.predict(df_try)[0] == 1:
                    suggestions.append(f"⏳ Try increasing loan term to {extended_term} months (Approved)")
                    break

            # Try increasing co-applicant income
            for extra_income in range(1000, 10001, 500):
                input_data['remainder__CoapplicantIncome'] = coapplicant_income + extra_income
                df_try = pd.DataFrame([input_data])
                if model.predict(df_try)[0] == 1:
                    suggestions.append(f"👥 Try increasing co-applicant income by ₹{extra_income} (Approved)")
                    break

            if not suggestions:
                suggestions.append("⚠ Could not find simple suggestions to auto-approve. Consider adjusting multiple factors.")

        return render_template("result.html", status=status, sanity_flag=sanity_flag,
                               ratio=income_to_loan_ratio, suggestions=suggestions,
                               warnings=warnings)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)

