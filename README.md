FinCheckAI – AI-Powered Loan Eligibility Predictor:

FinCheckAI is a machine learning-driven loan eligibility prediction system developed using Python, Flask, and XGBoost. It demonstrates how artificial intelligence and data science can be used in the financial domain to automate and optimize risk evaluation and loan approval processes.

Key Features:

End-to-end AI pipeline: Data preprocessing → Feature transformation → Prediction.

Built using a pre-trained XGBoost model optimized for binary classification.

Clean, responsive Flask web interface for real-time user input and result display.

Context-aware input guidance to avoid confusion (e.g., income scale, optional fields).

Display of predicted decision with contextual explanation (Eligible / Not Eligible).

How It Works:

User fills in loan-related inputs (income, dependents, credit history, etc.).

Input is preprocessed and mapped into the feature format required by the model.

The trained XGBoost model evaluates the input and returns a binary prediction.

The UI displays a user-friendly message explaining whether the applicant is eligible or not.

