🏦 Banking Fraud Detection System

A real-time transaction fraud detection web app built with Streamlit and scikit-learn, which predicts whether a banking transaction is likely fraudulent or legitimate. The app uses pre-trained models with engineered features to assess risk levels and display an intuitive probability bar.

Features

✅ Real-time fraud prediction for individual transactions.

📊 Shows fraud probability, risk level, and verdict.

🟢🟡🔴 Color-coded LOW / MEDIUM / HIGH risk.

🔬 Displays engineered features for debugging and inspection.

🖥️ Easy-to-use Streamlit UI with responsive layout.

💾 Supports saved models with optional scaler.

Screenshots

(Add your screenshots here for home page, input form, and results page)

Installation

Clone the repository:

git clone https://github.com/yourusername/banking-fraud-app.git
cd banking-fraud-app

Create a virtual environment (recommended):

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

Install dependencies:

pip install -r requirements.txt

Place your trained model in the project folder:

 fraud_model.pkl (bare model)

 for use 
streamlit app, use the following command in your project directory (where app.py is located):

streamlit run app.py

Usage

Run the Streamlit app:

streamlit run app.py

Enter transaction details in the form:

Step (time unit)

Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)

Amount

Origin old & new balances

Destination old & new balances

Click "Analyse Transaction" to see:

Fraud probability (0–100%)

Risk level (LOW / MEDIUM / HIGH)

Verdict (Fraudulent / Legitimate)

Probability bar visualization

Engineered features used for prediction

Test Cases (LOW / MEDIUM / HIGH)
Step	Tx Type	Amount	Origin Old	Origin New	Dest Old	Dest New	Expected Risk
10	PAYMENT	1000	5000	4000	2000	2500	LOW
50	TRANSFER	50,000	60,000	10,000	10,000	60,000	MEDIUM
100	CASH_OUT	250,000	300,000	50,000	5,000	255,000	HIGH
Project Structure
banking-fraud-app/
│
├─ app.py                  # Streamlit frontend & prediction logic
├─ best_fraud_model_tuned.pkl  # Pretrained model + features + scaler
├─ requirements.txt        # Python dependencies
├─ README.md               # This file
Requirements

Python 3.8+

Streamlit

pandas

numpy

scikit-learn

joblib

Install all dependencies via:

pip install -r requirements.txt
How It Works

Input processing: Transaction details are converted into engineered features:

hour, is_night, balance_diff_orig, balance_diff_dest, type_enc

Optional: log_amount, is_high_amount (based on training features)

Scaling: Features are scaled using the saved RobustScaler (if present).

Prediction: Model outputs a fraud probability between 0–1.

Thresholding: If probability >= threshold → flagged as fraudulent.

Output: Probability bar, risk level