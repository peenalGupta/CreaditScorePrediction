import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv("BankChurners.csv")

# Drop unnecessary columns (Client ID and Naive Bayes columns)
df.drop(columns=['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], inplace=True)

# Encode categorical features
label_encoders = {}
for col in ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define "Credit Score" Proxy: Credit Score = f(Credit Utilization, Credit Limit, Transaction Amount, Account Age)
df['Credit_Score_Proxy'] = df['Credit_Limit'] * (1 - df['Avg_Utilization_Ratio'])

# Define Features and Target for Regression
X = df.drop(columns=['Credit_Score_Proxy'])
y_reg = df['Credit_Score_Proxy']

# Define Classification Target: Categorizing Credit Score
score_bins = [0, 5000, 10000, 15000, np.inf]
score_labels = ['Poor', 'Fair', 'Good', 'Excellent']
df['Credit_Score_Category'] = pd.cut(df['Credit_Score_Proxy'], bins=score_bins, labels=score_labels)
le_class = LabelEncoder()
df['Credit_Score_Category'] = le_class.fit_transform(df['Credit_Score_Category'])
y_class = df['Credit_Score_Category']

# Train-Test Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Regression Models ---

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_reg)
y_pred_lin = lin_reg.predict(X_test_scaled)
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_lin)))

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train_reg)
y_pred_rf = rf_reg.predict(X_test)
print("Random Forest Regressor RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_rf)))

# XGBoost Regressor
xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(X_train, y_train_reg)
y_pred_xgb = xgb_reg.predict(X_test)
print("XGBoost Regressor RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb)))

# --- Classification Models ---

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train_class)
y_pred_log = log_reg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test_class, y_pred_log))

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train_class)
y_pred_rf_clf = rf_clf.predict(X_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test_class, y_pred_rf_clf))

# XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X_train, y_train_class)
y_pred_xgb_clf = xgb_clf.predict(X_test)
print("XGBoost Classifier Accuracy:", accuracy_score(y_test_class, y_pred_xgb_clf))

# Neural Network Classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_clf.fit(X_train_scaled, y_train_class)
y_pred_mlp = mlp_clf.predict(X_test_scaled)
print("MLP Neural Network Accuracy:", accuracy_score(y_test_class, y_pred_mlp))


# Function to Predict Credit Score of a New Customer
def predict_credit_score(new_customer):
    new_customer_df = pd.DataFrame([new_customer])
    for col in label_encoders:
        if col in new_customer_df.columns:
            new_customer_df[col] = label_encoders[col].transform(new_customer_df[col])
    new_customer_scaled = scaler.transform(new_customer_df)
    predicted_score = rf_reg.predict(new_customer_scaled)
    return predicted_score[0]


# Example Usage
new_customer_data = {
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 3,
    "Education_Level": "Graduate",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 39,
    "Total_Relationship_Count": 5,
    "Months_Inactive_12_mon": 2,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 12691.0,
    "Total_Revolving_Bal": 777,
    "Avg_Open_To_Buy": 11914.0,
    "Total_Amt_Chng_Q4_Q1": 1.335,
    "Total_Trans_Amt": 1144,
    "Total_Trans_Ct": 42,
    "Total_Ct_Chng_Q4_Q1": 1.625,
    "Avg_Utilization_Ratio": 0.061
}

predicted_score = predict_credit_score(new_customer_data)
print("Predicted Credit Score:", predicted_score)
