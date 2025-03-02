import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Define "Credit Score" Proxy: Categorizing Credit Score
score_bins = [0, 5000, 10000, 15000, np.inf]
score_labels = ['Poor', 'Fair', 'Good', 'Excellent']
df['Credit_Score_Category'] = pd.cut(df['Credit_Limit'] * (1 - df['Avg_Utilization_Ratio']), bins=score_bins, labels=score_labels)
le_class = LabelEncoder()
df['Credit_Score_Category'] = le_class.fit_transform(df['Credit_Score_Category'])

# Define Features and Target for Classification
X = df.drop(columns=['Credit_Score_Category'])
y_class = df['Credit_Score_Category']

# Train-Test Split
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train_class)

# Function to Predict Credit Score of a New Customer
def predict_credit_score(new_customer):
    new_customer_df = pd.DataFrame([new_customer])
    
    # Ensure all required columns exist and reorder columns to match training data
    for col in label_encoders:
        if col in new_customer_df.columns:
            new_customer_df[col] = label_encoders[col].transform(new_customer_df[col])
    
    # Reindex the dataframe to ensure all necessary features are included
    new_customer_df = new_customer_df.reindex(columns=X.columns, fill_value=0)
    
    # Standardize features
    new_customer_scaled = scaler.transform(new_customer_df)
    
    # Predict credit score category
    predicted_score_category = rf_clf.predict(new_customer_scaled)[0]
    return le_class.inverse_transform([predicted_score_category])[0]

# Example Usage
customers_data = [
    {
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
        "Avg_Utilization_Ratio": 0.061,
        "Attrition_Flag": "Existing Customer"
    },
    {
        "Customer_Age": 34,
        "Gender": "F",
        "Dependent_count": 2,
        "Education_Level": "Doctorate",
        "Marital_Status": "Single",
        "Income_Category": "$80K - $120K",
        "Card_Category": "Gold",
        "Months_on_book": 50,
        "Total_Relationship_Count": 6,
        "Months_Inactive_12_mon": 1,
        "Contacts_Count_12_mon": 2,
        "Credit_Limit": 15000.0,
        "Total_Revolving_Bal": 500,
        "Avg_Open_To_Buy": 14500.0,
        "Total_Amt_Chng_Q4_Q1": 1.8,
        "Total_Trans_Amt": 2000,
        "Total_Trans_Ct": 50,
        "Total_Ct_Chng_Q4_Q1": 1.2,
        "Avg_Utilization_Ratio": 0.03,
        "Attrition_Flag": "Existing Customer"
    }
]
predicted_score_category = predict_credit_score(customers_data[1])
print("Predicted Credit Score Category:", predicted_score_category)
