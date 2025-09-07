import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Fixing class imbalance
from fairlearn.metrics import selection_rate
from aif360.sklearn.metrics import disparate_impact_ratio, statistical_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference

# Load dataset
df = pd.read_csv("BankChurners.csv")

# Drop unnecessary columns (Client ID and Naive Bayes columns)
df.drop(columns=['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], inplace=True)

# Encode categorical features
label_encoders = {}
for col in ['Attrition_Flag', 'Gender', 'Education_Level', 'Income_Category', 'Card_Category', 'Marital_Status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define "Credit Score" Proxy: Categorizing Credit Score
score_bins = [0, 5000, 10000, 15000, np.inf]
score_labels = ['Poor', 'Fair', 'Good', 'Excellent']
df['Credit_Score_Category'] = pd.cut(df['Credit_Limit'] * (1 - df['Avg_Utilization_Ratio']), bins=score_bins, labels=score_labels)
le_class = LabelEncoder()
df['Credit_Score_Category'] = le_class.fit_transform(df['Credit_Score_Category'])
print(df['Credit_Score_Category'].value_counts())

# Define Features and Target for Classification
X = df.drop(columns=['Credit_Score_Category'])
y_class = df['Credit_Score_Category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Handling Imbalanced Data Using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train_resampled)

# Convert X_test_scaled (NumPy array) back to DataFrame with original column names
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# Predict using the RandomForestClassifier
y_pred = rf_clf.predict(X_test_scaled_df)

# Print Model Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# # Bias Detection Metrics
# protected_attribute_gender = df.loc[X_test.index, 'Gender']  # Fetch original values before encoding
# protected_attribute_gender = protected_attribute_gender.map({0: "Male", 1: "Female"})  # Ensure correct mapping

# # Ensure indexing is correct
# protected_attribute_gender = protected_attribute_gender.loc[X_test.index]

# # Compute Selection Rates for Male & Female
# male_selection_rate = selection_rate(y_test[protected_attribute_gender == "Male"], y_pred[protected_attribute_gender == "Male"])
# female_selection_rate = selection_rate(y_test[protected_attribute_gender == "Female"], y_pred[protected_attribute_gender == "Female"])

# # Handle Zero Selection Rates
# if male_selection_rate == 0 or female_selection_rate == 0:
#     gender_disparate_impact = np.nan  # Avoid division by zero
#     print("⚠️ Warning: One of the gender groups has zero selection rate.")
# else:
#     gender_disparate_impact = female_selection_rate / male_selection_rate

# # Display Bias Metrics
# print("\n🔍 Fairness Metrics:")
# print("Male Selection Rate:", male_selection_rate)
# print("Female Selection Rate:", female_selection_rate)
# print("Gender Disparate Impact Ratio:", gender_disparate_impact)

# # SHAP Explainability
# explainer = shap.TreeExplainer(rf_clf)
# shap_values = explainer.shap_values(X_test)

# # SHAP Summary Plot (Fix Applied)
# # shap.summary_plot(shap_values, X_test_scaled_df, feature_names=X.columns)

# # SHAP Force Plot (for an individual prediction)
# shap.initjs()
# sample_index = 5  # Change this to check different predictions
# # shap.plots.force(explainer.expected_value[1], shap_values[1][sample_index], X_test_scaled_df.iloc[sample_index])

# shap_importance = np.abs(shap_values).mean(axis=0).flatten()

# # Ensure shap_importance has the same length as X.columns
# if len(shap_importance) == len(X.columns):
#     # Create a DataFrame for feature importance
#     shap_importance_df = pd.DataFrame({
#         "Feature": X.columns,
#         "SHAP Importance": shap_importance
#     }).sort_values(by="SHAP Importance", ascending=False)

#     print("\n🔑 Feature Importance (SHAP Values):" )
#     print(shap_importance_df)
# else:
#     print("Error: Length of SHAP importance values does not match number of features.")

# Implement fairness evaluation
# Compute Statistical Parity Difference for
# Marital Status (0: Married, 1: Single)
# protected_attribute_marital_status = df.loc[X_test.index, 'Marital_Status']  # Fetch original values before encoding
# protected_attribute_marital_status = protected_attribute_marital_status.map({0: "Married", 1: "Single"})  # Ensure correct mapping

# # Compute Statistical Parity Difference
# stat_parity_diff = statistical_parity_difference(y_test, y_pred, prot_attr=protected_attribute_marital_status)
# print("\nStatistical Parity Difference (Marital Status):", stat_parity_diff)

# # Compute Disparate Impact Ratio
# disp_impact_ratio = disparate_impact_ratio(y_test, y_pred, prot_attr=protected_attribute_marital_status)
# print("Disparate Impact Ratio (Marital Status):", disp_impact_ratio)

# # Implement Bias mitigation using Fairlearn
# # Define Fairlearn ThresholdOptimizer
# thresh_opt = ThresholdOptimizer(estimator=rf_clf, constraints="demographic_parity")

# # Train Fairlearn ThresholdOptimizer
# thresh_opt.fit(X_train_scaled, y_train, prot_attr=protected_attribute_gender)

# # Predict using Fairlearn ThresholdOptimizer
# y_pred_fair = thresh_opt.predict(X_test_scaled)

# Prediction Function for New Customer
def predict_credit_score(new_customer):
    new_customer_df = pd.DataFrame([new_customer])

    # Ensure all columns match trained features
    new_customer_df = new_customer_df.reindex(columns=X.columns, fill_value=0)

    # Transform categorical variables
    for col in ['Attrition_Flag', 'Gender', 'Education_Level', 'Income_Category', 'Card_Category', 'Marital_Status']:
        new_customer_df[col] = label_encoders[col].transform(new_customer_df[col])

    # Standardize Data
    new_customer_scaled = scaler.transform(new_customer_df)

    # Predict
    prediction = rf_clf.predict(new_customer_scaled)
    return le_class.inverse_transform([prediction[0]])[0]  # Convert numerical class back to category

# New Customer Data
# Read csv file into a pandas dataframe
new_customer_data = pd.read_csv("test_customer_data.csv")

# Predict Credit Score
predicted_scores = new_customer_data.apply(predict_credit_score, axis=1)
print("Predicted Credit Score Category:", predicted_scores)