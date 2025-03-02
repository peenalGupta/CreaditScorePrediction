# Re-import necessary libraries due to execution state reset
import pandas as pd
import numpy as np
import random

# Re-define the prediction function since execution state was reset
def predict_credit_score(new_customer):
    """
    Mock function to predict credit score category.
    Actual implementation should load a trained model and apply preprocessing before prediction.
    """
    credit_score_value = new_customer["Credit_Limit"] * (1 - new_customer["Avg_Utilization_Ratio"])
    
    if credit_score_value < 5000:
        return "Poor"
    elif credit_score_value < 10000:
        return "Fair"
    elif credit_score_value < 15000:
        return "Good"
    else:
        return "Excellent"

# Generate sample customers with varying credit scores
sample_customers = [
    {
        "Gender": random.choice(["M", "F"]),
        "Dependent_count": random.randint(0, 5),
        "Education_Level": random.choice(["High School", "Graduate", "Uneducated", "Doctorate"]),
        "Income_Category": random.choice(["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]),
        "Card_Category": random.choice(["Blue", "Silver", "Gold", "Platinum"]),
        "Months_on_book": random.randint(6, 60),
        "Total_Relationship_Count": random.randint(1, 6),
        "Months_Inactive_12_mon": random.randint(0, 6),
        "Contacts_Count_12_mon": random.randint(0, 6),
        "Credit_Limit": random.uniform(1000, 20000),  # Credit limit variation
        "Total_Revolving_Bal": random.randint(0, 5000),
        "Avg_Open_To_Buy": random.uniform(0, 20000),
        "Total_Amt_Chng_Q4_Q1": random.uniform(0.5, 2.5),
        "Total_Trans_Amt": random.randint(500, 20000),
        "Total_Trans_Ct": random.randint(5, 200),
        "Total_Ct_Chng_Q4_Q1": random.uniform(0.5, 2.5),
        "Avg_Utilization_Ratio": random.uniform(0, 1)  # Credit utilization ratio
    }
    for _ in range(10)  # Generate 10 different customers
]

# Predict credit score categories for these customers
predictions = [predict_credit_score(customer) for customer in sample_customers]

# Create a DataFrame for visualization
customer_df = pd.DataFrame(sample_customers)
customer_df["Predicted_Credit_Score"] = predictions

#save dataframe into a csv file
customer_df.to_csv('test_customer_data.csv', index=False)
print(customer_df)


