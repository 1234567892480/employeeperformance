
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).xls')
    df_processed = df.drop(['EmpNumber','Attrition'], axis=1)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    X = df_processed.drop('PerformanceRating', axis=1)
    y = df_processed['PerformanceRating']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return df, df_processed, X, y_encoded, label_encoder

# Call the load_data function to get the data
df, df_processed, X, y_encoded, label_encoder = load_data()

# Streamlit app layout
st.title("Employee Performance Analysis")

# Displaying the data
st.write("## Raw Data")
st.write(df)

# Visualization: Distribution of Performance Rating by Department
st.write("## Distribution of Performance Rating by Department")
sns.barplot(x="EmpDepartment", y="PerformanceRating", hue="EmpDepartment", data=df)
plt.xticks(rotation=90)
st.pyplot(plt.gcf())

st.write("## Distribution of Performance Rating by martial status")
sns.lineplot(x="MaritalStatus", y="PerformanceRating", hue="MaritalStatus", data=df)
plt.xticks(rotation=90)
st.pyplot(plt.gcf())

st.write("## Distribution of Performance Rating by EmpJobRole")
sns.barplot(x="EmpJobRole", y="PerformanceRating", hue="EmpJobRole", data=df)
plt.xticks(rotation=90)
st.pyplot(plt.gcf())
# Splitting data for the model
st.write("## Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Random Forest Classifier
st.write("## Random Forest Classifier")
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

# Model Evaluation
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Accuracy Score")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Optionally, add a grid search for hyperparameter tuning
if st.checkbox("Perform Grid Search for Hyperparameter Tuning"):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    st.write(f"Best parameters: {best_params}")
    
    # Re-train with best parameters
    rfc_best = grid_search.best_estimator_
    y_pred_best = rfc_best.predict(X_test)
    st.write("### Classification Report with Best Parameters")
    st.text(classification_report(y_test, y_pred_best))
    st.write(f"Accuracy with Best Parameters: {accuracy_score(y_test, y_pred_best):.2f}")
