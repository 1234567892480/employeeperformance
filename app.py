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
    return X, y_encoded, label_encoder

# Your existing code for plotting
plt.figure(figsize=(18, 12))
categorical_features = ['Gender', 'EmpDepartment', 'EmpJobRole', 'MaritalStatus', 'BusinessTravelFrequency', 'Attrition']
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=df[feature])
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Performance Ratings Across Different Departments
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="EmpDepartment", y="PerformanceRating", hue="EmpDepartment")
plt.title("Departmental vs Performance Rating")
plt.xlabel("EmpDepartment")
plt.ylabel("PerformanceRating")
plt.show()

# New plot for Performance Rating and Age
sns.relplot(
    data=df, kind="line",
    x="Age", y="PerformanceRating",
    # hue="region",  # Uncomment if you have a 'region' column and want to use it for grouping
    # units="subject", estimator=None,  # Adjust these parameters as needed
)
plt.title("Performance Rating and Age")
plt.xlabel("Age")
plt.ylabel("Performance Rating")
plt.show()

# Set up the figure for multiple plots
plt.figure(figsize=(15, 10))

# Pie chart for Marital Status analysis
percent_1 = list(df['MaritalStatus'].value_counts())
plt.subplot(2, 2, 1)
plt.pie(percent_1, labels=["Married", "Single", "Divorced"],
        explode=[0, 0, 0], autopct="%0.2f%%", startangle=46,
        pctdistance=0.65, textprops={"fontsize": 15, "fontweight": "bold", 'color': "k"})
plt.title("Marital Status Analysis\n", fontsize=20, fontweight='bold')

# Countplot for Gender vs Marital Status
plt.subplot(2, 2, 2)
sns.countplot(x='Gender', hue='MaritalStatus', data=df, palette="YlOrRd")
plt.title("Marital Status with Gender Analysis\n", fontweight="bold", fontsize=20)
plt.xlabel("\nGender")
plt.ylabel("Count\n")
legend = plt.legend(prop={"size": 13})
legend.set_title("Marital Status\n", prop={"size": 15, "weight": "bold"})
plt.setp(legend.get_texts(), color='black')
legend.draw_frame(False)

# Countplot for Marital status vs Travel Frequency
plt.subplot(2, 2, 3)
sns.countplot(x='BusinessTravelFrequency', hue='MaritalStatus', data=df, palette="mako")
plt.title("Business Travel Frequency by Marital Status", fontweight="bold", fontsize=20)
plt.xlabel("Travel Frequency")
plt.ylabel("Count")
legend = plt.legend(prop={"size": 13})
legend.set_title("Marital Status", prop={"size": 15, "weight": "bold"})
plt.setp(legend.get_texts(), color='black')
legend.draw_frame(False)

# Countplot for Marital Status vs Overtime
plt.subplot(2, 2, 4)
sns.countplot(x='OverTime', hue='MaritalStatus', data=df, palette="husl")
plt.title("Overtime by Marital Status", fontweight="bold", fontsize=20)
plt.xlabel("Over Time")
plt.ylabel("Count")
legend = plt.legend(prop={"size": 13})
legend.set_title("Marital Status", prop={"size": 15, "weight": "bold"})
plt.setp(legend.get_texts(), color='black')
legend.draw_frame(False)

plt.tight_layout(pad=2)
plt.show()


# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return best_model, accuracy, report

# Streamlit app
st.title("Employee Performance Prediction")

st.write("""
This app uses a RandomForestClassifier model to predict employee performance based on historical data.
You can use it to assess the potential performance of new employees.
""")

X, y, label_encoder = load_data()
model, accuracy, report = train_model(X, y)

st.write(f"Model Accuracy: {accuracy}")
st.text("Classification Report:")
st.text(report)

st.subheader("Predict Employee Performance")

# Collect user input for new employee data
input_data = {}
for col in X.columns:
    if col in X.select_dtypes(include=['object']).columns:  # Handle categorical inputs
        input_data[col] = st.selectbox(f"{col}", options=label_encoder.classes_)
        input_data[col] = label_encoder.transform([input_data[col]])[0]
    else:  # Handle numerical inputs
        input_data[col] = st.number_input(f"{col}", value=0, step=1)

# Predict performance
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"Predicted Performance Rating: {prediction[0]}")
