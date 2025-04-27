import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model and scaler

with open('models/decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load the dataset
df = pd.read_csv('C:\\Users\\M SOWMITHA\\Desktop\\DSA_CAPSTONE\\data\\Intrusion_detection.csv')

# Preprocessing
df.fillna(method='ffill', inplace=True)

# Label Encoding for categorical columns (if any)
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoders[column].transform(df[column])

# Features and Target
y = df['Label']
X = df.drop(['Label'], axis=1)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scaling
X_scaled = scaler.transform(X_resampled)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Prediction
y_pred = model.predict(X_test)

# Streamlit layout
st.title("Decision Tree Model Analysis and Visualization")

# Sidebar for selecting which plot to display
st.sidebar.header("Choose a Section to View")
plot_choice = st.sidebar.radio("Select a plot type", ("Data Exploration", "Model Evaluation"))

# Data Exploration Section
if plot_choice == "Data Exploration":
    st.subheader("Explore the Distribution of a Feature (Histogram)")
    selected_column = st.selectbox("Select a Column to Display Histogram", df.columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[selected_column], kde=True, ax=ax)  # Add kernel density estimate (KDE)
    st.pyplot(fig)

    st.subheader("Visualize Relationships Between Two Features (Scatter Plot)")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pairwise scatterplots (you can select two features to plot)
    selected_columns = st.multiselect("Select Two Features to Compare", df.columns.tolist(), default=df.columns[:2].tolist())
    
    if len(selected_columns) == 2:
        sns.scatterplot(x=df[selected_columns[0]], y=df[selected_columns[1]], ax=ax)
        st.pyplot(fig)

    st.subheader("Statistical Summary of a Feature (Box Plot)")
    selected_column = st.selectbox("Select Column for Box Plot", df.columns)
    fig, ax = plt.subplots(figsize=(6, 4))  # Create a figure and axis
    sns.boxplot(x=df[selected_column], ax=ax)
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# Model Evaluation Section
if plot_choice == "Model Evaluation":
    # Feature Importance
    st.subheader("Top 10 Most Important Features According to Decision Tree")
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis
    feat_importances.nlargest(10).plot(kind='barh', ax=ax)
    st.pyplot(fig)  # Pass the figure to st.pyplot()

    # Performance Metrics: Precision, Recall, F1-Score
    st.subheader("Model Performance Metrics: Precision, Recall, F1-Score")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### Precision, Recall, and F1-Score:")
    metrics_df = pd.DataFrame(report).transpose()
    st.write(metrics_df)

    # Statistical Measures of the Dataset
    st.subheader("Dataset Statistical Overview")
    st.write("### Summary Statistics of the Dataset:")
    st.write(df.describe())
