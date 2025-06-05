import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier, StackingClassifier
import shap

# Set page config
st.set_page_config(
    page_title="ML Model Comparison App",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📈 Machine Learning Model Comparison")
st.markdown("""
This app allows you to explore the Iris and Breast Cancer datasets and compare different machine learning models.
""")

# Load datasets
@st.cache_data
def load_data():
    breast = load_breast_cancer()
    iris = load_iris()
    
    # Convert to DataFrames
    df_breast = pd.DataFrame(breast.data, columns=breast.feature_names)
    df_breast['target'] = breast.target
    
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    
    return df_breast, df_iris, breast.feature_names, iris.feature_names

df_breast, df_iris, breast_features, iris_features = load_data()

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
dataset = st.sidebar.selectbox(
    "Choose a dataset",
    ["Iris", "Breast Cancer"]
)

# Prepare data based on selection
if dataset == "Iris":
    df = df_iris
    feature_names = iris_features
    target_names = ["Setosa", "Versicolor", "Virginica"]
    dataset_description = """
    The Iris dataset is a classic dataset for classification. It contains 150 samples of iris flowers, 
    each with 4 features (sepal length, sepal width, petal length, petal width) and 3 classes 
    (Setosa, Versicolor, Virginica).
    """
else:
    df = df_breast
    feature_names = breast_features
    target_names = ["Malignant", "Benign"]
    dataset_description = """
    The Breast Cancer dataset contains features computed from a digitized image of a fine needle aspirate (FNA) 
    of a breast mass. The task is to predict whether the mass is malignant or benign.
    """

# Dataset Overview Section
st.header("🔎 Dataset Overview")
st.markdown(dataset_description)

# Display basic dataset information
st.subheader("Dataset Information")
col1, col2 = st.columns(2)

with col1:
    st.write("Number of samples:", df.shape[0])
    st.write("Number of features:", df.shape[1] - 1)
    st.write("Number of classes:", len(target_names))
    st.write("Class distribution:")
    class_dist = df['target'].value_counts()
    for i, count in class_dist.items():
        st.write(f"- {target_names[i]}: {count} samples")

with col2:
    st.write("Feature names:")
    for feature in feature_names:
        st.write(f"- {feature}")

# Data Preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Statistical Summary
st.subheader("Statistical Summary")
st.dataframe(df.describe())

# Visualizations
st.header("📊 Data Visualizations")

# Correlation Matrix
st.subheader("Feature Correlation Matrix")
fig, ax = plt.subplots(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
plt.title(f"{dataset} Dataset Correlation Matrix")
st.pyplot(fig)

# Feature Distributions
st.subheader("Feature Distributions by Class")
selected_feature = st.selectbox("Select a feature to visualize:", feature_names)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='target', y=selected_feature, data=df)
plt.xticks(range(len(target_names)), target_names)
plt.title(f"{selected_feature} Distribution by Class")
st.pyplot(fig)

# Pair Plot for Iris dataset
if dataset == "Iris":
    st.subheader("Pair Plot")
    fig = sns.pairplot(df, hue='target', diag_kind='kde')
    plt.suptitle("Iris Dataset Pair Plot", y=1.02)
    st.pyplot(fig)

# Model Comparison Section
st.header("🤖 Model Comparison")

# Model selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "Gaussian Naive Bayes", "SVM Linear", "SVM RBF", "Voting Classifier", "Stacking Classifier"]
)

# Split data
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(C=0.1, penalty='l2', solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=5),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM Linear": SVC(kernel='linear', C=0.1, gamma="scale", probability=True),
    "SVM RBF": SVC(kernel='rbf', C=0.1, gamma="scale", probability=True)
}

# Train selected model
if model_name in ["Voting Classifier", "Stacking Classifier"]:
    if model_name == "Voting Classifier":
        model = VotingClassifier(
            estimators=[(name, clf) for name, clf in models.items()],
            voting='hard'
        )
    else:
        model = StackingClassifier(
            estimators=[(name, clf) for name, clf in models.items()],
            final_estimator=LogisticRegression()
        )
else:
    model = models[model_name]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
st.subheader("Model Performance")

# Create two columns for metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Accuracy Score")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.3f}")

with col2:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', ax=ax)
st.pyplot(fig)

# Feature Importance (if applicable)
if hasattr(model, 'coef_'):
    st.subheader("Feature Importance")
    if model_name == "Logistic Regression":
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_[0])
        })
        importance = importance.sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance.head(10), x='Importance', y='Feature')
        plt.title("Top 10 Most Important Features")
        st.pyplot(fig)

# Prediction Interface
st.header("🎯 Make Predictions")
st.markdown("Enter values for each feature to make a prediction:")

# Create input fields for features
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            f"{feature}",
            value=float(df[feature].mean()),
            format="%.2f"
        )

if st.button("Predict"):
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"Predicted Class: {target_names[prediction[0]]}")
    
    # Display probabilities
    prob_df = pd.DataFrame({
        'Class': target_names,
        'Probability': probability[0]
    })
    st.write("Class Probabilities:")
    st.dataframe(prob_df)

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Machine Learning Model Comparison App") 