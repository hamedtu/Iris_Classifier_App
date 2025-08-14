"""
Created on 14 Aug 2025 

@author: Hamed Koochaki
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Machine Learning Iris Classification",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("Iris Classification App")
# st.markdown("""
# This app demonstrates XGBoost classification on the famous Iris dataset. 
# Adjust the hyperparameters and see how they affect the model's performance!
# """)
image = Image.open('irises.jpeg')
st.image(image, use_container_width=True)

# Load data
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris

iris = load_iris_data()

# Sidebar for hyperparameters
st.sidebar.header("Model Hyperparameters")

# Hyperparameter controls
max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=10, value=4, help="Maximum depth of the tree")
eta = st.sidebar.slider("Learning Rate (eta)", min_value=0.01, max_value=1.0, value=0.3, step=0.01, help="Learning rate for boosting")
epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=10, help="Number of boosting rounds")
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Proportion of data for testing")

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=test_size, random_state=42
)

# Display dataset info
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Information")
    st.write(f"**Total samples:** {iris.data.shape[0]}")
    st.write(f"**Features:** {iris.data.shape[1]}")
    st.write(f"**Classes:** {list(iris.target_names)}")
    st.write(f"**Training samples:** {len(X_train)}")
    st.write(f"**Test samples:** {len(X_test)}")

with col2:
    st.subheader("Feature Names")
    feature_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Description': [
            'Sepal length in cm',
            'Sepal width in cm', 
            'Petal length in cm',
            'Petal width in cm'
        ]
    })
    st.dataframe(feature_df, use_container_width=True)

# Data visualization
st.subheader("📈 Data Visualization")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Feature Distribution", "Feature Correlations", "Class Distribution"])

with tab1:
    # Feature distribution by class
    fig = make_subplots(rows=2, cols=2, subplot_titles=iris.feature_names)
    
    for i, feature_name in enumerate(iris.feature_names):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        for class_idx, class_name in enumerate(iris.target_names):
            class_data = iris.data[iris.target == class_idx, i]
            fig.add_trace(
                go.Histogram(x=class_data, name=f"{class_name}", opacity=0.7),
                row=row, col=col
            )
    
    fig.update_layout(height=600, title_text="Feature Distribution by Class")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Correlation matrix
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target_names[iris.target]
    
    corr_matrix = df.drop('target', axis=1).corr()
    fig = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Class distribution
    class_counts = pd.Series(iris.target).value_counts().sort_index()
    fig = px.bar(
        x=[iris.target_names[i] for i in class_counts.index],
        y=class_counts.values,
        title="Class Distribution",
        labels={'x': 'Iris Species', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Model training section
st.subheader("XGBoost Model Training")

# Display current parameters
st.write("**Current Parameters:**")
param_display = pd.DataFrame({
    'Parameter': ['Max Depth', 'Learning Rate', 'Epochs', 'Test Size'],
    'Value': [max_depth, eta, epochs, f"{test_size:.1%}"]
})
st.dataframe(param_display, use_container_width=True)

# Train button
if st.button("Train Model", type="primary"):
    with st.spinner("Training XGBoost model..."):
        # Prepare data
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters
        param = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': 'multi:softmax',
            'num_class': 3
        }
        
        # Train model
        model = xgb.train(param, train, epochs)
        
        # Make predictions
        predictions = model.predict(test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Store results in session state
        st.session_state.model = model
        st.session_state.predictions = predictions
        st.session_state.accuracy = accuracy
        st.session_state.y_test = y_test
        
        st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")

# Display results if model exists
if 'model' in st.session_state:
    st.subheader("Model Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{st.session_state.accuracy:.4f}")
        
        # Classification report
        st.write("**Classification Report:**")
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.predictions, 
            target_names=iris.target_names,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    with col2:
        # Confusion matrix
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=iris.target_names,
            y=iris.target_names
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    # Get feature importance scores
    importance_scores = st.session_state.model.get_score(importance_type='gain')
    
    # Create feature importance dataframe
    feature_importance = []
    for i, feature_name in enumerate(iris.feature_names):
        score = importance_scores.get(f"f{i}", 0)
        feature_importance.append({
            'Feature': feature_name,
            'Importance': score
        })
    
    importance_df = pd.DataFrame(feature_importance).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance (Gain)",
        labels={'Importance': 'Gain Score', 'Feature': 'Feature Name'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction details
    st.subheader("Prediction Details")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Actual': [iris.target_names[y] for y in st.session_state.y_test],
        'Predicted': [iris.target_names[int(p)] for p in st.session_state.predictions],
        'Correct': [iris.target_names[y] == iris.target_names[int(p)] for y, p in zip(st.session_state.y_test, st.session_state.predictions)]
    })
    
    # Color code the results
    def color_correct(val):
        if val:
            return 'background-color: lightgreen'
        else:
            return 'background-color: lightcoral'
    
    st.dataframe(results_df.style.applymap(color_correct, subset=['Correct']), use_container_width=True)


