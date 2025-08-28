#  XGBoost Iris Classification Streamlit App

This is a machine learning pipeline integrated with an interactive Streamlit-based web application. It enables real-time inference through the XGBoost algorithm applied to the IRIS dataset.

##  Features

- **Interactive Hyperparameter Tuning**: Adjust max depth, learning rate, epochs, and test size
- **Real-time Model Training**: Train XGBoost models with custom parameters
- ** Visualizations**: 
  - Feature distributions by class
  - Feature correlation matrix
  - Class distribution
  - Confusion matrix
  - Feature importance plots
- **Detailed Results**: Accuracy metrics, classification reports, and prediction details

##  Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

##  How to Run

1. **Navigate** to the project directory
2. **Run the Streamlit app**:
   ```bash
   streamlit run xgboost_streamlit_app.py
   ```
3. **Open your browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)


##  Key Parameters

- **Max Depth**: Controls the maximum depth of decision trees (1-10)
- **Learning Rate (eta)**: Controls how much each tree contributes (0.01-1.0)
- **Epochs**: Number of boosting rounds (1-100)
- **Test Size**: Proportion of data reserved for testing (10%-50%)

##  Troubleshooting

- **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

## Output

**I have included screenshots to illustrate the expected observations. Please refer to the output layout images.**




