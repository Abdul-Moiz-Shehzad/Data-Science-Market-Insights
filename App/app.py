import streamlit as st
import pandas as pd
from SalaryPredictor import SalaryPredictor

predictor = SalaryPredictor()

st.title("Salary Prediction App")
st.sidebar.title("Options")

if 'data' not in st.session_state:
    st.session_state.data = None

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"], key="upload_dataset")
if uploaded_file:
    st.write("Dataset Preview:")
    st.session_state.data = pd.read_csv(uploaded_file)
    st.write(st.session_state.data.head())

    st.write("Preprocessing the dataset...")
    try:
        X_train, X_test, y_train, y_test = predictor.load_and_preprocess(st.session_state.data)
        st.success("Dataset successfully preprocessed.")
        
        if 'model_choice' not in st.session_state:
            st.session_state.model_choice = None
            
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")

if st.session_state.data is not None:
    st.sidebar.subheader("Train a Model")
    model_option = st.sidebar.selectbox("Choose a Model to Train", 
                                        ["Ridge Regression", "Lasso Regression", 
                                         "Linear Regression", "Random Forest", 
                                         "Gradient Boosting"])
    
    if st.sidebar.button("Train Model"):
        with st.spinner('Training the model...'):
            try:
                if model_option == "Ridge Regression":
                    result = predictor.train_ridge(X_train, X_test, y_train, y_test)
                elif model_option == "Lasso Regression":
                    important_features, result = predictor.train_lasso(X_train, X_test, y_train, y_test)
                    st.write("Important Features:")
                    st.write(important_features)
                elif model_option == "Linear Regression":
                    result = predictor.train_linear_regression(X_train, X_test, y_train, y_test)
                elif model_option == "Random Forest":
                    result = predictor.train_random_forest(X_train, X_test, y_train, y_test)
                elif model_option == "Gradient Boosting":
                    result = predictor.train_gradient_boost(X_train, X_test, y_train, y_test)

                st.session_state.model_choice = model_option
                
                st.write("Model Evaluation:")
                st.write(f"RMSE: {result['rmse']}")
                st.write(f"R²: {result['r2']}")
            except Exception as e:
                st.error(f"Error during training: {e}")
    st.sidebar.subheader('Make a prediction (Will be added soon)')
else:
    if not uploaded_file:
        st.info("Please upload a dataset to begin.")
