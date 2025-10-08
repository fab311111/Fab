# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

# Load the trained model and scaler
best_rf_model = None
scaler = None
df_original = None
expected_feature_columns = None

try:
    best_rf_model = joblib.load('best_rf_model.joblib')
    scaler = joblib.load('scaler.joblib')

    # Load the original dataset to retrieve employee data by ID
    df_original = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Assuming the original training columns are needed for consistent feature ordering
    # This would ideally be saved during training or derived from the original data loading/preprocessing
    # For now, let's derive the expected feature columns based on the preprocessing steps in the notebook.

    # Drop irrelevant/constant columns as done in the notebook
    columns_to_drop_for_template = [
        'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'
    ]
    # Create a temporary copy to avoid modifying df_original before retrieving data by ID
    temp_df_for_columns = df_original.drop(columns=columns_to_drop_for_template).copy()

    # Identify original numerical and categorical columns
    original_categorical_cols = temp_df_for_columns.select_dtypes(include=['object']).columns
    original_numerical_cols = temp_df_for_columns.select_dtypes(include=[np.number]).columns.drop('Attrition')

    # Perform one-hot encoding on a temporary dataframe to get the list of expected feature columns
    # This ensures the order matches the training data
    temp_df_for_columns = pd.get_dummies(temp_df_for_columns.drop('Attrition', axis=1), columns=original_categorical_cols, drop_first=True)
    expected_feature_columns = temp_df_for_columns.columns.tolist()


    st.success("Model, scaler, and original data loaded successfully.")

except FileNotFoundError:
    st.error("Error: Model, scaler, or original data file not found. Please ensure 'best_rf_model.joblib', 'scaler.joblib', and 'WA_Fn-UseC_-HR-Employee-Attrition.csv' are in the correct directory.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"An error occurred during loading: {e}")
    st.stop() # Stop the app if any other loading error occurs


# Add a title and a brief description for the application
st.title('Employee Attrition Prediction')
st.write("""
This application predicts the likelihood of an employee leaving the company based on their characteristics.
Please enter the employee number to get their predicted attrition risk and potential intervention areas.
""")

# Add a header for the input section
st.header('Enter Employee Information')

# Add a numerical input field for the "Employee Number"
if df_original is not None:
    employee_number_input = st.number_input(
        'Employee Number',
        min_value=int(df_original['EmployeeNumber'].min()), # Set minimum to the smallest employee number
        max_value=int(df_original['EmployeeNumber'].max()), # Set maximum to the largest employee number
        value=int(df_original['EmployeeNumber'].min()), # Default to the smallest employee number
        step=1
    )
else:
    st.warning("Could not load original data to set employee number range.")
    employee_number_input = st.number_input(
        'Employee Number',
        min_value=1, # Set minimum to a default value
        max_value=10000, # Set maximum to a default value
        value=1, # Default to a default value
        step=1
    )


# Function to retrieve and prepare input data for prediction based on Employee Number
def prepare_input_data_by_id(employee_id, df_original, scaler, original_numerical_cols, original_categorical_cols, expected_columns):
    # Retrieve the row for the given EmployeeNumber
    employee_data = df_original[df_original['EmployeeNumber'] == employee_id].copy()

    # Check if an employee with the given ID exists
    if employee_data.empty:
        return None, "Employee Number not found in the dataset.", None

    # Store original data for intervention suggestions later
    original_employee_data = employee_data.iloc[0].copy()

    # Drop irrelevant columns as done during training
    columns_to_drop = [
        'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber', 'Attrition' # Also drop Attrition as it's the target
    ]
    employee_data_processed = employee_data.drop(columns=columns_to_drop)

    # Apply one-hot encoding to the employee data
    # Ensure all possible dummy columns are created, even if not present in this single row
    # A robust way is to create a template DataFrame with all expected columns and fill with 0
    input_df = pd.DataFrame(columns=expected_columns)
    input_df.loc[0] = 0 # Initialize with zeros

    # Fill in the numerical values from the employee data
    for col in original_numerical_cols:
        if col in employee_data_processed.columns:
             input_df.loc[0, col] = employee_data_processed.iloc[0][col]


    # Fill in the one-hot encoded categorical values
    # This requires knowing the exact column names created by get_dummies with drop_first=True
    # A more robust way is to save the list of columns from X_train or the one-hot encoder itself.
    # For this implementation, we reconstruct based on original columns and expected dummy column names.
    for col in original_categorical_cols:
        if col in employee_data_processed.columns:
            category_value = employee_data_processed.iloc[0][col]
            dummy_col_name = f'{col}_{category_value}'
            # Check if this specific dummy column name is in our expected columns (due to drop_first=True)
            if dummy_col_name in expected_columns:
                 input_df.loc[0, dummy_col_name] = 1

    # Ensure the columns are in the correct order
    input_df = input_df[expected_columns]


    # Scale numerical features using the loaded scaler
    # Identify numerical columns in the input_df based on the original numerical columns list
    numerical_cols_in_input = original_numerical_cols[original_cols_in_input.isin(input_df.columns)]
    input_df[numerical_cols_in_input] = scaler.transform(input_df[numerical_cols_in_input])


    # Return the preprocessed data and original data
    return input_df, original_employee_data, None # Return the prepared DataFrame, original data, and no error message

# Function to suggest intervention areas based on employee data
# This function now correctly uses the original employee data Series
def suggest_interventions(employee_data, df_original):
    interventions = []

    # Example intervention suggestions based on key features from earlier analysis and general risk factors
    # Access values using employee_data (the original Series for the employee)
    # Added checks for column existence for robustness

    # Lower MonthlyIncome might be a risk factor
    if 'MonthlyIncome' in employee_data and 'MonthlyIncome' in df_original.columns and employee_data['MonthlyIncome'] < df_original['MonthlyIncome'].mean() * 0.8: # Example threshold (bottom 20% approx)
         interventions.append("- Review compensation and benefits.")

    # OverTime is a significant feature
    if 'OverTime' in employee_data and employee_data['OverTime'] == 'Yes':
        interventions.append("- Address workload and consider work-life balance initiatives.")

    # Shorter tenure might be a risk factor
    if 'YearsAtCompany' in employee_data and employee_data['YearsAtCompany'] < 3: # Example threshold
         interventions.append("- Provide mentorship and career development opportunities for newer employees.")
    if 'YearsWithCurrManager' in employee_data and employee_data['YearsWithCurrManager'] < 2: # Example threshold
         interventions.append("- Facilitate improved relationships with managers and provide support in current role.")


    # Lower satisfaction levels are often linked to attrition
    # Assuming satisfaction scales are 1-4, check for values 1 or 2
    if 'JobSatisfaction' in employee_data and employee_data['JobSatisfaction'] < 3:
        interventions.append("- Discuss job role satisfaction and explore opportunities for engagement.")
    if 'EnvironmentSatisfaction' in employee_data and employee_data['EnvironmentSatisfaction'] < 3:
        interventions.append("- Assess work environment factors.")
    if 'RelationshipSatisfaction' in employee_data and employee_data['RelationshipSatisfaction'] < 3:
         interventions.append("- Facilitate improved relationships with colleagues or managers.")
    if 'WorkLifeBalance' in employee_data and employee_data['WorkLifeBalance'] < 3:
        interventions.append("- Support work-life balance through flexible arrangements or workload management.")

    # Business Travel Frequency
    if 'BusinessTravel' in employee_data and employee_data['BusinessTravel'] == 'Travel_Frequently':
         interventions.append("- Review business travel frequency and its impact on work-life balance.")

    # Add other relevant checks based on your analysis (e.g., Job Role, Department if they showed high attrition rates)

    return interventions


# Prediction button
if st.button('Predict Attrition and Suggest Interventions'):
    # Prepare the input data using the employee number
    if df_original is not None and scaler is not None and expected_feature_columns is not None:
        input_df_processed, original_employee_data, error_message = prepare_input_data_by_id(
            employee_number_input,
            df_original,
            scaler,
            original_numerical_cols,
            original_categorical_cols,
            expected_feature_columns
        )

        if error_message:
            st.error(error_message)
        else:
            # Make prediction
            # predict_proba returns probabilities for both classes [prob_class_0, prob_class_1]
            # We want the probability of attrition (class 1)
            prediction_proba = best_rf_model.predict_proba(input_df_processed)[:, 1]

            # Display the predicted probability
            st.subheader('Prediction Result')
            st.write(f'Predicted Attrition Likelihood: **{prediction_proba[0]:.2f}**')

            # Interpretation based on probability and suggestion of interventions
            if prediction_proba[0] >= 0.5: # Threshold for suggesting interventions
                st.warning('Based on the model, this employee is predicted to be at High Risk of Attrition.')
                st.write("It might be beneficial to look into factors that could be contributing to this risk.")

                # Suggest intervention areas for high-risk employees
                st.subheader('Suggested Intervention Areas')
                intervention_suggestions = suggest_interventions(original_employee_data, df_original)


                if intervention_suggestions:
                    for suggestion in intervention_suggestions:
                        st.markdown(suggestion)
                else:
                    st.write("No specific intervention areas identified based on current rules for this employee.")


            elif prediction_proba[0] >= 0.3: # Moderate risk (adjust thresholds as needed)
                 st.info('This employee is predicted to be at Moderate Risk of Attrition.')
                 st.write("Monitor this employee's situation. While immediate action may not be necessary, understanding potential factors is helpful.")
                 # Optionally suggest interventions for moderate risk as well, perhaps a less extensive list
                 # st.subheader('Potential Areas to Monitor')
                 # intervention_suggestions = suggest_interventions(original_employee_data)
                 # if intervention_suggestions:
                 #     for suggestion in intervention_suggestions:
                 #         st.markdown(suggestion)


            else: # Low risk
                st.success('This employee is predicted to be at Low Risk of Attrition.')
                st.write("Based on the current information, this employee is likely to stay.")

            # Include a disclaimer
            st.write("""
            *Note: This prediction is based on the trained machine learning model and the input data provided.
            It is a probabilistic estimate and may not capture all factors influencing an employee's decision to leave.
            Intervention suggestions are based on general patterns and employee characteristics, not a definitive diagnosis.*
            """)
    else:
        st.error("Application not fully loaded. Please ensure model, scaler, and data files are available and try again.")
