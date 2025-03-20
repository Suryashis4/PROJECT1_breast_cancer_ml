import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Breast Cancer Prediction")

# Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)

        # Ensure the Excel file has the correct columns
        expected_columns = [
            "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean",
            "Smoothness Mean", "Compactness Mean", "Concavity Mean", "Concave Points Mean",
            "Symmetry Mean", "Fractal Dimension Mean", "Radius SE", "Texture SE",
            "Perimeter SE", "Area SE", "Smoothness SE", "Compactness SE",
            "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
            "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst",
            "Smoothness Worst", "Compactness Worst", "Concavity Worst", "Concave Points Worst",
            "Symmetry Worst", "Fractal Dimension Worst"
        ]
        
        if list(df.columns) != expected_columns:
            st.error("⚠️ The uploaded file does not have the correct column names. Please use the correct format.")
        else:
            # Convert to numpy array
            features_array = df.to_numpy()

            # Make predictions
            predictions = model.predict(features_array)

            # Add predictions to the dataframe
            df["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in predictions]

            # Display results
            st.write("### Predictions:")
            st.dataframe(df)

            # Provide download link for results
            output_file = "breast_cancer_predictions.xlsx"
            df.to_excel(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button("Download Predictions", f, file_name=output_file)

    except Exception as e:
        st.error(f"Error processing file: {e}")
