#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing the necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://raw.githubusercontent.com/tamanna1312/TASClassification/main/Applogo.jpg');
                background-repeat: no-repeat;
                background-size: 150px 150px; /* Set explicit width and height */
                background-position: 30px 10px; /* Position it in the top left */
                # margin-top: 20px; /* Add space above */
                padding-top: 170px; /* Add space below to separate from text */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.image(
    'Goethe-Logo.gif')

# App title
st.title("Synthetic Data Generator and Correlation Analysis")
st.markdown("""
- **What is the Synthetic Data Generator?**  
  It is a tool designed to generate syntehtic geochemical data. It utilises statistical techniques (e.g., mean and covariance calculations) to copy the characteristics of real-world data.

- **Results:**  
  - **Correlation Matrices**: Displays the relationships between oxides in both real and synthetic data.  
  1. These matrices reveal how different oxides relate to one another. For example:  
  2.  A high positive value indicates a strong positive correlation (e.g., when one oxide increases, the other also increases).  
  3. A negative value suggests an inverse relationship. 
  - **Difference Matrix**: Highlights any deviations between real and synthetic data correlations.  
  1. Smaller differences indicate the synthetic data closely matches the real data's statistical structure.  
  2. Larger differences suggest areas where the generated data diverges from real-world patterns.
  - **Synthetic Dataset**: A downloadable CSV file containing the synthetic geochemical data.  
""")

use_test_data = st.toggle("Use Test Data", value=False)
test_data_path = "NormalisedFilteredandGEOROCData.csv"

if use_test_data:
    real_data = pd.read_csv(test_data_path)
        # st.write("Preview of Test Data:")
    st.dataframe(real_data.head())
else:
    st.header("Upload Real Data")
    uploaded_file = st.file_uploader("Upload your CSV file containing the real data", type="csv")
    
    if uploaded_file:
        real_data = pd.read_csv(uploaded_file)
        st.write("Preview of Real Data:")
        st.dataframe(real_data.head())

if 'real_data' in locals():
    required_columns = ['Rock_name']  # Add other required columns here
    missing_columns = [col for col in required_columns if col not in real_data.columns]
    
    if missing_columns:
        st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
    else:
        if st.button("Produce Synthetic Data"):
            n_samples = 100
            fake_data = []

            for rock_type in real_data['Rock_name'].unique():
                rock_data = real_data[real_data['Rock_name'] == rock_type]
                rock_data_transformed = np.log1p(rock_data.drop(columns=['Rock_name']))
                cov_matrix = rock_data_transformed.cov()
                mean_vector = rock_data_transformed.mean()
                
                fake_transformed = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_samples)
                fake_original_scale = np.expm1(fake_transformed)
                fake_rock_data = pd.DataFrame(fake_original_scale, columns=rock_data.columns.drop('Rock_name'))
                fake_rock_data['Rock_name'] = rock_type
                fake_data.append(fake_rock_data)

            fake_data_combined = pd.concat(fake_data, ignore_index=True)

            # Calculate correlation matrices
            real_corr = real_data.drop(columns=['Rock_name']).corr()
            fake_corr = fake_data_combined.drop(columns=['Rock_name']).corr()
            diff_corr = np.abs(real_corr - fake_corr)

            # Display correlation matrices
            st.header("Correlation Matrices")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", annot=True, fmt=".2f")
            axes[0].set_title("Real Data Correlation Matrix")
            
            sns.heatmap(fake_corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f")
            axes[1].set_title("Synthetic Data Correlation Matrix")
            
            st.pyplot(fig)
            
            # Display difference matrix
            st.subheader("Difference in Correlation Matrices (Real vs Synthetic)")
            fig_diff, ax_diff = plt.subplots(figsize=(8, 6))
            sns.heatmap(diff_corr, cmap="viridis", annot=True, fmt=".2f", ax=ax_diff)
            ax_diff.set_title("Difference Matrix")
            st.pyplot(fig_diff)

            # Show mean absolute difference
            mean_diff = diff_corr.mean().mean()
            st.write(f"**Mean Absolute Difference in Correlations:** {mean_diff:.4f}")

            # Provide a download button for fake data
            st.header("Download Synthetic Data")
            csv = fake_data_combined.to_csv(index=False)
            st.download_button(
                label="Download Synthetic Data as CSV",
                data=csv,
                file_name="fSynthetic_data.csv",
                mime="text/csv"
            )
