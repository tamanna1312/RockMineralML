#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing the necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU  
from sklearn.preprocessing import StandardScaler

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

#title of the app.
st.title('TAS Rock Classifier')
# Introduction section
st.subheader("What is TAS Rock Classifier?")
st.markdown(
    """
    The **TAS Rock Classifier** is a machine learning (ML) tool designed for the automatic classification of volcanic rocks. 

    - **Dataset**: Upload your own geochemical dataset (See the provided template for reference).
    - **Classification Results**: Predicted rock type column in your uploaded data.
    - **Visualisation**: Results are plotted on a TAS diagram.
   
    **Supported Cases**
    1. **All Oxides**: Uses all 10 major and minor oxides.
    2. **No SiO₂**: Excludes SiO₂.
    3. **No Alkali Oxides**: Excludes Na₂O and K₂O.

    **Results** 
    - The app validates your data by checking the element oxides requirements according to the case selected, predicts rock types, and then displays:
       1. A table with your dataset and the predicted rock types.
       2. A TAS plot showing the classification visually.
   
    """
)


#loading the model for the 3 cases.
def load_model_for_case(case):
    if case == 'All Oxides':
        # return load_model('fine_tuned_model.h5', custom_objects={'LeakyReLU': LeakyReLU})
        return load_model('TASClassification.h5', custom_objects={'LeakyReLU': LeakyReLU})
    elif case == 'No SiO2':
        return load_model('fine_tuned_model_noSiO2.h5', custom_objects={'LeakyReLU': LeakyReLU})
    elif case == 'No Alkali Oxides':
        return load_model('fine_tuned_model_noAlkali.h5', custom_objects={'LeakyReLU': LeakyReLU})

def TAS(ax,fontsize=8):
    from collections import namedtuple
    FieldLine = namedtuple('FieldLine', 'x1 y1 x2 y2')
    lines = (FieldLine(x1=41, y1=0, x2=41, y2=7),
             FieldLine(x1=41, y1=7, x2=52.5, y2=14),
             FieldLine(x1=45, y1=0, x2=45, y2=5),
             FieldLine(x1=41, y1=3, x2=45, y2=3),
             FieldLine(x1=45, y1=5, x2=61, y2=13.5),
             FieldLine(x1=45, y1=5, x2=52, y2=5),
             FieldLine(x1=52, y1=5, x2=69, y2=8),
             FieldLine(x1=49.4, y1=7.3, x2=52, y2=5),
             FieldLine(x1=52, y1=5, x2=52, y2=0),
             FieldLine(x1=48.4, y1=11.5, x2=53, y2=9.3),
             FieldLine(x1=53, y1=9.3, x2=57, y2=5.9),
             FieldLine(x1=57, y1=5.9, x2=57, y2=0),
             FieldLine(x1=52.5, y1=14, x2=57.6, y2=11.7),
             FieldLine(x1=57.6, y1=11.7, x2=63, y2=7),
             FieldLine(x1=63, y1=7, x2=63, y2=0),
             FieldLine(x1=69, y1=12, x2=69, y2=8),
             FieldLine(x1=45, y1=9.4, x2=49.4, y2=7.3),
             FieldLine(x1=59.4, y1=10.5, x2=68.8, y2=10.5),
             FieldLine(x1=69, y1=8, x2=77, y2=0)
            )

    FieldName = namedtuple('FieldName', 'name x y rotation')
    names = (FieldName('Picro\nbasalt', 43, 2, 0),
             FieldName('Basalt', 48.5, 2, 0),
             FieldName('Basaltic\nandesite', 54.5, 3.9, 0),
             FieldName('Andesite', 60, 2, 0),
             FieldName('Dacite', 68.5, 2, 0),
             FieldName('Rhyolite', 75.5, 6.5, 0),
             FieldName('Trachyte',
                       64.5, 12.5, 0),
             FieldName('Trachy\ndacite',
                       64.7, 9.5, 0),
             FieldName('2', 52.6, 7.3, 0),
             FieldName('1', 48.7, 6.2, 0),
             FieldName('3', 57.2, 8.8, 0),
             FieldName('Phono\ntephrite', 49, 9.9, 0),
             FieldName('Tephri\nphonolite', 53.0, 12.1, 0),
             FieldName('Phonolite', 57.5, 14.5, 0),
             FieldName('Tephrite', 45, 7.3, 0),
             FieldName('Foidite', 44, 11.5, 0),
             FieldName('Basa\nnite', 43, 4.5, 0),
             FieldName('1: Trachybasalt', 79, 15.8, 0),
             FieldName('2: Basaltic Trachyandesite', 79, 15.3, 0),
             FieldName('3: Trachyandesite', 79, 14.8, 0))
            

    for line in lines:
        ax.plot([line.x1, line.x2], [line.y1, line.y2],
                       '-', color='black', zorder=2)
    for name in names:
        if '1:' in name.name or '2:' in name.name or '3:' in name.name:
            ax.text(name.x, name.y, name.name, color='black', size=8,  
                    horizontalalignment='right', verticalalignment='top',
                    rotation=name.rotation, zorder=2) 
        else:
            ax.text(name.x, name.y, name.name, color='black', size=13,
                    horizontalalignment='center', verticalalignment='top',
                    rotation=name.rotation, zorder=2,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2')) 
    # ax.text(0.95, 0.95, "1: Field Name 1\n2: Field Name 2\n3: Field Name 3", 
    #         transform=ax.transAxes, fontsize=12, color='black', 
    #         verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

rock_colors = {
    'Rhyolite': 'y',
    'Basalt': 'r',
    'Andesite': 'b',
    'Dacite': 'g',
    'Basanite': 'm',
    'Trachyte': 'c',
    'Tephrite': 'brown',
    'Phonolite': 'gray',
    'Phonotephrite': 'pink',
    'Trachybasalt': 'purple',
    'Trachyandesite': 'brown',
    'Basaltic Andesite': '#FFA500',
    'Picrobasalt': 'black',
    'Tephri-Phonolite': 'r',
    'Basaltic Trachyandesite': 'm',
    'Foidite': 'y',
    'Foidite 2': 'y',
    'Trachy-Dacite': '#FFA500',
}
tas_coordinates = {
    'Picrobasalt': (43, 2),
    'Basalt': (48.5, 2),
    'Basaltic Andesite': (54.5, 3.9),
    'Andesite': (60, 2),
    'Dacite': (68.5, 2),
    'Rhyolite': (75.5, 6.5),
    'Trachyte': (64.5, 12.5),
    'Trachydacite': (64.7, 9.5),
    'Phonotephrite': (49, 9.9),
    'Tephriphonolite': (53.0, 12.1),
    'Phonolite': (57.5, 14.5),
    'Tephrite': (45, 7.3),
    'Foidite': (44, 11.5),
    'Basanite': (43, 4.5),
    'Trachybasalt' : (49, 6.2),
    'Basaltictrachyandesite': (53, 8), 
    'Trachyandesite': (57.2, 9)
}

case_column_requirements = {
    'All Oxides': ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)', 'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O', 'Na2O+K2O/SiO2'],
    'No SiO2': ['TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)', 'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O'],
    'No Alkali Oxides': ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)', 'MnO(wt%)', 'P2O5(wt%)']
}
# validating the columns of the csv file.
def validate_columns(data, case):
    required_columns = case_column_requirements.get(case, [])
    if not required_columns:
        return True, None
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, missing_columns
    return True, None

#normalising the data.
def normalise_data(data, case):
    scaler = StandardScaler()
    features = data
    normalised_data = scaler.fit_transform(features)
    return normalised_data
    
#arranging the columns of the test data in the same way as that of training data.
def arrange_columns(data, case):
    if case == 'All Oxides':
        column_order = ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O', 'Na2O+K2O/SiO2']
    elif case == 'No SiO2':
        column_order = ['TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O']
    elif case == 'No Alkali Oxides':
        column_order = ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)']
    
    return data[column_order]


#mapping key.
label_to_rock = {0: 'Andesite', 1: 'Basalt', 2: 'Basaltic Andesite', 3: 'Basanite', 4:'Dacite',
                 5: 'Foidite', 6: 'Phonolite', 7: 'Phonotephrite', 8: 'Picrobasalt', 
                 9: 'Rhyolite', 10: 'Tephrite', 11: 'Trachyandesite', 12: 'Trachybasalt', 
                 13: 'Trachydacite', 14: 'Trachyte'}

test_data_path = "Altered1TestFinetunded20.csv"  

st.write('See template to upload your data.')
template_file_path = "Template.csv"
template_data = pd.read_csv(template_file_path)
with st.expander("View Template File"):
    st.dataframe(template_data)
st.write('You can use test data for demo or upload your own csv file.')
use_test_data = st.toggle("Test data")

if use_test_data:
    data = pd.read_csv(test_data_path)
else:
    uploaded_file = st.file_uploader('Upload csv file' ,type=["csv"], key="file_uploader_key")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(data.head())

if 'data' in locals():
    rock_counts_dict = {}
    case_options = [
        "**_All Oxides_**",
        "**_No SiO₂_**", 
        "**_No Alkali Oxides (Na₂O, K₂O)_**",
        "Compare"
    ]
    case = st.radio(
        "Select the case:",
        case_options,
        index=0,  
        horizontal=True,
        format_func=lambda x: x  
    )
    case_mapping = {
        "**_All Oxides_**": 'All Oxides',
        "**_No SiO₂_**": 'No SiO2',
        "**_No Alkali Oxides (Na₂O, K₂O)_**": 'No Alkali Oxides',
        "Compare": 'Compare'
    }
    
    selected_case = case_mapping[case]  
    if selected_case != 'Compare':
        is_valid, missing_cols = validate_columns(data, selected_case)
        if not is_valid:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            arranged_data = arrange_columns(data, selected_case)
            model = load_model_for_case(selected_case)
            normalised_data = normalise_data(arranged_data, selected_case)
            predictions = model.predict(normalised_data)
            predicted_labels = np.argmax(predictions, axis=1)  
            predicted_rock_types = [label_to_rock[label] for label in predicted_labels]
            arranged_data.insert(0, 'Predicted_Rock_Type', predicted_rock_types) 
            st.write(arranged_data)
            rock_counts = pd.Series(predicted_rock_types).value_counts().reindex(label_to_rock.values(), fill_value=0)
            rock_counts_dict[selected_case] = rock_counts
            csv = arranged_data.to_csv(index=False)
        # st.download_button(label="Download Predicted rock type file as CSV",
        #                    data=csv,
        #                    file_name='predicted_rock_types.csv',
        #                    mime='text/csv')
        if selected_case == 'All Oxides':
            st.write('All samples with number of data for each rock type plotted on TAS plot:')
            fig, ax = plt.subplots(figsize=(8, 6))
            TAS(ax)
            ax.set_xlim([40, 80])
            ax.set_ylim([0, 16])
            ax.set_xlabel(r'SiO$_2$ (wt%)')
            ax.set_ylabel(r'Na$_2$O+K$_2$O (wt%)')
            ax.tick_params(axis='x', direction='inout', length=8, width=1, colors='black', top=True)
            ax.tick_params(axis='y', direction='inout', length=8, width=1, colors='black', right=True)
            for rock_type in tas_coordinates.keys():
                sio2, na2o_k2o = tas_coordinates[rock_type]
                rock_samples = arranged_data[arranged_data['Predicted_Rock_Type'] == rock_type]
                ax.scatter(rock_samples['SiO2(wt%)'], rock_samples['Na2O+K2O'], label=rock_type, zorder=1, s=2)  
                rock_count = len(rock_samples)
                if rock_count > 0:
                    ax.text(sio2, na2o_k2o + 0.5, str(rock_count), fontsize=12, ha='center', va='center', color='green')
                    # bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
            st.pyplot(fig)
        if 'Predicted_Rock_Type' in arranged_data.columns:
            if selected_case == 'No SiO2':
                st.write('Total number of data points for each rock type on TAS plot:')
                fig, ax = plt.subplots(figsize=(8, 6))
                TAS(ax)  
                ax.set_xlim([40, 80])  
                ax.set_ylim([0, 16])
                ax.set_xlabel(r'SiO$_2$ (wt%)')  
                ax.set_ylabel(r'Na$_2$O+K$_2$O (wt%)')
                ax.tick_params(axis='x', direction='inout', length=8, width=1, colors='black', top=True)
                ax.tick_params(axis='y', direction='inout', length=8, width=1, colors='black', right=True)
                rock_counts = arranged_data['Predicted_Rock_Type'].value_counts()
                for rock_type, count in rock_counts.items():
                    if rock_type in tas_coordinates:
                        sio2, na2o_k2o = tas_coordinates[rock_type]
                        ax.text(sio2, na2o_k2o + 0.5, str(count), fontsize=12, ha='center', va='center', color='green')
                            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

                st.pyplot(fig)
            elif selected_case == 'No Alkali Oxides':
                st.write('Total number of data points for each rock type on TAS plot:')
                fig, ax = plt.subplots(figsize=(8, 6))
                TAS(ax)  
                ax.set_xlim([40, 80])  
                ax.set_ylim([0, 16])
                ax.set_xlabel(r'SiO$_2$ (wt%)')  
                ax.set_ylabel(r'Na$_2$O+K$_2$O (wt%)')
                ax.tick_params(axis='x', direction='inout', length=8, width=1, colors='black', top=True)
                ax.tick_params(axis='y', direction='inout', length=8, width=1, colors='black', right=True)
                rock_counts = arranged_data['Predicted_Rock_Type'].value_counts()
                for rock_type, count in rock_counts.items():
                    if rock_type in tas_coordinates:
                        sio2, na2o_k2o = tas_coordinates[rock_type]
                        ax.text(sio2, na2o_k2o + 0.5, str(count), fontsize=12, ha='center', va='center', color='green')
                            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

                st.pyplot(fig)



    else:
        st.write("Comparison of rock type counts across all cases:")
        rock_type_counts = {}
        for case_name, case_key in {'All Oxides': 'All Oxides', 'No SiO₂': 'No SiO2', 'No Alkali Oxides': 'No Alkali Oxides'}.items():
            is_valid, missing_cols = validate_columns(data, case_key)
            if not is_valid:
                st.error(f"Missing required columns for {case_name}: {', '.join(missing_cols)}")
                continue

            arranged_data = arrange_columns(data, case_key)
            model = load_model_for_case(case_key)
            normalised_data = normalise_data(arranged_data, case_key)
            predictions = model.predict(normalised_data)
            predicted_labels = np.argmax(predictions, axis=1)
            predicted_rocks = [label_to_rock[label] for label in predicted_labels]
            rock_type_counts[case_name] = pd.Series(predicted_rocks).value_counts().reindex(label_to_rock.values(), fill_value=0)

        if rock_type_counts:
            comparison_df = pd.DataFrame(rock_type_counts)
            comparison_df.index.name = 'Rock Type'
            comparison_df.reset_index(inplace=True)
            st.dataframe(comparison_df)
        
        

            













































# # @st.cache(allow_output_mutation=True)
# # def load_results():
# #     # Load the precomputed predictions
# #     results = pd.read_csv('predicted_results.csv')
# #     return results

# # results = load_results()

# # Button to show results
# if st.button('Upload Data'):
#     # Display the DataFrame with the inverse-transformed features and predictions
#     st.write("Data with Predicted Rock Types")
#     st.dataframe(results)
# else:
#     st.write("Click the button to upload your data.")

