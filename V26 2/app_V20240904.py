# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:11:01 2024

"""
###############################################################################
# Load libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import joblib
import pandas as pd, numpy as np


print('Libraries loaded')

###############################################################################
# Section when the app initialize and load the required information
@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():
    # Load Examples
    example_path = r'data'
    example_filename = r'/examples.xlsx'
    example_data = pd.read_excel(example_path + example_filename)
    print('File loaded -->' , example_path + example_filename)
    
    # Load original data
    input_path = r'data'
    input_filename = r'/reduced_op_UK_merged_data_final_21062024.csv'
    input_data = pd.read_csv(input_path + input_filename , decimal = '.' , sep = ';')
    print('File loaded -->' , input_path + input_filename)
    
    # Load Pipeline
    path_model = r'models'
    model_filename = r'/26_2_Best_Model_with_Configuration.pkl'
    with open(path_model + model_filename, 'rb') as file:
        model = joblib.load(file)
    print('File loades -->' , path_model + model_filename)
    
    # Define columns
    selected_features = ['sex', 'age', 'bmi', 'active_smoking',
                        'preoperative_hemoglobin_level', 'neoadjuvant_therapy',
                        'asa_score', 'prior_abdominal_surgery', 'indication', 'operation',
                        'emergency_surgery', 'approach',
                        'type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis',
                        'anastomotic_technique', 'anastomotic_configuration',
                        'surgeon_experience', 'BIHistoryOfIschaemicHeartDisease',
                        'BIHistoryOfDiabetes', 'data_group_encoded']
    target = ['anastomotic_leackage']
    
    print('App Initialized correctly!')
    
    return input_data ,  model , selected_features ,  target , example_data

# Funcion to process user input
def parser_user_input(dataframe_input , model , selected_features , target , dictionary_categorical_features):

       
    # Encode categorical features
    for i in dictionary_categorical_features.keys():
        if i not in ['approach' , 'surgeon_experience' , 'anastomotic_configuration']:
            dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])
    # Add configuration value
    probabilities = {}
    values_of_anastomotic_configuration = {'End to End' : 1,
                                           'Side to End' : 2,
                                           'Side to Side' : 3,
                                           'End to Side' : 4}
    values_of_surgeon_experience = {"Consultant" : 1,
                            "Teaching operation" : 2}
    values_of_approach = {'1: Laparoscopic' : 1 ,
                          '2: Robotic' : 2 ,
                          '3: Open to open' : 3,
                          '4: Conversion to open' : 4,
                          '5: Conversion to laparoscopy' : 5}
    posible_values = list(product(values_of_surgeon_experience , values_of_approach , values_of_anastomotic_configuration))
    min_probability = {-1 : 2.0}
    max_probability = {-1 : -1.0}
    
    for i in range(len(posible_values)):
        print('#' * 50)
        print('Making prediction usung surgeon experience = ' , posible_values[i][0] , ', Approach =' , posible_values[i][1] , 'and Anastomotic Configuration =' , posible_values[i][2])
        dataframe_input['surgeon_experience'] = dictionary_categorical_features['surgeon_experience'][posible_values[i][0]]
        dataframe_input['approach'] = values_of_approach[posible_values[i][1]]
        dataframe_input['anastomotic_configuration'] = values_of_anastomotic_configuration[posible_values[i][2]]
        dataframe_input = dataframe_input[selected_features]
        prediction = model.predict_proba(dataframe_input)[ : , 1] # Probability for class 1
        probabilities[i] = pd.DataFrame(prediction.copy() , columns = ['Probabilities']).T
        probabilities[i] = probabilities[i][0]
        # Update max and min value
        if probabilities[i].values[0] < min_probability[list(min_probability.keys())[-1]]:
            min_probability[i] = probabilities[i].values[0]
        if probabilities[i].values[0] > max_probability[list(max_probability.keys())[-1]]:
            max_probability[i] = probabilities[i].values[0]
    aux_df = pd.DataFrame()
    for i in range(len(probabilities)):
        aux = pd.DataFrame({'Surgeon Experience' : [posible_values[i][0]],
                            'Approach' : [posible_values[i][1]],
                            'Anastomotic Configuration' : [posible_values[i][2]],
                           'Probability' : [probabilities[i].values[0]]})
        aux_df = pd.concat([aux_df,
                            aux] , axis = 0)
    pivot_df = pd.pivot_table(aux_df,
                              values = ['Probability'],
                              index = ['Surgeon Experience' , 'Approach' , 'Anastomotic Configuration'],
                              aggfunc = 'sum').sort_values(by = 'Probability' , ascending = True)
    
    mean_value = pivot_df['Probability'].values.mean()       
        
    # Format of prediction
    formatted_df = pivot_df.style.format({"Probability": "{:.6f}".format})
    # Create message to show the best option
    best_option = f"With **Surgeon Experience:** {posible_values[list(min_probability.keys())[-1]][0]}, **Approach:** {posible_values[list(min_probability.keys())[-1]][1]}, **Anastomotic Configuration:** {posible_values[list(min_probability.keys())[-1]][2]}, the likelihood of anastomotic leakage is the lowest with a value of:{min_probability[list(min_probability.keys())[-1]] * 100 : .6f}%." 
    difference_message = f"This is a mean reduction of {(max_probability[list(max_probability.keys())[-1]] - mean_value) * 100 : .6f}% with respect to other options."
    st.write(best_option)
    st.write(difference_message)
    st.write('Results for all options:')
    st.write(formatted_df)

    return None

def parser_user_input_2(dataframe_input , model , selected_features , target , dictionary_categorical_features , threshold = 0.3):

       
    # Encode categorical features
    for i in dictionary_categorical_features.keys():
        if i not in ['surgeon_experience']:
            dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])
    # Add configuration value
    probabilities = {}
    values_of_anastomotic_configuration = {'End to End' : 1,
                                           'Side to End' : 2,
                                           'Side to Side' : 3,
                                           'End to Side' : 4}
    values_of_surgeon_experience = {"Consultant" : 1,
                            "Teaching operation" : 2}
    values_of_approach = {'1: Laparoscopic' : 1 ,
                          '2: Robotic' : 2 ,
                          '3: Open to open' : 3,
                          '4: Conversion to open' : 4,
                          '5: Conversion to laparoscopy' : 5}
    posible_values = list(product(values_of_surgeon_experience))
    min_probability = {-1 : 2.0}
    max_probability = {-1 : -1.0}
    
    for i in range(len(posible_values)):
        print('#' * 50)
        print('Making prediction usung surgeon experience = ' , posible_values[i][0])
        dataframe_input['surgeon_experience'] = dictionary_categorical_features['surgeon_experience'][posible_values[i][0]]
        dataframe_input = dataframe_input[selected_features]
        prediction = model.predict_proba(dataframe_input)[ : , 1] # Probability for class 1
        probabilities[i] = pd.DataFrame(prediction.copy() , columns = ['Probabilities']).T
        probabilities[i] = probabilities[i][0]
        # Update max and min value
        if probabilities[i].values[0] < min_probability[list(min_probability.keys())[-1]]:
            min_probability[i] = probabilities[i].values[0]
        if probabilities[i].values[0] > max_probability[list(max_probability.keys())[-1]]:
            max_probability[i] = probabilities[i].values[0]
    aux_df = pd.DataFrame()
    for i in range(len(probabilities)):
        aux = pd.DataFrame({'Surgeon Experience' : [posible_values[i][0]],
                           'Probability' : [probabilities[i].values[0]]})
        aux_df = pd.concat([aux_df,
                            aux] , axis = 0)
    pivot_df = pd.pivot_table(aux_df,
                              values = ['Probability'],
                              index = ['Surgeon Experience' ],
                              aggfunc = 'sum').sort_values(by = 'Probability' , ascending = True)
    
    mean_value = pivot_df['Probability'].values.mean()       
        
    # Format of prediction
    formatted_df = pivot_df.style.format({"Probability": "{:.6f}".format})
    # Create message to show the best option
    best_option = f"With **Surgeon Experience:** {posible_values[list(min_probability.keys())[-1]][0]}, the likelihood of anastomotic leakage is the lowest with a value of:{min_probability[list(min_probability.keys())[-1]] * 100 : .6f}%." 
    difference_message = f"This is a mean reduction of {(max_probability[list(max_probability.keys())[-1]] - mean_value) * 100 : .6f}% with respect to other options."
    st.write(best_option)
    st.write(difference_message)
    st.write('Results for all options:')
    st.write(formatted_df)
    
    # Check for the probability using a experienced consultan and compare with the threshold
    probability_consultant = pivot_df.reset_index()
    probability_consultant = probability_consultant[probability_consultant['Surgeon Experience'] == 'Consultant']['Probability'].values[0]
    
    probability_teaching = pivot_df.reset_index()
    probability_teaching = probability_teaching[probability_teaching['Surgeon Experience'] == 'Teaching operation']['Probability'].values[0]
    
    if probability_consultant <= threshold and probability_teaching <= threshold:
        message_surgeon = f"The probability using a Teaching Operation **({probability_teaching})** is under the stablished threshold **({threshold})**. Model suggests a low risk using this type of surgeon in comparison with a experienced consultant"
    elif probability_consultant <= threshold and probability_teaching > threshold:
        message_surgeon = f"The probability using a Teaching Operation **({probability_teaching})** is above the stablished threshold **({threshold})**. Model suggests a high risk using this type of surgeon in comparison with a experienced consultant"
    else:
        message_surgeon = f"Patient's operation with a high risk of Anastomotic Leakage, check for input parameters"
    
    st.write(message_surgeon)
    return None



###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
input_data , model , selected_features , target , example_data = initialize_app()

# Define threshold is using for probabilities
THRESHOLD = 0.425

# Define dictionary for the categorical features
dictionary_categorical_features = {'sex' : {'Male' : 2,
                                            'Female' : 1},
                                   'active_smoking' : {'Yes' : 1,
                                                       'No' : 0},
                                   'asa_score' : {'1: Healthy Person' : 1,
                                            '2: Mild Systemic disease' : 2,
                                            '3: Severe syatemic disease' : 3,
                                            '4: Severe systemic disease that is a constan threat to life' : 4,
                                            '5: Moribund person' : 5,
                                            '6: Unkonw' : 6},
                                   'prior_abdominal_surgery' : {'Yes' : 1,
                                                                'No' : 2},  
                                   'indication' : {"Recurrent Diverticulitis" : 1,
                                                   "Acute Diverticulitis" : 2,
                                                   "Ileus/Stenosis" : 3,
                                                   "Ischemia" : 4,
                                                   "Tumor" : 5,
                                                   "Volvulus" : 6,
                                                   "Morbus crohn" : 7,
                                                   "Colitis ulcerosa" : 8,
                                                   "Perforation" : 9,
                                                   "Other" : 10,
                                                   "Ileostoma reversal" : 11,
                                                   "Colostoma reversal" : 12},
                                   'operation' : {"Rectosigmoid resection/sigmoidectomy" : 1, 
                                                  "Left hemicolectomy" : 2, 
                                                  "Extended left hemicolectomy" : 3, 
                                                  "Right hemicolectomy" : 4, 
                                                  "Extended right hemicolectomy" : 5, 
                                                  "Transverse colectomy" : 6, 
                                                  "Hartmann conversion" : 7, 
                                                  "Ileocaecal resection" : 8, 
                                                  "Total colectomy" : 9, 
                                                  "High anterior resection (anastomosis higher than 12cm)" : 10, 
                                                  "Low anterior resection (anastomosis 12 cm from anal average and below)" : 11, 
                                                  "Abdominoperineal resection" : 12, 
                                                  "Adhesiolysis with small bowel resection" : 13, 
                                                  "Adhesiolysis only" : 14, 
                                                  "Hartmann resection / Colostomy" : 15, 
                                                  "Colon segment resection" : 16, 
                                                  "Small bowel resection" : 17},
                                   'emergency_surgery' : {'Yes' : 1,
                                                          'No' : 0},
                                   'approach' : {'1: Laparoscopic' : 1 ,
                                                 '2: Robotic' : 2 ,
                                                 '3: Open to open' : 3,
                                                 '4: Conversion to open' : 4,
                                                 '5: Conversion to laparoscopy' : 5,
                                                 '6: Transanal' : 6},
                                   'type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis' : {"Colon anastomosis" : 1,
                                                                                                              "Colorectal anastomosis" : 2,
                                                                                                              "Ileocolonic anastomosis" : 3,
                                                                                                              "Ileorectal anastomosis" : 4,
                                                                                                              "Ileopouch-anal" : 5,
                                                                                                              "Colopouch" : 6,
                                                                                                              "Small intestinal anastomosis" : 7},
                                   'anastomotic_technique' : {'1: Stapler' : 1,
                                                              '2: Hand-sewn' : 2,
                                                              '3: Stapler and Hand-sewn' : 3},
                                   'surgeon_experience' : {"Consultant" : 1,
                                                           "Teaching operation" : 2},
                                   'BIHistoryOfIschaemicHeartDisease' : {'Yes' : 1,
                                                                               'No' : 0},
                                   'BIHistoryOfDiabetes' : {'Yes' : 1,
                                                               'No' : 0},
                                   'data_group_encoded' : {'clarunisclaraspita': 0,
                                                             'emmental_hospital': 1,
                                                             'gzo_wetzikon': 2,
                                                             'kantonspital_liest': 3,
                                                             'military_universit': 4,
                                                             'uk': 5,
                                                             'universitt_innsbru': 6,
                                                             'university_basel': 7,
                                                             'university_dalhous': 8,
                                                             'university_hamburg': 9,
                                                             'university_las_veg': 10,
                                                             'university_of_east': 11,
                                                             'university_vilnius': 12,
                                                             'university_wrzburg': 13},
                                   'anastomotic_configuration' : {"end-to-end" : 1,
                                                                  "side-to-end" : 2,
                                                                  "side-to-side" : 3,
                                                                  "end-to-side" : 4},
                                   'neoadjuvant_therapy' : {'Yes' : 1,
                                                            'No' : 0}}


# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction' , 'Examples' , 'Surgeon Experience Simulation'],
        icons = ['house' , 'book' , 'book' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
    
######################
# Home page layout
######################
if selected == 'Home':
    st.title('Anastomotic Leackage App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models iterate over all posible anastomotic configuration and surgeon experience for suggesting
    the best option.\n
    **Examples:** On this section you can use pre-defined values for a patient
    to see the predictions.\n
    **Surgeon Experience Simulation**: On this section you can select patients information and the model simulate
    the anastomotic leakage likelihood for an operation with and expert consultan or a teaching operation.
    """)
    
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    st.subheader("Description")
    st.subheader("To predict Anastomotic Leackage, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for each type of configuration and surgeon experience.
    """)
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    sex = st.sidebar.selectbox('Gender', tuple(dictionary_categorical_features['sex'].keys()))

    age = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    
    bmi = st.sidebar.slider("Preoperative BMI:", min_value = 18, max_value = 50,step = 1)
    
    active_smoking = st.sidebar.selectbox('Active Smoking',  tuple(dictionary_categorical_features['active_smoking'].keys()))
    
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    
    asa_score = st.sidebar.selectbox('ASA Score', tuple(dictionary_categorical_features['asa_score'].keys()))
    
    prior_abdominal_surgery = st.sidebar.selectbox('Prior Abdominal Surgery', tuple(dictionary_categorical_features['prior_abdominal_surgery'].keys()))
    
    indication = st.sidebar.selectbox('Indication', tuple(dictionary_categorical_features['indication'].keys()))
    
    operation = st.sidebar.selectbox('Operation', tuple(dictionary_categorical_features['operation'].keys()))
    
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery', tuple(dictionary_categorical_features['emergency_surgery'].keys()))
    
    type_of_anastomosis = st.sidebar.selectbox('Type of Anastomosis', tuple(dictionary_categorical_features['type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis'].keys()))
    
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique', tuple(dictionary_categorical_features['anastomotic_technique'].keys()))
    
    BIHistoryOfIschaemicHeartDisease = st.sidebar.selectbox('Ischaemic Heart Disease', tuple(dictionary_categorical_features['BIHistoryOfIschaemicHeartDisease'].keys()))
    
    BIHistoryOfDiabetes = st.sidebar.selectbox('Diabetes', tuple(dictionary_categorical_features['BIHistoryOfDiabetes'].keys()))
    
    #data_group_encoded = st.sidebar.selectbox('Clinic', tuple(dictionary_categorical_features['data_group_encoded'].keys()))
    
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', tuple(dictionary_categorical_features['neoadjuvant_therapy'].keys()))
    
    
    dataframe_input = pd.DataFrame({'sex' : [sex],
                                    'age' : [age],
                                    'bmi' : [bmi],
                                    'active_smoking' : [active_smoking],
                                    'preoperative_hemoglobin_level' : [preoperative_hemoglobin_level],
                                    'asa_score' : [asa_score],
                                    'prior_abdominal_surgery' : [prior_abdominal_surgery],
                                    'indication' : [indication],
                                    'operation' : [operation],
                                    'emergency_surgery' : [emergency_surgery],
                                    'type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis' : [type_of_anastomosis],
                                    'anastomotic_technique' : [anastomotic_technique],
                                    'BIHistoryOfIschaemicHeartDisease' : [BIHistoryOfIschaemicHeartDisease],
                                    'BIHistoryOfDiabetes' : [BIHistoryOfDiabetes],
                                    'data_group_encoded' : ['gzo_wetzikon'],
                                    'neoadjuvant_therapy' : [neoadjuvant_therapy]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_user_input(dataframe_input ,  model , selected_features , target , dictionary_categorical_features)
        #st.dataframe(predictions)
        
##############################################################################
# Example page layout
# Prediction page layout
if selected == 'Examples':
    # Create inverted dictionary
    inverted_dictionary = {}

    for key, value in dictionary_categorical_features.items():
        inverted_dictionary[key] = {v: k for k, v in value.items()}
    
    st.title('Examples Section')
    st.subheader("Description")
    st.subheader("This Section Show different examples for prediction, follow the instructions:")
    st.markdown("""
    1. Select the example row you want to use.
    2. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for each type of configuration and surgeon experience.
    """)
    
    patient_row = st.slider("Patient's row:", min_value = 0, max_value = example_data.shape[0] - 1,step = 1)
    aux_patient = pd.DataFrame(example_data.iloc[patient_row]).T
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Parameters are setting based on example you choose")
    
    # Input features
    
    sex = st.sidebar.selectbox('Gender',
                               tuple(dictionary_categorical_features['sex'].keys()),
                               index = tuple(dictionary_categorical_features['sex'].keys()).index(inverted_dictionary['sex'][aux_patient['sex'].values[0]]))
    
    age = st.sidebar.slider("Age:", 
                            min_value = 18, 
                            max_value = 100,
                            step = 1 , 
                            value = aux_patient['age'].values[0])
    
    bmi = st.sidebar.slider("Preoperative BMI:",
                            min_value = 18.0,
                            max_value = 50.0,
                            step = 0.1,
                            value = float(aux_patient['bmi'].values[0]))
    
    active_smoking = st.sidebar.selectbox('Active Smoking',
                                          tuple(dictionary_categorical_features['active_smoking'].keys()),
                                          index = tuple(dictionary_categorical_features['active_smoking'].keys()).index(inverted_dictionary['active_smoking'][aux_patient['active_smoking'].values[0]]))
    
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:",
                                                      min_value = 0.0,
                                                      max_value = 30.0,
                                                      step = 0.1,
                                                      value = float(aux_patient['preoperative_hemoglobin_level'].values[0]))
    asa_score = st.sidebar.selectbox('ASA Score',
                                    tuple(dictionary_categorical_features['asa_score'].keys()),
                                    index = tuple(dictionary_categorical_features['asa_score'].keys()).index(inverted_dictionary['asa_score'][aux_patient['asa_score'].values[0]]))
    
    prior_abdominal_surgery = st.sidebar.selectbox('Prior Abdominal Surgery',
                                    tuple(dictionary_categorical_features['prior_abdominal_surgery'].keys()),
                                    index = tuple(dictionary_categorical_features['prior_abdominal_surgery'].keys()).index(inverted_dictionary['prior_abdominal_surgery'][aux_patient['prior_abdominal_surgery'].values[0]]))
    
    indication = st.sidebar.selectbox('Indication',
                                    tuple(dictionary_categorical_features['indication'].keys()),
                                    index = tuple(dictionary_categorical_features['indication'].keys()).index(inverted_dictionary['indication'][aux_patient['indication'].values[0]]))
    
    operation = st.sidebar.selectbox('Operation',
                                    tuple(dictionary_categorical_features['operation'].keys()),
                                    index = tuple(dictionary_categorical_features['operation'].keys()).index(inverted_dictionary['operation'][aux_patient['operation'].values[0]]))
    
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery',
                                    tuple(dictionary_categorical_features['emergency_surgery'].keys()),
                                    index = tuple(dictionary_categorical_features['emergency_surgery'].keys()).index(inverted_dictionary['emergency_surgery'][aux_patient['emergency_surgery'].values[0]]))
 
    type_of_anastomosis = st.sidebar.selectbox('Type of Anastomosis',
                                    tuple(dictionary_categorical_features['type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis'].keys()),
                                    index = tuple(dictionary_categorical_features['type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis'].keys()).index(inverted_dictionary['type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis'][aux_patient['type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis'].values[0]]))
    
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique',
                                    tuple(dictionary_categorical_features['anastomotic_technique'].keys()),
                                    index = tuple(dictionary_categorical_features['anastomotic_technique'].keys()).index(inverted_dictionary['anastomotic_technique'][aux_patient['anastomotic_technique'].values[0]]))
    
    BIHistoryOfIschaemicHeartDisease = st.sidebar.selectbox('Ischaemic Heart Disease',
                                    tuple(dictionary_categorical_features['BIHistoryOfIschaemicHeartDisease'].keys()),
                                    index = tuple(dictionary_categorical_features['BIHistoryOfIschaemicHeartDisease'].keys()).index(inverted_dictionary['BIHistoryOfIschaemicHeartDisease'][aux_patient['BIHistoryOfIschaemicHeartDisease'].values[0]]))
    
    BIHistoryOfDiabetes = st.sidebar.selectbox('Diabetes',
                                    tuple(dictionary_categorical_features['BIHistoryOfDiabetes'].keys()),
                                    index = tuple(dictionary_categorical_features['BIHistoryOfDiabetes'].keys()).index(inverted_dictionary['BIHistoryOfDiabetes'][aux_patient['BIHistoryOfDiabetes'].values[0]]))
    
    #data_group_encoded = st.sidebar.selectbox('Clinic',
    #                                tuple(dictionary_categorical_features['data_group_encoded'].keys()),
    #                                index = list(dictionary_categorical_features['data_group_encoded'].keys()).index(aux_patient['data_group'].values[0]))
    
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy',
                                               tuple(dictionary_categorical_features['neoadjuvant_therapy'].keys()),
                                               index = tuple(dictionary_categorical_features['neoadjuvant_therapy'].keys()).index(inverted_dictionary['neoadjuvant_therapy'][aux_patient['neoadjuvant_therapy'].values[0]]))
    
    dataframe_input = pd.DataFrame({'sex' : [sex],
                                    'age' : [age],
                                    'bmi' : [bmi],
                                    'active_smoking' : [active_smoking],
                                    'preoperative_hemoglobin_level' : [preoperative_hemoglobin_level],
                                    'asa_score' : [asa_score],
                                    'prior_abdominal_surgery' : [prior_abdominal_surgery],
                                    'indication' : [indication],
                                    'operation' : [operation],
                                    'emergency_surgery' : [emergency_surgery],
                                    'type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis' : [type_of_anastomosis],
                                    'anastomotic_technique' : [anastomotic_technique],
                                    'BIHistoryOfIschaemicHeartDisease' : [BIHistoryOfIschaemicHeartDisease],
                                    'BIHistoryOfDiabetes' : [BIHistoryOfDiabetes],
                                    'data_group_encoded' : ['gzo_wetzikon'],
                                    'neoadjuvant_therapy' : [neoadjuvant_therapy]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_user_input(dataframe_input ,  model , selected_features , target , dictionary_categorical_features)
        #st.dataframe(predictions)
##############################################################################
# Surgeon Experience SImulation layout
if selected ==  'Surgeon Experience Simulation':
    st.title('Surgeon Experience Simulation Section')
    st.subheader("Description")
    st.subheader("To predict Anastomotic Leackage, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for each type of surgeon experience.
    """)
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    sex = st.sidebar.selectbox('Gender', tuple(dictionary_categorical_features['sex'].keys()))

    age = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    
    bmi = st.sidebar.slider("Preoperative BMI:", min_value = 18, max_value = 50,step = 1)
    
    active_smoking = st.sidebar.selectbox('Active Smoking',  tuple(dictionary_categorical_features['active_smoking'].keys()))
    
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    
    asa_score = st.sidebar.selectbox('ASA Score', tuple(dictionary_categorical_features['asa_score'].keys()))
    
    prior_abdominal_surgery = st.sidebar.selectbox('Prior Abdominal Surgery', tuple(dictionary_categorical_features['prior_abdominal_surgery'].keys()))
    
    indication = st.sidebar.selectbox('Indication', tuple(dictionary_categorical_features['indication'].keys()))
    
    operation = st.sidebar.selectbox('Operation', tuple(dictionary_categorical_features['operation'].keys()))
    
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery', tuple(dictionary_categorical_features['emergency_surgery'].keys()))
    
    type_of_anastomosis = st.sidebar.selectbox('Type of Anastomosis', tuple(dictionary_categorical_features['type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis'].keys()))
    
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique', tuple(dictionary_categorical_features['anastomotic_technique'].keys()))
    
    BIHistoryOfIschaemicHeartDisease = st.sidebar.selectbox('Ischaemic Heart Disease', tuple(dictionary_categorical_features['BIHistoryOfIschaemicHeartDisease'].keys()))
    
    BIHistoryOfDiabetes = st.sidebar.selectbox('Diabetes', tuple(dictionary_categorical_features['BIHistoryOfDiabetes'].keys()))
    
    #data_group_encoded = st.sidebar.selectbox('Clinic', tuple(dictionary_categorical_features['data_group_encoded'].keys()))
    
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', tuple(dictionary_categorical_features['neoadjuvant_therapy'].keys()))
    
    approach = st.sidebar.selectbox('Approach', tuple(dictionary_categorical_features['approach'].keys()))
    
    anastomotic_configuration = st.sidebar.selectbox('Configuration', tuple(dictionary_categorical_features['anastomotic_configuration'].keys()))
    
    dataframe_input = pd.DataFrame({'sex' : [sex],
                                    'age' : [age],
                                    'bmi' : [bmi],
                                    'active_smoking' : [active_smoking],
                                    'preoperative_hemoglobin_level' : [preoperative_hemoglobin_level],
                                    'asa_score' : [asa_score],
                                    'prior_abdominal_surgery' : [prior_abdominal_surgery],
                                    'indication' : [indication],
                                    'operation' : [operation],
                                    'emergency_surgery' : [emergency_surgery],
                                    'type_of_anastomosis -> das von UK sind alles  Ileocolonic anastomosis' : [type_of_anastomosis],
                                    'anastomotic_technique' : [anastomotic_technique],
                                    'BIHistoryOfIschaemicHeartDisease' : [BIHistoryOfIschaemicHeartDisease],
                                    'BIHistoryOfDiabetes' : [BIHistoryOfDiabetes],
                                    'data_group_encoded' : ['gzo_wetzikon'],
                                    'neoadjuvant_therapy' : [neoadjuvant_therapy],
                                    'approach' : [approach],
                                    'anastomotic_configuration' : [anastomotic_configuration]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_user_input_2(dataframe_input ,  model , selected_features , target , dictionary_categorical_features)
        #st.dataframe(predictions)
        
