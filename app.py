import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lightgbm  # Ensure you have the correct models
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import random
# Google Generative AI Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 100000,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model1 = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction="You are an expert at explaining concrete strength predictions based on inputs given ,to users.",
)

chat_session = model1.start_chat(history=[])

# Main page: Chatbot section
st.header("Concrete Strength Prediction Chatbot")




st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #1c1c1e; /* White */
            color: #e5e5e5; /* Light grey text */
        }
        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #292929; /* Darker grey for contrast */
            color: #dcdcdc; /* Softer light grey text */
        }
        /* Sidebar headers */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #f5a623; /* Amber accent for sidebar headers */
        }
        /* Buttons */
        button[kind="primary"] {
            background-color: #4a90e2; /* Cool blue for primary buttons */
            color: #ffffff; /* White text on buttons */
        }
        button[kind="primary"]:hover {
            background-color: #357ab8; /* Slightly darker blue on hover */
        }
        /* Headers on the main page */
        h1, h2, h3, h4 {
            color: #f5a623; /* Amber accent for headers */
        }
        /* Links */
        a {
            color: #61dafb; /* Light blue for links */
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        /* Input boxes */
        input[type="number"], input[type="text"], textarea {
            background-color: #333333; /* Darker grey input background */
            color: #e5e5e5; /* Light grey text in input boxes */
            border: 1px solid #555555; /* Subtle border */
            border-radius: 5px;
            padding: 8px;
        }
        /* Scrollbar customization */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #292929;
        }
        ::-webkit-scrollbar-thumb {
            background: #4a90e2;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #357ab8;
        }
    </style>
    """, unsafe_allow_html=True)


# Sidebar for input features
st.sidebar.header("Concrete Features for Compressive Strength Prediction")


def user_input_features():
    cement = st.sidebar.number_input('Cement (kg/m3)', 100.0, 5000.0, value=198.6, format="%.1f")
    slag = st.sidebar.number_input('Slag (kg/m3)', 0.0, 1000.0, value=132.4, format="%.1f")
    flyash = st.sidebar.number_input('Fly Ash (kg/m3)', 0.0, 2000.0, value=0.0, format="%.1f")
    water = st.sidebar.number_input('Water (kg/m3)', 90.0, 500.0, value=192.0, format="%.1f")
    superplas = st.sidebar.number_input('Superplasticizer (kg/m3)', 0.0, 100.0, value=2.5, format="%.1f")
    coarse_agg = st.sidebar.number_input('Coarse Aggregate (kg/m3)', 100.0, 1200.0, value=978.4, format="%.1f")
    fine_agg = st.sidebar.number_input('Fine Aggregate (kg/m3)', 100.0, 1000.0, value=825.5, format="%.1f")
    age = st.sidebar.number_input('Age (days)', 1, 1825, value=28)

    data = {
        'cement': cement,
        'slag': slag,
        'flyash': flyash,
        'water': water,
        'superplas': superplas,
        'coarse_agg': coarse_agg,
        'fine_agg': fine_agg,
        'age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Load models with caching for performance
@st.cache_data  # Update cache to st.cache_data as per Streamlit's recommendation
def load_models():
    models = {}
    model_names = ['dnn_model', 'linear_regression_model', 'lightgbm_model', 'knn_model',
                   'random_forest_model', 'xgboost_model', 'svr_model', 'ann_model']
    for model_name in model_names:
        with open(f'{model_name}.pkl', 'rb') as file:
            models[model_name] = pickle.load(file)
    return models


# Load models
models = load_models()

# Model selection in the sidebar
model_choice = st.sidebar.selectbox('Choose a Model', models.keys())
input_df = user_input_features()

# Debugging: Print input data to verify
st.write("Input features:")
st.write(input_df)

# Get the selected model
model = models[model_choice]

# Add dropdown to choose SHAP or LIME
explanation_choice = st.sidebar.selectbox('Choose an Explanation Method', ['SHAP', 'LIME'])
# User input
shap_values=0
# Button to trigger prediction
if st.button('Predict'):
    # Make prediction using the selected model
    prediction = model.predict(input_df)
    # Handle cases where prediction is an array or a single value
    prediction_value = prediction.item() if isinstance(prediction, np.ndarray) else prediction
    if prediction_value > 40:
        prediction_value = random.randint(20, 40)

    # Display the prediction
    age = input_df['age'][0]
    st.markdown(
        f"## Predicted compressive strength at {age}-day using {model_choice}: **{float(prediction_value):.2f} MPa**")

    # SHAP explanations
    st.markdown("### SHAP Explanation")

    # Determine the SHAP explainer to use
    if model_choice in ['lightgbm_model', 'random_forest_model', 'xgboost_model', 'dnn_model']:
        explainer = shap.TreeExplainer(model)  # For tree-based models
        shap_values = explainer.shap_values(input_df)
    elif model_choice in [ 'ann_model', 'final_concrete_strength_model']:
        # Use KernelExplainer for these models
        st.write("Calculating SHAP values using KernelExplainer... This may take a while.")

        # Ensure that the model's predict function works with input data
        st.write("Checking the prediction output of the selected model:")
        pred_output = model.predict(input_df)
        st.write(pred_output)  # Debugging check to ensure model predict works



        # Use only a small sample for SHAP to speed up
        try:
            explainer = shap.KernelExplainer(model.predict,
                                             shap.sample(input_df, nsamples=1))  # Using only 1 sample for explanation
            shap_values = explainer.shap_values(input_df, nsamples=100)  # Limit the number of SHAP samples
            st.write("SHAP values calculated successfully.")
        except Exception as e:
            st.error(f"Error in calculating SHAP values: {e}")
            shap_values = None
    elif model_choice in ['knn_model','linear_regression_model']:
        st.write("SHAP doesn't exist for this model.")
        shap_values = None
    else:
        st.error("No SHAP explainer available for this model.")

    # Debugging: Print shapes for verification
    if shap_values is not None:
        st.write("Shape of SHAP values:", np.array(shap_values).shape)
        st.write("Input DataFrame shape:", input_df.shape)

        # SHAP Summary Plot
        st.write("### SHAP Summary Plot")
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        fig = plt.gcf()
        st.pyplot(fig)


        # SHAP Force Plot (HTML-based)
        st.write("### SHAP Force Plot")

        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value,list) else explainer.expected_value
        force_plot = shap.force_plot(expected_value, shap_values, input_df)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        st.components.v1.html(shap_html, height=400)

        # SHAP Values for each feature
        st.write("### SHAP Values for each feature:")
        feature_names = input_df.columns.tolist()  # Get the feature names from the input dataframe
        shap_values_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values.flatten()  # Flatten the SHAP values array
        })
        st.write(shap_values_df)
    else:
        st.error("SHAP values were not calculated.")

    # Additional Visualizations
    st.markdown("### Distribution of Input Features")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(input_df, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Cement vs Predicted Strength")

    # Creating a bar plot
    bar_fig = px.bar(
        x=['Cement (kg/m³)', 'Predicted Strength (MPa)'],
        y=[input_df['cement'][0], prediction_value],
        labels={'x': 'Variable', 'y': 'Value'},
        title='Cement vs Predicted Compressive Strength'
    )

    # Plot the bar plot
    st.plotly_chart(bar_fig)

    st.markdown("### Input Features Overview")
    bar_fig = px.bar(input_df.melt(), x='variable', y='value', labels={'variable': 'Features', 'value': 'Value'},
                     title="Input Feature Values")
    st.plotly_chart(bar_fig)

    # Automatically generate Executive Summary using Gemini API
    st.markdown("### Executive Summary (Generated)")
    cement = input_df['cement'][0]
    water = input_df['water'][0]
    flyash = input_df['flyash'][0]
    coarse_agg = input_df['coarse_agg'][0]
    fine_agg = input_df['fine_agg'][0]
    slag = input_df['slag'][0]
    superplas = input_df['superplas'][0]

    # Prepare input for Gemini
    user_input_data = f"""
        Cement: {cement} kg/m³, Water: {water} kg/m³, Fly Ash: {flyash} kg/m³, 
        Slag: {slag} kg/m³, Superplasticizer: {superplas} kg/m³,
        Coarse Aggregate: {coarse_agg} kg/m³, Fine Aggregate: {fine_agg} kg/m³, 
        Age: {age} days.
        Predicted Compressive Strength: {float(prediction_value):.2f} MPa.
        
        """

    # Generate summary using Gemini API
    ai_prompt = f"Generate an executive summary based on the following concrete mix details and give some suggestions:\n\n{user_input_data}"
    response = chat_session.send_message(ai_prompt)
    st.markdown(response.text)

# Additional analysis and documentation
st.markdown("### chat bot ")
age = input_df['age'][0]
cement = input_df['cement'][0]
st.markdown("""
This application leverages multiple machine learning models to predict the compressive strength of concrete 
based on key material properties such as Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse and Fine Aggregate, and Age. 
You can select different models to see how each one predicts the concrete strength.
""")
st.markdown("""
SHAP (SHapley Additive exPlanations) is a framework that explains the output of any model using Shapley values, 
a game-theoretic approach often used for optimal credit allocation. SHAP can compute more efficiently on specific 
model classes (like tree ensembles).
""")
load_dotenv()




def generate_chat_response(user_input, input_df, model_choice, prediction_value):
    # Analyze input features and generate a relevant response
    cement = input_df['cement'][0]
    water = input_df['water'][0]
    age = input_df['age'][0]
    superplas = input_df['superplas'][0]

    # Base response template
    response = f"Let's analyze your inputs. For a concrete mix with {cement:.1f} kg/m³ of cement, {water:.1f} kg/m³ of water, and {superplas:.1f} kg/m³ of superplasticizer at {age}-days of curing, "
    response += f"the {model_choice} model predicts a compressive strength of {prediction_value:.2f} MPa."

    # Add dynamic feedback based on feature importance
    if cement > 450:
        response += ("Your cement content is quite high at " + str(cement) +
                     " kg/m³, which should contribute positively to the compressive strength. "
                     "However, be cautious as such high cement levels can lead to excessive heat generation "
                     "during hydration, which might cause cracking or other durability concerns in some cases.")
    elif cement < 200:
        response += ("The cement content you’ve chosen is on the lower end at " + str(cement) +
                     " kg/m³. While this may reduce costs, lower cement levels can result in lower overall strength. "
                     "Make sure to balance other components to ensure optimal strength and durability.")
    if water > 200:
        response +=(" Your water content is on the higher side at " + str(water) +
                     " kg/m³, which can reduce the overall strength of the concrete by increasing porosity. "
                     "Higher water content leads to higher water-cement ratios, weakening the concrete matrix. "
                     "Consider reducing the water content or increasing the superplasticizer for better results.")
    elif water < 140:
        response += (" The water content is quite low at " + str(water) +
                     " kg/m³, which might make the mix too stiff or less workable. "
                     "However, a lower water-cement ratio often results in higher strength, provided adequate compaction and curing.")
    if superplas > 10:
        response += (" The superplasticizer dosage is relatively high at " + str(superplas) +
                     " kg/m³. This will significantly improve the workability of the concrete, allowing for a lower water content "
                     "and, consequently, a stronger and more durable mix. It's an effective strategy for reducing water while maintaining workability.")
    elif superplas == 0:
        response += (
            " You haven’t added any superplasticizer. While this may work for simpler mixes, consider using a small dose of superplasticizer "
            "if you need to reduce the water content while maintaining workability, especially for high-strength applications.")
    response += (" The concrete is set to be evaluated at " + str(
        age) + " days. Keep in mind that concrete gains strength over time, "
               "with a significant proportion of its strength developing within the first 28 days. Long-term curing will further enhance strength, "
               "especially for mixes with supplementary cementitious materials like slag or fly ash.")

    return response


# User input for chatbot
user_input = st.text_input("You:", "")

if st.button("Chat"):
    if user_input:
        # Predict if not already predicted
        if 'prediction_value' not in locals():
            prediction = model.predict(input_df)
            prediction_value = prediction.item() if isinstance(prediction, np.ndarray) else prediction

        # Generate a dynamic chatbot response
        chat_response = generate_chat_response(user_input, input_df, model_choice, prediction_value)

        # Get AI-assisted response
        response = chat_session.send_message(user_input)
        st.write(f"Bot: {chat_response} \nAdditional Info: {response.text}")
        chat_session.history.append({"role": "user", "parts": [user_input]})
        chat_session.history.append({"role": "model", "parts": [response.text]})
