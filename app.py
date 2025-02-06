import joblib
import pandas as pd
import streamlit as st
from streamlit_extras.colored_header import colored_header

# Configure page settings
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stNumberInput {width: 100%;}
    .st-bb {background-color: transparent;}
    .st-at {background-color: #f0f2f6;}
    .risk-label {font-size: 1.4rem !important; font-weight: 600 !important;}
    .probability-bar {background-color: #f0f2f6; border-radius: 20px; padding: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# Load models and transformers
@st.cache_resource
def load_artifacts():
    return (
        joblib.load('quantile_transformer.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('diabetes_model.pkl')
    )

qt, scaler, model = load_artifacts()

# Main app header
colored_header(
    label="Diabetes Risk Assessment Tool",
    description="Predict likelihood of diabetes based on health metrics",
    color_name="blue-70"
)

# Input section
with st.container():
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0,
                                    help="Number of times pregnant")
        glucose = st.number_input("Glucose Level", 50, 300, 117,
                                help="Plasma glucose concentration (mg/dL)")
        skin_thickness = st.number_input("Skin Thickness", 0, 99, 23,
                                       help="Triceps skin fold thickness (mm)")
        
    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 23.3,
                            help="Body Mass Index (kg/m²)")
        age = st.number_input("Age", 21, 100, 29,
                            help="Age in years")
        insulin = st.number_input("Insulin Level", 0, 500, 102,
                                help="2-Hour serum insulin (mu U/ml)")

# Prediction logic
def predict_diabetes_risk(input_data):
    try:
        # Feature transformation
        input_data['Insulin_quantile'] = qt.transform(input_data[['Insulin']])
        scaled_features = scaler.transform(input_data[['Glucose', 'SkinThickness', 'BMI', 'Age']])
        input_data[['Glucose', 'SkinThickness', 'BMI', 'Age']] = scaled_features
        input_data.drop('Insulin', axis=1, inplace=True)
        
        # Make prediction
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        return {
            'prediction': prediction[0],
            'probability': probabilities[0][1] * 100,
            'features': input_data
        }
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

# Results display
def display_results(prediction_data):
    st.subheader("Assessment Results")
    
    # Probability visualization
    with st.container():
        st.markdown(f"""
        <div class="probability-bar">
            <div style="display: flex; justify-content: space-between;">
                <span>Diabetes Probability:</span>
                <strong>{prediction_data['probability']:.1f}%</strong>
            </div>
            <progress value="{prediction_data['probability']}" max="100" 
                     style="width: 100%; height: 15px; border-radius: 10px;"></progress>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk categorization
    risk_level, color = ("High Risk 🚨", "#ff4b4b") if prediction_data['prediction'] == 1 else \
                       ("Low Risk ✅", "#2ecc71") if prediction_data['probability'] < 30 else \
                       ("Moderate Risk ⚠️", "#f1c40f")
    
    st.markdown(f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 10px; 
                text-align: center; margin: 20px 0;">
        <h3 class="risk-label">{risk_level}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance explanation
    with st.expander("Understand Key Factors"):
        st.write("""
        **Main contributing factors:**
        - Glucose levels (most significant)
        - BMI and Age
        - Insulin resistance
        - Genetic predisposition
        """)
        st.write("Lower risk factors observed in:")
        st.json({
            "Glucose": f"{prediction_data['features']['Glucose'].values[0]:.2f} (scaled)",
            "BMI": f"{prediction_data['features']['BMI'].values[0]:.2f} (scaled)"
        })

# Prediction trigger
if st.button("Analyze Diabetes Risk", use_container_width=True):
    input_df = pd.DataFrame([[
        pregnancies, glucose, skin_thickness, bmi, age, insulin
    ]], columns=['Pregnancies', 'Glucose', 'SkinThickness', 
                'BMI', 'Age', 'Insulin'])
    
    with st.spinner("Analyzing health metrics..."):
        result = predict_diabetes_risk(input_df)
    
    if result:
        display_results(result)

# Footer information
st.markdown("---")
st.markdown("""
**Clinical Notes:**
- This tool provides probabilistic estimates, not medical diagnoses
- Always consult with a healthcare professional
- Model accuracy: 89% (AUC: 0.95)
""")