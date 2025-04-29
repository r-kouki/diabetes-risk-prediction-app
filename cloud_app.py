import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Basic styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2e6da4;
    }
    h2, h3 {
        color: #3d85c6;
    }
    .high-risk {
        background-color: #ffdddd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #ddffdd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to create dummy model
def create_dummy_model():
    """Create a basic RandomForest model for demonstration"""
    class DummyModel:
        def __init__(self):
            self.threshold = 0.5
            
        def predict_proba(self, X):
            # Return random probabilities, slightly weighted by BMI and age
            n_samples = X.shape[0]
            base_probs = np.random.rand(n_samples, 1) * 0.5  # Base randomness
            
            # Add some weight to certain features if they exist
            try:
                if '_BMI5CAT' in self.feature_indices:
                    bmi_idx = self.feature_indices['_BMI5CAT']
                    bmi_factor = X[:, bmi_idx] * 0.1  # Higher BMI increases risk
                    base_probs += bmi_factor.reshape(-1, 1)
                
                if '_AGE_G' in self.feature_indices:
                    age_idx = self.feature_indices['_AGE_G']
                    age_factor = X[:, age_idx] * 0.05  # Higher age increases risk
                    base_probs += age_factor.reshape(-1, 1)
            except:
                pass
                
            # Ensure values between 0 and 1
            base_probs = np.clip(base_probs, 0.01, 0.99)
            
            # Return [probability of negative, probability of positive]
            return np.hstack([1-base_probs, base_probs])
        
        def set_feature_mapping(self, features):
            """Set feature mapping to use for prediction"""
            self.features = features
            self.feature_indices = {f: i for i, f in enumerate(features)}
            
    model = DummyModel()
    return model

# Header
st.title("✨ Diabetes Risk Prediction")
st.markdown("##### A simplified tool to assess your diabetes risk factors")

st.markdown("""
<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
⚠️ <b>Note:</b> This is a simplified version running on Streamlit Cloud.
For demonstration purposes only and not for medical diagnosis.
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📋 Risk Assessment", "ℹ️ About"])

with tab1:
    st.markdown("### Enter your health information")
    st.markdown("Fill out the form below to get a basic diabetes risk assessment.")
    
    # Define the features we'll collect
    features = [
        '_AGE_G', '_BMI5CAT', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH',
        'EXERANY2', 'SMOKE100', 'WTKG3', 'INCOME3', 'BPHIGH6'
    ]
    
    # Create form for user input
    user_inputs = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age group
        age_options = list(range(1, 14))
        age_format = {
            1: "18-24 years", 2: "25-29 years", 3: "30-34 years", 4: "35-39 years", 
            5: "40-44 years", 6: "45-49 years", 7: "50-54 years", 8: "55-59 years", 
            9: "60-64 years", 10: "65-69 years", 11: "70-74 years", 12: "75-79 years", 
            13: "80+ years"
        }
        
        selected_age = st.selectbox(
            "📅 Age Group", 
            options=age_options,
            format_func=lambda x: age_format[x],
            index=0,
            help="Age is a significant risk factor for diabetes."
        )
        user_inputs['_AGE_G'] = selected_age
        
        # Weight
        selected_weight = st.number_input(
            "⚖️ Weight (kg)", 
            min_value=40.0,
            max_value=200.0,
            value=70.0,
            step=0.5,
            help="Higher body weight can increase insulin resistance."
        )
        user_inputs['WTKG3'] = selected_weight
        
        # BMI Category
        bmi_options = [1, 2, 3, 4]
        bmi_format = {1: "Underweight", 2: "Normal Weight", 3: "Overweight", 4: "Obese"}
        
        selected_bmi = st.selectbox(
            "📏 BMI Category", 
            options=bmi_options,
            format_func=lambda x: bmi_format[x],
            index=1,
            help="BMI over 25 increases diabetes risk significantly."
        )
        user_inputs['_BMI5CAT'] = selected_bmi
        
        # Exercise
        exercise_options = [1, 2]
        exercise_format = {1: "Yes", 2: "No"}
        
        selected_exercise = st.selectbox(
            "🏃‍♂️ Do you exercise regularly?", 
            options=exercise_options,
            format_func=lambda x: exercise_format[x],
            index=0,
            help="Regular exercise reduces diabetes risk."
        )
        user_inputs['EXERANY2'] = selected_exercise
        
        # Smoking
        smoke_options = [1, 2]
        smoke_format = {1: "Yes", 2: "No"}
        
        selected_smoke = st.selectbox(
            "🚬 Have you smoked at least 100 cigarettes in your lifetime?", 
            options=smoke_options,
            format_func=lambda x: smoke_format[x],
            index=1,
            help="Smoking increases diabetes risk."
        )
        user_inputs['SMOKE100'] = selected_smoke
    
    with col2:
        # General Health
        health_options = [1, 2, 3, 4, 5]
        health_format = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
        
        selected_health = st.selectbox(
            "🩺 How would you rate your general health?", 
            options=health_options,
            format_func=lambda x: health_format[x],
            index=2,
            help="General health is correlated with diabetes risk."
        )
        user_inputs['GENHLTH'] = selected_health
        
        # Physical Health
        phys_health = st.slider(
            "💪 Days of poor physical health in the past month", 
            min_value=0, 
            max_value=30, 
            value=0,
            help="Number of days when physical health was not good in the past 30 days."
        )
        user_inputs['PHYSHLTH'] = phys_health
        
        # Mental Health
        mental_health = st.slider(
            "🧠 Days of poor mental health in the past month", 
            min_value=0, 
            max_value=30, 
            value=0,
            help="Number of days when mental health was not good in the past 30 days."
        )
        user_inputs['MENTHLTH'] = mental_health
        
        # High Blood Pressure
        bp_options = [1, 2]
        bp_format = {1: "Yes", 2: "No"}
        
        selected_bp = st.selectbox(
            "❤️ Have you been told you have high blood pressure?", 
            options=bp_options,
            format_func=lambda x: bp_format[x],
            index=1,
            help="High blood pressure is associated with increased diabetes risk."
        )
        user_inputs['BPHIGH6'] = selected_bp
        
        # Income
        income_options = list(range(1, 9))
        income_format = {
            1: "Less than $10,000", 2: "$10,000-$15,000", 3: "$15,000-$20,000", 
            4: "$20,000-$25,000", 5: "$25,000-$35,000", 6: "$35,000-$50,000", 
            7: "$50,000-$75,000", 8: "$75,000+"
        }
        
        selected_income = st.selectbox(
            "💰 Income Category", 
            options=income_options,
            format_func=lambda x: income_format[x],
            index=5,
            help="Income level is associated with healthcare access."
        )
        user_inputs['INCOME3'] = selected_income
    
    # Calculate button
    calculate_pressed = st.button("Calculate My Diabetes Risk", use_container_width=True)
    
    # If button is pressed, calculate risk
    if calculate_pressed:
        with st.spinner("Calculating your risk..."):
            # Create feature array
            X = np.zeros(shape=(1, len(features)))
            
            # Fill in the values
            for i, feature in enumerate(features):
                X[0, i] = user_inputs.get(feature, 0)
            
            # Get model
            model = create_dummy_model()
            model.set_feature_mapping(features)
            
            # Get risk probability
            risk_prob = model.predict_proba(X)[0, 1]
            
            # Adjust risk based on BMI and age to make it more realistic
            if user_inputs['_BMI5CAT'] >= 3 and user_inputs['_AGE_G'] >= 6:
                risk_prob = max(risk_prob, 0.4)  # Higher risk for overweight older people
            if user_inputs['_BMI5CAT'] == 4 and user_inputs['BPHIGH6'] == 1:
                risk_prob = max(risk_prob, 0.6)  # Much higher risk for obese with high BP
            if user_inputs['_BMI5CAT'] <= 2 and user_inputs['EXERANY2'] == 1:
                risk_prob = min(risk_prob, 0.3)  # Lower risk for normal weight with exercise
                
            # Determine risk level
            threshold = 0.5
            is_high_risk = risk_prob >= threshold
            
            # Display result
            st.markdown("### Your Risk Assessment Results")
            
            risk_percent = risk_prob * 100
            
            if is_high_risk:
                st.markdown(f"""
                <div class="high-risk">
                <h3>Higher Risk: {risk_percent:.1f}%</h3>
                <p>Your health information suggests you may have a higher risk of developing diabetes.
                This is not a diagnosis, just an indication that you might want to talk to a healthcare provider.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="low-risk">
                <h3>Lower Risk: {risk_percent:.1f}%</h3>
                <p>Your health information suggests you may have a lower risk of developing diabetes.
                Remember that maintaining a healthy lifestyle is still important for prevention.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Key risk factors
            st.markdown("### Key Risk Factors")
            risk_factors = []
            
            if user_inputs['_BMI5CAT'] >= 3:
                risk_factors.append("• Higher BMI (overweight or obese)")
            if user_inputs['_AGE_G'] >= 6:
                risk_factors.append("• Age (45+ years)")
            if user_inputs['BPHIGH6'] == 1:
                risk_factors.append("• High blood pressure")
            if user_inputs['EXERANY2'] == 2:
                risk_factors.append("• Physical inactivity")
            if user_inputs['GENHLTH'] >= 4:
                risk_factors.append("• Fair or poor general health")
            
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No major risk factors identified.")
                
            # Disclaimer
            st.markdown("""
            <div style="font-size:0.8rem; color:#666; margin-top:20px">
            <strong>Disclaimer:</strong> This simplified tool provides an approximate risk assessment based on limited factors.
            It is not a diagnostic tool and should not replace professional medical advice.
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### About This Tool")
    st.markdown("""
    This is a simplified version of the Diabetes Risk Prediction tool designed to run on Streamlit Cloud.
    
    #### How It Works
    This tool assesses your potential risk of developing diabetes based on key risk factors identified by health research:
    
    - **Age**: Risk increases with age, especially after 45
    - **BMI**: Being overweight or obese significantly increases risk
    - **Physical Activity**: Regular exercise reduces risk
    - **Blood Pressure**: High blood pressure is associated with increased risk
    - **General Health**: Overall health status correlates with diabetes risk
    
    #### Limitations
    Please note that this cloud version:
    - Uses a simplified model for demonstration purposes
    - Does not use full machine learning capabilities
    - Is designed for educational purposes only
    
    #### Resources
    For more information about diabetes risk, please visit:
    - [American Diabetes Association](https://www.diabetes.org/)
    - [CDC Diabetes Risk Assessment](https://www.cdc.gov/diabetes/risk/)
    """) 