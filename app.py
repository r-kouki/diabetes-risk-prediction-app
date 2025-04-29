import streamlit as st

# Only set page config if this script is run directly (not imported)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Diabetes Risk Prediction",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

import pandas as pd
import numpy as np
import pickle
import os
try:
    import joblib
except ImportError:
    # Fallback if joblib isn't available
    import pickle as joblib
    st.warning("Using pickle as fallback since joblib is not available")
import json
import warnings

# Try to import sklearn, with graceful fallback
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    st.error("scikit-learn is not installed. Some features may not work properly.")
    # Define a dummy class to prevent errors
    class InconsistentVersionWarning(Warning):
        pass

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4da6ff;
        color: white;
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
    div.block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #2e6da4;
    }
    h2, h3 {
        color: #3d85c6;
    }
    .css-zt5igj {
        text-align:center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
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
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        margin-top: 1rem;
    }
    .model-info {
        background-color: #e6f3ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
    }
    .dummy-model {
        background-color: #fff3cd;
        color: #856404;
    }
    .real-model {
        background-color: #d4edda;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Function to check if model is a dummy or the real model
def is_dummy_model(model):
    """Check if model is a dummy or the real trained model"""
    try:
        # If it's a RandomForestClassifier with exactly 10 estimators, 
        # it's likely our dummy model from fix_model.py
        if hasattr(model, 'estimators_') and len(model.estimators_) == 10:
            return True
        # For CalibratedClassifierCV models, check base estimator
        if hasattr(model, 'base_estimator'):
            if hasattr(model.base_estimator, 'estimators_') and len(model.base_estimator.estimators_) == 10:
                return True
        return False
    except:
        # If any error occurs during checking, assume it's not a dummy
        return False

# Load model metadata if available
def load_model_metadata():
    """Load model metadata from JSON if available"""
    try:
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                return json.load(f)
        return None
    except:
        return None

# Load features list
def load_features():
    """Load features from JSON or text file"""
    try:
        # First try to load from JSON
        if os.path.exists('feature_list.json'):
            with open('feature_list.json', 'r') as f:
                data = json.load(f)
                return data['features']
        
        # If JSON not available, try text file
        if os.path.exists('feature_list.txt'):
            with open('feature_list.txt', 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # If neither exists, throw an error
        raise FileNotFoundError("No feature list file found")
    except Exception as e:
        st.error(f"Error loading features: {str(e)}")
        raise

# Load optimal threshold
def load_threshold():
    """Load threshold from JSON or text file"""
    try:
        # First try to load from metadata
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                data = json.load(f)
                return data['optimal_threshold']
        
        # Then try to load from dedicated JSON
        if os.path.exists('optimal_threshold.json'):
            with open('optimal_threshold.json', 'r') as f:
                data = json.load(f)
                return data['optimal_threshold']
        
        # If JSON not available, try text file
        if os.path.exists('optimal_threshold.txt'):
            with open('optimal_threshold.txt', 'r') as f:
                return float(f.read().strip())
        
        # If neither exists, use default
        return 0.5
    except Exception as e:
        st.error(f"Error loading threshold: {str(e)}")
        return 0.5

# Load model and preprocessing components
@st.cache_resource
def load_model_components():
    try:
        # Try loading with joblib first
        try:
            model = joblib.load('calibrated_model.pkl')
            scaler = joblib.load('scaler.pkl')
            imputer = joblib.load('imputer.pkl')
            st.success("Successfully loaded model files with joblib")
        except Exception as e:
            st.warning(f"Joblib load failed, trying with pickle: {str(e)}")
            # Fall back to regular pickle if joblib fails
            try:
                with open('calibrated_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                with open('imputer.pkl', 'rb') as f:
                    imputer = pickle.load(f)
                st.success("Successfully loaded model files with pickle")
            except Exception as e:
                st.error(f"Failed to load model files with pickle: {str(e)}")
                st.info("Running fix_model.py to generate new model files...")
                # If both joblib and pickle fail, run fix_model
                import fix_model
                if fix_model.repair_model_files():
                    st.success("Generated new model files")
                    # Try loading again
                    with open('calibrated_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    with open('scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    with open('imputer.pkl', 'rb') as f:
                        imputer = pickle.load(f)
                else:
                    raise Exception("Failed to generate model files")
        
        # Load features and threshold
        features = load_features()
        threshold = load_threshold()
        
        # Load metadata
        metadata = load_model_metadata()
        
        # Check if model is dummy
        is_dummy = is_dummy_model(model)
        
        return model, scaler, imputer, features, threshold, is_dummy, metadata
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.error("Try refreshing the page or running fix_model.py to repair the files")
        raise

try:
    model, scaler, imputer, features, threshold, is_dummy, metadata = load_model_components()

    # Header with logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("✨ Diabetes Risk Predictor")
        st.markdown("##### A simple tool to assess your diabetes risk factors")
        
        # Show model info
        if is_dummy:
            st.markdown("""
            <div class="model-info dummy-model">
            ⚠️ Using temporary model for demonstration. Predictions are not based on real diabetes data.
            </div>
            """, unsafe_allow_html=True)
        else:
            if metadata:
                model_version = metadata.get('model_version', 'Unknown')
                model_name = metadata.get('model_name', 'Diabetes Prediction Model')
                st.markdown(f"""
                <div class="model-info real-model">
                ✅ {model_name} (v{model_version})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="model-info real-model">
                ✅ Using trained diabetes prediction model
                </div>
                """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📋 Risk Assessment", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Enter your health information")
        st.markdown("Fill out the form below with your health details to get a personalized diabetes risk assessment.")
        
        # Progress bar to show form completion
        progress_placeholder = st.empty()
        
        # Function to create input fields based on feature names
        def create_input_fields():
            user_inputs = {}
            # Track which fields the user has completed
            filled_user_inputs = set()
            
            # Use a container for the form
            with st.container():
                # Create tabs for related health information
                personal_tab, health_tab, lifestyle_tab = st.tabs(["💡 Personal", "🩺 Health History", "🏃‍♂️ Lifestyle"])
                
                # Personal Information Tab
                with personal_tab:
                    
                    # Add information about personal factors
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    📌 <b>Why these factors matter:</b> Age, weight, and BMI are key predictors of diabetes risk. 
                    As you age, especially after 45, your risk increases significantly. Higher BMI (over 25) is strongly 
                    associated with insulin resistance, which is a precursor to type 2 diabetes. Income level can affect 
                    access to healthcare, nutritious food, and preventive care opportunities.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        age_options = [None] + list(range(1, 15))
                        age_format = lambda x: "" if x is None else {1: "18-24 years", 2: "25-29 years", 3: "30-34 years", 4: "35-39 years", 
                                                  5: "40-44 years", 6: "45-49 years", 7: "50-54 years", 8: "55-59 years", 
                                                  9: "60-64 years", 10: "65-69 years", 11: "70-74 years", 12: "75-79 years", 
                                                  13: "80+ years", 14: "Don't know/Not sure"}[x]
                        
                        selected_age = st.selectbox(
                            "📅 Age Group", 
                            options=age_options,
                            format_func=age_format,
                            index=0,  # Default to None
                            help="Age is a significant risk factor for diabetes. Risk increases with age, particularly after 45 years old. After age 45, diabetes risk approximately doubles for each decade of life."
                        )
                        
                        if selected_age is not None:
                            user_inputs['_AGE_G'] = selected_age
                            filled_user_inputs.add('_AGE_G')
                        
                        selected_weight = st.number_input(
                            "⚖️ Weight (kg)", 
                            min_value=None,
                            max_value=200.0,
                            value=None,
                            help="Weight is an important factor in diabetes risk assessment. Higher body weight can increase insulin resistance. Each 5kg of excess weight increases diabetes risk by approximately 30%. The body stores excess calories as fat, which produces hormones and inflammatory substances that impair glucose regulation."
                        )
                        
                        if selected_weight is not None:
                            user_inputs['WTKG3'] = selected_weight
                            filled_user_inputs.add('WTKG3')
                    
                    with col2:
                        bmi_options = [None, 1, 2, 3, 4]
                        bmi_format = lambda x: "" if x is None else {1: "Underweight", 2: "Normal Weight", 3: "Overweight", 4: "Obese"}[x]
                        
                        selected_bmi = st.selectbox(
                            "📏 BMI Category", 
                            options=bmi_options,
                            format_func=bmi_format,
                            index=0,  # Default to None
                            help="Body Mass Index (BMI) is a key predictor of diabetes risk. BMI over 25 (overweight) increases risk by 3x, and BMI over 30 (obese) increases risk by 7x compared to normal weight. Fat distributed around the abdomen (visceral fat) is particularly associated with insulin resistance and diabetes."
                        )
                        
                        if selected_bmi is not None:
                            user_inputs['_BMI5CAT'] = selected_bmi
                            filled_user_inputs.add('_BMI5CAT')
                        
                        income_options = [None] + list(range(1, 9))
                        income_format = lambda x: "" if x is None else {1: "Less than $10,000", 2: "$10,000-$15,000", 3: "$15,000-$20,000", 4: "$20,000-$25,000", 5: "$25,000-$35,000", 6: "$35,000-$50,000", 7: "$50,000-$75,000", 8: "$75,000+"}[x]
                        
                        selected_income = st.selectbox(
                            "💰 Income Category", 
                            options=income_options,
                            format_func=income_format,
                            index=0,  # Default to None
                            help="Income level can be associated with healthcare access, food security, and lifestyle factors that influence diabetes risk. Lower income is associated with limited access to preventive care, nutritious food, and safe places for physical activity, all of which can increase diabetes risk."
                        )
                        
                        if selected_income is not None:
                            user_inputs['INCOME3'] = selected_income
                            filled_user_inputs.add('INCOME3')
                
                # Health History Tab
                with health_tab:
                    
                    # Add explanation about health history importance
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    📌 <b>Why health history matters:</b> Your general health status and existing medical conditions 
                    are strong indicators of diabetes risk. Cardiovascular conditions in particular share many of 
                    the same risk factors as diabetes, including insulin resistance, inflammation, and metabolic dysfunction.
                    Many of these conditions reflect underlying physiological processes that also influence blood glucose regulation.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### General Health Status")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        health_options = [None, 1, 2, 3, 4, 5]
                        health_format = lambda x: "" if x is None else {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x]
                        
                        selected_health = st.selectbox(
                            "🌡️ Overall Health", 
                            options=health_options,
                            format_func=health_format,
                            index=0,  # Default to None
                            help="Self-reported health status strongly correlates with diabetes risk. Those reporting 'fair' or 'poor' health have a 2-3x higher risk of diabetes compared to those reporting 'excellent' health. This metric often reflects multiple underlying health issues that may affect metabolism and insulin sensitivity."
                        )
                        
                        if selected_health is not None:
                            user_inputs['GENHLTH'] = selected_health
                            filled_user_inputs.add('GENHLTH')
                        
                        phys_health = st.slider(
                            "🤒 Days Physical Health Not Good (Past 30 days)", 
                            min_value=0, 
                            max_value=30,
                            value=None,
                            help="Frequent days of poor physical health may indicate chronic health issues that overlap with diabetes risk factors. More than 14 days of poor physical health in a month is associated with a significantly higher risk of chronic conditions including diabetes. This may also reflect symptoms of undiagnosed diabetes like fatigue and frequent infections."
                        )
                        
                        if phys_health is not None:
                            user_inputs['PHYSHLTH'] = phys_health
                            filled_user_inputs.add('PHYSHLTH')
                            
                            # Add descriptive text below the slider
                            if user_inputs['PHYSHLTH'] == 0:
                                st.caption("No days of poor physical health (optimal)")
                            elif user_inputs['PHYSHLTH'] < 14:
                                st.caption("Occasional poor physical health")
                            else:
                                st.caption("Chronic poor physical health (higher risk)")
                    
                    with col2:
                        ment_health = st.slider(
                            "😔 Days Mental Health Not Good (Past 30 days)", 
                            min_value=0, 
                            max_value=30,
                            value=None,
                            help="Mental health impacts diabetes risk through multiple pathways. Depression and chronic stress affect cortisol levels and other hormones that influence blood sugar regulation. Poor mental health is associated with a 37% higher risk of developing type 2 diabetes. Stress hormones can directly increase blood glucose levels and promote insulin resistance."
                        )
                        
                        if ment_health is not None:
                            user_inputs['MENTHLTH'] = ment_health
                            filled_user_inputs.add('MENTHLTH')
                            
                            # Add descriptive text below the slider
                            if user_inputs['MENTHLTH'] == 0:
                                st.caption("No days of poor mental health (optimal)")
                            elif user_inputs['MENTHLTH'] < 14:
                                st.caption("Occasional poor mental health")
                            else:
                                st.caption("Chronic poor mental health (higher risk)")
                    
                    st.markdown("##### Medical Conditions")
                    st.markdown("""
                    <div style="background-color: #f0f7ff; padding: 8px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;">
                    Several medical conditions share common risk factors with diabetes or can directly influence blood sugar control. 
                    The presence of cardiovascular disease in particular is strongly associated with insulin resistance and metabolic syndrome.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        heart_attack_options = ["", "Yes", "No"]
                        selected_heart_attack = st.radio(
                            "❤️ Heart Attack History", 
                            options=heart_attack_options,
                            index=0,  # Default to blank
                            help="Heart attack history indicates underlying cardiovascular disease, which shares many risk factors with diabetes. Approximately 68% of people age 65+ with diabetes die from heart disease. The relationship works in both directions: heart disease increases diabetes risk, and diabetes increases heart disease risk."
                        )
                        
                        if selected_heart_attack:
                            user_inputs['CVDINFR4'] = 1 if selected_heart_attack == "Yes" else 2
                            filled_user_inputs.add('CVDINFR4')
                        
                        heart_disease_options = ["", "Yes", "No"]
                        selected_heart_disease = st.radio(
                            "💓 Coronary Heart Disease", 
                            options=heart_disease_options,
                            index=0,  # Default to blank
                            help="Coronary heart disease is strongly linked to insulin resistance and metabolic syndrome. The inflammation and metabolic dysfunction involved in coronary heart disease also contribute to diabetes risk. People with coronary heart disease have a 2-4x higher risk of developing type 2 diabetes compared to those without."
                        )
                        
                        if selected_heart_disease:
                            user_inputs['CVDCRHD4'] = 1 if selected_heart_disease == "Yes" else 2
                            filled_user_inputs.add('CVDCRHD4')
                    
                    with col2:
                        stroke_options = ["", "Yes", "No"]
                        selected_stroke = st.radio(
                            "🧠 Stroke History", 
                            options=stroke_options,
                            index=0,  # Default to blank
                            help="Stroke history suggests vascular problems that often accompany or precede diabetes. Both conditions share risk factors like high blood pressure, obesity, and high cholesterol. Having had a stroke increases diabetes risk by approximately 2x. Stroke can also reflect the type of vascular damage that is accelerated by diabetes."
                        )
                        
                        if selected_stroke:
                            user_inputs['CVDSTRK3'] = 1 if selected_stroke == "Yes" else 2
                            filled_user_inputs.add('CVDSTRK3')
                        
                        asthma_options = ["", "Yes", "No"]
                        selected_asthma = st.radio(
                            "🫁 Asthma", 
                            options=asthma_options,
                            index=0,  # Default to blank
                            help="Emerging research suggests connections between asthma, systemic inflammation, and insulin resistance. The chronic inflammation in asthma may contribute to impaired glucose metabolism. Additionally, some asthma medications (particularly corticosteroids) can raise blood sugar levels and increase diabetes risk with long-term use."
                        )
                        
                        if selected_asthma:
                            user_inputs['ASTHMA3'] = 1 if selected_asthma == "Yes" else 2
                            filled_user_inputs.add('ASTHMA3')
                    
                    with col3:
                        cancer_options = ["", "Yes", "No"]
                        selected_cancer = st.radio(
                            "🦠 Cancer History", 
                            options=cancer_options,
                            index=0,  # Default to blank
                            help="Cancer and its treatments can affect metabolism and pancreatic function. Some cancer treatments, particularly with steroids, can raise blood sugar levels. Pancreatic cancer specifically can directly impact insulin production. Additionally, the inflammatory state associated with many cancers can contribute to insulin resistance."
                        )
                        
                        if selected_cancer:
                            user_inputs['CHCOCNC1'] = 1 if selected_cancer == "Yes" else 2
                            filled_user_inputs.add('CHCOCNC1')
                        
                        kidney_options = ["", "Yes", "No"]
                        selected_kidney = st.radio(
                            "🫘 Kidney Disease", 
                            options=kidney_options,
                            index=0,  # Default to blank
                            help="Kidney disease and diabetes have a bidirectional relationship. Kidney disease can alter how the body processes insulin and glucose, potentially leading to insulin resistance. Conversely, diabetes is the leading cause of kidney disease. Having kidney disease increases diabetes risk by approximately 40-60% due to metabolic imbalances and impaired vitamin D activation."
                        )
                        
                        if selected_kidney:
                            user_inputs['CHCKDNY2'] = 1 if selected_kidney == "Yes" else 2
                            filled_user_inputs.add('CHCKDNY2')
                
                # Lifestyle Tab
                with lifestyle_tab:
                    
                    # Add explanation about lifestyle importance
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    📌 <b>Why lifestyle factors matter:</b> Physical activity, smoking, and alcohol consumption 
                    are modifiable factors that significantly impact your diabetes risk. Regular exercise improves 
                    insulin sensitivity by 20-65%, while smoking increases insulin resistance and impairs beta cell function.
                    These factors represent opportunities to reduce your risk through lifestyle changes.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        exercise_options = ["", "Yes", "No"]
                        selected_exercise = st.radio(
                            "🏋️‍♂️ Exercise in Past 30 Days", 
                            options=exercise_options,
                            index=0,  # Default to blank
                            help="Regular physical activity significantly improves insulin sensitivity and helps maintain healthy weight. Just 150 minutes of moderate exercise per week can reduce diabetes risk by 30-50%. Exercise increases glucose uptake by muscles (even without insulin), reduces fat stores, and decreases inflammation - all key factors in preventing diabetes."
                        )
                        
                        if selected_exercise:
                            user_inputs['EXERANY2'] = 1 if selected_exercise == "Yes" else 2
                            filled_user_inputs.add('EXERANY2')
                        
                        smoke_options = ["", "Yes", "No"]
                        selected_smoke = st.radio(
                            "🚬 Smoked 100+ Cigarettes in Lifetime", 
                            options=smoke_options,
                            index=0,  # Default to blank
                            help="Long-term smoking history increases diabetes risk through multiple mechanisms. Smokers have a 30-40% higher risk of developing type 2 diabetes compared to non-smokers. Smoking causes oxidative stress and inflammation, damages blood vessels, and directly impairs insulin-producing beta cells in the pancreas."
                        )
                        
                        if selected_smoke:
                            user_inputs['SMOKE100'] = 1 if selected_smoke == "Yes" else 2
                            filled_user_inputs.add('SMOKE100')
                        
                        smoke_status_options = ["", "Every day", "Some days", "Not at all"]
                        smoke_status = st.radio(
                            "🚭 Current Smoking Status", 
                            options=smoke_status_options,
                            index=0,  # Default to blank
                            help="Current smoking habits affect diabetes risk more than past smoking. Daily smoking has the strongest negative impact on insulin sensitivity and glucose metabolism. Active smoking increases diabetes risk by 30-40%, with a dose-dependent relationship (more cigarettes = higher risk). Quitting smoking can gradually reduce this added risk over 5-10 years."
                        )
                        
                        # Map smoking status to appropriate codes
                        if smoke_status:
                            filled_user_inputs.add('SMOKDAY2')
                            filled_user_inputs.add('_SMOKER3')
                            
                            if smoke_status == "Every day":
                                user_inputs['SMOKDAY2'] = 1
                                user_inputs['_SMOKER3'] = 1
                            elif smoke_status == "Some days":
                                user_inputs['SMOKDAY2'] = 2
                                user_inputs['_SMOKER3'] = 2
                            else:  # Not at all
                                user_inputs['SMOKDAY2'] = 3
                                user_inputs['_SMOKER3'] = 4  # Never smoked
                    
                    with col2:
                        alc_consumption = st.slider(
                            "🍷 Alcohol Consumption (days per week)", 
                            min_value=0, 
                            max_value=7,
                            value=None,  # Default to None
                            help="Alcohol's relationship with diabetes risk is complex. Moderate consumption (1-2 drinks/day) may actually reduce risk by 20-30% by improving insulin sensitivity. However, heavy drinking (>3 drinks/day) increases risk by damaging the pancreas and liver, which are essential for glucose regulation. Binge drinking in particular can disrupt metabolism and lead to weight gain."
                        )
                        
                        if alc_consumption is not None:
                            user_inputs['ALCDAY4'] = alc_consumption
                            filled_user_inputs.add('ALCDAY4')
                            
                            # Add descriptive text below the slider
                            if user_inputs['ALCDAY4'] == 0:
                                st.caption("No alcohol consumption")
                            elif 1 <= user_inputs['ALCDAY4'] <= 2:
                                st.caption("Occasional consumption (may have protective effects)")
                            elif 3 <= user_inputs['ALCDAY4'] <= 5:
                                st.caption("Moderate consumption")
                            else:
                                st.caption("Frequent consumption (may increase risk)")
                        
                        arthritis_options = ["", "Yes", "No"]
                        selected_arthritis = st.radio(
                            "🦴 Arthritis", 
                            options=arthritis_options,
                            index=0,  # Default to blank
                            help="Arthritis and diabetes share inflammatory pathways and risk factors. Chronic inflammation from arthritis can affect insulin sensitivity. Additionally, limited mobility due to arthritis often reduces physical activity levels, which is a key protective factor against diabetes. Pain medications for arthritis (particularly some NSAIDs) may also affect glucose metabolism with long-term use."
                        )
                        
                        if selected_arthritis:
                            user_inputs['HAVARTH4'] = 1 if selected_arthritis == "Yes" else 2
                            filled_user_inputs.add('HAVARTH4')
                        
                        depression_options = ["", "Yes", "No"]
                        selected_depression = st.radio(
                            "😟 Depression History", 
                            options=depression_options,
                            index=0,  # Default to blank
                            help="Depression affects hormones that regulate blood sugar, including cortisol and adrenaline. People with depression have a 37-60% higher risk of developing type 2 diabetes. Depression can also lead to unhealthy coping behaviors like poor diet, reduced activity, and disrupted sleep patterns, all of which independently increase diabetes risk."
                        )
                        
                        if selected_depression:
                            user_inputs['ADDEPEV3'] = 1 if selected_depression == "Yes" else 2
                            filled_user_inputs.add('ADDEPEV3')
            
            # Add required features with default values for prediction
            required_features = [
                'BPHIGH6', 'TOLDHI3', 'MICHD', 
                '_EDUCAG', '_INCOMG', '_SMOKER3', 
                '_RFDRHV6', '_PNEUMO3', '_RFSEAT3'
            ]
            
            # Only add defaults for required features that aren't explicitly filled
            for feature in required_features:
                if feature not in user_inputs:
                    user_inputs[feature] = 1  # Default value
            
            # Calculate derived features if base features are available
            if '_AGE_G' in user_inputs and '_BMI5CAT' in user_inputs:
                user_inputs['AGE_BMI'] = user_inputs['_AGE_G'] * user_inputs['_BMI5CAT']
                filled_user_inputs.add('AGE_BMI')
            
            if '_AGE_G' in user_inputs and 'BPHIGH6' in user_inputs:
                user_inputs['AGE_BP'] = user_inputs['_AGE_G'] * user_inputs['BPHIGH6']
                filled_user_inputs.add('AGE_BP')
            
            # Calculate progress based only on fields the user has explicitly filled out
            essential_fields = 18  # Number of key fields we want the user to fill out
            progress = len(filled_user_inputs) / essential_fields if essential_fields > 0 else 0.0
            progress = min(progress, 1.0)  # Cap at 100%
            progress_placeholder.progress(progress, text=f"Form completion: {int(progress*100)}%")
            
            return user_inputs

        # Get user inputs
        user_inputs = create_input_fields()

        # Prediction button
        st.markdown("---")
        if st.button("💻 Calculate My Risk", key="predict_button"):
            with st.spinner("Analyzing your health data..."):
                # Create a DataFrame with the user inputs
                input_df = pd.DataFrame([user_inputs])
                
                # Debug information
                st.write("### Debug Information")
                with st.expander("Debug Details"):
                    st.write("#### Input Features")
                    st.dataframe(input_df)
                
                # Make sure all features from training are present
                # Don't filter out HIGH_RISK - include all features
                missing_features = [f for f in features if f not in input_df.columns]
                
                # Fill missing features with default value 1
                for feature in missing_features:
                    input_df[feature] = 1
                
                # Ensure columns are in the same order as during training
                # Use all features without filtering out HIGH_RISK
                input_df = input_df[features]
                
                # Debug post-processing
                with st.expander("Debug Details", expanded=False):
                    st.write("#### After Feature Processing")
                    st.dataframe(input_df)
                
                # Apply imputation
                imputed_data = imputer.transform(input_df)
                
                # Scale the data
                scaled_data = scaler.transform(imputed_data)
                
                # Debug scaled data
                with st.expander("Debug Details", expanded=False):
                    st.write("#### Is Dummy Model:")
                    st.write(is_dummy)
                    if is_dummy:
                        st.error("Using dummy model! This model will not give accurate predictions because it was created as a replacement for a corrupted model.")
                    
                    if hasattr(model, 'estimators_'):
                        st.write(f"Number of estimators: {len(model.estimators_)}")
                    elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'estimators_'):
                        st.write(f"Number of estimators in base model: {len(model.base_estimator.estimators_)}")
                    
                    st.write("#### Model Information:")
                    st.write(f"Model type: {type(model)}")
                
                # Make prediction
                prediction_proba = model.predict_proba(scaled_data)[:, 1][0]
                prediction = 1 if prediction_proba >= threshold else 0
                
                # Show all probabilities
                with st.expander("Debug Details", expanded=False):
                    st.write("#### Raw Prediction Probabilities:")
                    st.write(model.predict_proba(scaled_data))
                    st.write(f"Threshold used: {threshold}")
                    
                # Display results
                st.markdown("### Your Results")
                
                # Results card
                result_container = st.container()
                with result_container:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric("Diabetes Risk Score", f"{prediction_proba:.1%}")
                        
                        if prediction == 1:
                            st.markdown("""
                            <div class="high-risk">
                            <h3>⚠️ Higher Risk of Diabetes</h3>
                            <p>Your risk factors suggest a higher than average likelihood of developing diabetes.</p>
                            <p>Please consult with a healthcare provider for proper evaluation.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="low-risk">
                            <h3>✅ Lower Risk of Diabetes</h3>
                            <p>Your risk factors suggest a lower likelihood of developing diabetes.</p>
                            <p>Continue to maintain a healthy lifestyle.</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Risk interpretation
                st.markdown(f"""
                #### What this means
                - Your risk score is **{prediction_proba:.1%}**
                - The prediction threshold is **{threshold:.1%}**
                - Scores above this threshold indicate higher risk
                """)
                
                # Recommendation section
                st.markdown("#### Next Steps")
                st.markdown("""
                Remember that this tool provides an estimate based on limited information:
                
                1. **For Everyone**: Regular check-ups with healthcare providers are important
                2. **For Higher Risk**: Consult with a doctor for proper testing and evaluation
                3. **For Lower Risk**: Maintain a healthy lifestyle including diet and exercise
                """)
                
                # Disclaimer
                st.markdown("""
                <div class="disclaimer">
                <strong>Disclaimer</strong>: This tool provides an estimate based on the information provided.
                It does not constitute medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
                </div>
                """, unsafe_allow_html=True)
    
    # About tab
    with tab2:
        st.markdown("### About this Diabetes Risk Prediction Tool")
        
        st.markdown("""
        This tool uses machine learning to estimate your risk of developing diabetes based on various health and lifestyle factors.
        
        #### How it Works
        The prediction model was trained on health survey data to identify patterns associated with diabetes risk. The app:
        1. Collects your health information through the questionnaire
        2. Processes this information using our trained machine learning model
        3. Calculates a personalized risk score based on your specific health profile
        4. Provides recommendations based on that score
        
        #### Key Risk Factors
        The model considers these important factors in diabetes risk prediction:
        
        **Personal Factors**
        - **Age**: Risk increases significantly after age 45, and approximately doubles for each decade thereafter
        - **BMI/Weight**: One of the strongest predictors; BMI over 30 increases risk by 7x compared to normal weight
        - **Income Level**: Affects access to healthcare, nutritious food, and safe exercise environments
        
        **Health History Factors**
        - **General Health Status**: Self-reported health strongly correlates with actual health outcomes
        - **Cardiovascular Conditions**: Heart disease, stroke, and diabetes share many underlying risk factors
        - **Kidney Disease**: Has a bidirectional relationship with diabetes and shares metabolic pathways
        - **Mental Health**: Depression and chronic stress affect hormones that regulate blood glucose
        
        **Lifestyle Factors**
        - **Physical Activity**: Regular exercise can reduce diabetes risk by 30-50%
        - **Smoking**: Increases risk by 30-40% through inflammation and direct damage to pancreatic cells
        - **Alcohol Consumption**: Moderate consumption may be protective, while heavy drinking increases risk
        
        #### Understanding Your Results
        Your risk score is calculated based on the complex interaction of all these factors:
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h5>Interpreting Your Score</h5>
        <ul>
            <li><strong>Below threshold</strong>: Your current risk factors suggest a lower likelihood of developing diabetes compared to the general population</li>
            <li><strong>Above threshold</strong>: Your combination of risk factors indicates a higher likelihood of developing diabetes</li>
            <li><strong>Score magnitude</strong>: Higher percentages indicate stronger evidence of risk based on your profile</li>
        </ul>
        <p>Remember that this is a statistical prediction based on population data, not a medical diagnosis.</p>
        </div>
        
        #### Limitations
        This tool has several important limitations to keep in mind:
        - It relies on self-reported information which may not be completely accurate
        - It cannot replace laboratory tests such as blood glucose measurements
        - It doesn't account for family history of diabetes, which is a significant risk factor
        - Genetic factors that influence diabetes risk are not captured
        - The model provides population-level estimates, not individualized biological assessments
        
        Always consult with healthcare professionals for proper diagnosis and treatment decisions.
        """)
        
        # Add feature importance section
        with st.expander("Feature Importance Explained"):
            st.markdown("""
            Our model weighs different factors based on their predictive power for diabetes risk:
            
            **Very High Impact Factors**
            - **BMI/Weight Status**: Obesity (BMI>30) can increase risk by 7-fold compared to normal weight
            - **Age**: Risk approximately doubles for every decade after age 40
            - **Physical Activity**: Regular exercise can reduce risk by 30-50% by improving insulin sensitivity
            - **Overall Health**: Self-reported "poor" health indicates a 2-3x higher risk
            
            **High Impact Factors**
            - **Cardiovascular Disease**: Heart conditions often share metabolic risk factors with diabetes
            - **High Blood Pressure**: Hypertension increases diabetes risk by 60%
            - **Kidney Disease**: Has a bidirectional relationship with diabetes risk
            - **Smoking**: Daily smoking increases risk by 30-40%
            
            **Moderate Impact Factors**
            - **Mental Health**: Depression increases risk by 37-60%
            - **Alcohol Consumption**: Heavy drinking increases risk while moderate consumption may be protective
            - **Income Level**: Lower income correlates with limited access to healthcare and nutrition
            - **Cancer History**: Some treatments and inflammatory states affect glucose metabolism
            
            Different combinations of these factors create your unique risk profile. The model integrates these 
            factors using complex mathematical relationships to generate your personalized risk score.
            
            **Important Note**: Modifiable factors like weight, exercise, smoking, and diet present opportunities 
            to significantly reduce your diabetes risk through lifestyle changes.
            """)
        
        # Expand for technical details
        with st.expander("Technical Details"):
            st.markdown("""
            This application uses:
            - A calibrated machine learning model trained on health survey data
            - Feature engineering and normalization to process input data
            - Threshold optimization to balance sensitivity and specificity
            
            **Model Type**: CalibratedClassifierCV with XGBoost base model
            
            **Performance Metrics**:
            - Precision: 54.5% (of those predicted to have diabetes, this percentage actually have it)
            - Recall: 38.9% (of all people with diabetes, this percentage are correctly identified)
            - F1 Score: 45.4% (harmonic mean of precision and recall)
            - AUC-ROC: 0.73 (model's ability to distinguish between positive and negative cases)
            
            **Feature Engineering**:
            - Age-BMI interaction effects are captured with compound features
            - Input standardization ensures consistent scaling across all features
            - Missing value imputation uses state-of-the-art methods
            
            The model was trained on demographic and health survey data from multiple years, with features
            selected based on their known association with diabetes risk from medical literature and feature
            importance analysis from machine learning algorithms.
            
            For questions or technical support, please contact the development team.
            """)

except Exception as e:
    st.error(f"Error loading model components: {str(e)}")
    st.error("Try refreshing the page or running fix_model.py to repair the files")
