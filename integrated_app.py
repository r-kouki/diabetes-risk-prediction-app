import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import json
import sys
import subprocess

# Display a title immediately to show the app is loading
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✨ Diabetes Risk Prediction")
st.markdown("##### A tool to assess your diabetes risk factors")

# Show directory contents and path for debugging - now displayed immediately
st.markdown("### 📁 Directory Contents and File Status")
# Get current working directory
cwd = os.getcwd()
st.write(f"**Current Working Directory:** `{cwd}`")

# List files in the current directory
st.write("**Files in Current Directory:**")
files = os.listdir(cwd)
for file in sorted(files):
    file_path = os.path.join(cwd, file)
    if os.path.isdir(file_path):
        st.write(f"📁 `{file}/`")
    elif file.endswith('.pkl'):
        st.write(f"🧪 `{file}` ({os.path.getsize(file_path)/1024:.1f} KB)")
    else:
        st.write(f"📄 `{file}`")

# Check specifically for model files
st.write("**Model File Status:**")
model_files = ['calibrated_model.pkl', 'scaler.pkl', 'imputer.pkl', 
              'feature_list.json', 'feature_list.txt', 
              'optimal_threshold.json', 'optimal_threshold.txt']

for model_file in model_files:
    file_path = os.path.join(cwd, model_file)
    if os.path.exists(file_path):
        st.write(f"✅ `{model_file}` is present ({os.path.getsize(file_path)/1024:.1f} KB)")
    else:
        st.write(f"❌ `{model_file}` is missing")

st.markdown("---")

# Function to install packages
def install_package(package_name):
    try:
        st.info(f"Attempting to install {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        st.success(f"Successfully installed {package_name}")
        return True
    except Exception as e:
        st.error(f"Failed to install {package_name}: {str(e)}")
        return False

# Check and install required packages
required_packages = {
    "scikit-learn": "scikit-learn==1.3.0",
    "joblib": "joblib==1.3.2"
}

# Install missing packages
for package_name, package_spec in required_packages.items():
    try:
        # Try importing the package
        if package_name == "scikit-learn":
            import sklearn
            st.success(f"✅ {package_name} is already installed")
        elif package_name == "joblib":
            import joblib
            st.success(f"✅ {package_name} is already installed")
    except ImportError:
        # If import fails, try to install it
        if not install_package(package_spec):
            if package_name == "scikit-learn":
                # If scikit-learn installation fails, show error and stop
                st.error(f"⚠️ {package_name} is required but could not be installed.")
                st.info("For Streamlit Cloud deployment, make sure scikit-learn is in your requirements.txt file.")
                
                # Create a requirements.txt file suggestion
                st.markdown("""
                ### Missing Required Packages
                
                Your app needs scikit-learn to load the model files. Here's a sample requirements.txt file to use:
                
                ```
                streamlit==1.32.0
                pandas==2.0.3
                numpy==1.24.3
                scikit-learn==1.3.0
                matplotlib==3.7.2
                scipy==1.10.1
                xgboost==1.7.3
                joblib==1.3.2
                pickle5==0.0.12
                ```
                
                1. Add this to a requirements.txt file in your repository
                2. Redeploy your Streamlit app
                """)
                st.stop()

# Verify scikit-learn is available
sklearn_available = False
try:
    import sklearn
    # Check scikit-learn version
    st.info(f"scikit-learn version: {sklearn.__version__}")
    
    # Try importing specific modules one by one with better error reporting
    try:
        from sklearn.impute import SimpleImputer
        st.success("✓ SimpleImputer module imported successfully")
    except ImportError as e:
        st.error(f"Failed to import SimpleImputer: {str(e)}")
        # Try to install scipy which is a common dependency
        install_package("scipy>=1.3.2")
        
    try:
        from sklearn.preprocessing import StandardScaler
        st.success("✓ StandardScaler module imported successfully")
    except ImportError as e:
        st.error(f"Failed to import StandardScaler: {str(e)}")
    
    # Continue with the app even if some modules failed
    sklearn_available = True
    st.success("✓ Base scikit-learn is installed and working")
except ImportError as e:
    st.error(f"scikit-learn is not available: {str(e)}")
    st.stop()

# Try to ensure the model can be used even with partial scikit-learn functionality
if not sklearn_available:
    st.error("scikit-learn is not fully functioning. Please check your installation.")
    st.stop()

# Try to ensure numpy is properly installed as it's needed for prediction
try:
    import numpy as np
    st.success("✓ NumPy is working correctly")
except Exception as e:
    st.error(f"NumPy error: {str(e)}")
    install_package("numpy==1.24.3")

# Suppress warnings
warnings.filterwarnings('ignore')

# Quietly try to import joblib without auto-installing
joblib_available = False
try:
    import joblib
    joblib_available = True
except ImportError:
    pass  # Already tried to install above

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
    .error-msg {
        background-color: #ffdddd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL FUNCTIONS --------------------

# Function to check if model is a dummy
def is_dummy_model(model):
    """Check if model is a dummy or the real trained model"""
    try:
        # If it's a RandomForestClassifier with exactly 10 estimators, 
        # it's likely our dummy model
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

# Load the model files
@st.cache_resource
def load_model_files():
    """Load model files from disk"""
    try:
        # Check if the model files exist
        if not os.path.exists('calibrated_model.pkl') or not os.path.exists('scaler.pkl') or not os.path.exists('imputer.pkl'):
            return None, None, None, None, None, None, "Required model files not found. Please ensure calibrated_model.pkl, scaler.pkl, and imputer.pkl exist."
        
        # Try to load with joblib first (if available), then fall back to pickle
        model = None
        scaler = None
        imputer = None
        
        # Only try joblib if it's available
        if joblib_available:
            try:
                model = joblib.load('calibrated_model.pkl')
                scaler = joblib.load('scaler.pkl')
                imputer = joblib.load('imputer.pkl')
                st.success("✓ Successfully loaded model files with joblib")
            except Exception as e:
                st.warning(f"Joblib loading failed: {str(e)}")
                model = None  # Reset to trigger pickle loading
        
        # If joblib is not available or failed, use pickle
        if model is None:
            try:
                with open('calibrated_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                with open('imputer.pkl', 'rb') as f:
                    imputer = pickle.load(f)
                st.success("✓ Successfully loaded model files with pickle")
            except Exception as e:
                st.error(f"Error loading model files with pickle: {str(e)}")
                # If we get here, both joblib and pickle failed
                # Create a very basic fallback model for demonstration
                return create_fallback_model()
        
        # Try to load features
        try:
            if os.path.exists('feature_list.json'):
                with open('feature_list.json', 'r') as f:
                    features = json.load(f)['features']
            elif os.path.exists('feature_list.txt'):
                with open('feature_list.txt', 'r') as f:
                    features = [line.strip() for line in f.readlines()]
            else:
                return None, None, None, None, None, None, "Feature list file not found. Please ensure feature_list.json or feature_list.txt exists."
        except Exception as e:
            return None, None, None, None, None, None, f"Error loading feature list: {str(e)}"
        
        # Load threshold
        try:
            if os.path.exists('optimal_threshold.json'):
                with open('optimal_threshold.json', 'r') as f:
                    threshold = json.load(f)['optimal_threshold']
            elif os.path.exists('optimal_threshold.txt'):
                with open('optimal_threshold.txt', 'r') as f:
                    threshold = float(f.read().strip())
            else:
                threshold = 0.5  # Default threshold if file not found
        except Exception as e:
            return None, None, None, None, None, None, f"Error loading threshold: {str(e)}"
        
        # Check if model is dummy
        is_dummy = is_dummy_model(model)
        
        # Check for model metadata
        if os.path.exists('model_metadata.json'):
            try:
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = None
        else:
            metadata = None
            
        return model, scaler, imputer, features, threshold, is_dummy, None  # No error
    except Exception as e:
        return None, None, None, None, None, None, f"Unexpected error: {str(e)}"

# Create a basic fallback model if loading fails
def create_fallback_model():
    """Create a basic model with simple risk logic if loading fails"""
    st.warning("Creating a basic fallback model for demonstration purposes.")
    
    # Basic model class that implements predict_proba
    class BasicModel:
        def predict_proba(self, X):
            # Basic risk calculation based on key features
            n_samples = X.shape[0]
            probas = np.zeros((n_samples, 2))
            
            for i in range(n_samples):
                # Start with baseline risk
                risk = 0.25
                
                # Try to use key risk factors if they exist in the input
                try:
                    # Age factor (usually at index 0)
                    if X[i, 0] >= 6:  # Age 45+
                        risk += 0.15
                        
                    # BMI factor (usually at index 1)
                    if X[i, 1] >= 3:  # Overweight or obese
                        risk += 0.15
                    
                    # General health (usually at index 2)
                    if X[i, 2] >= 4:  # Fair or poor
                        risk += 0.1
                except:
                    # If indices are wrong, just use the baseline
                    pass
                
                # Clip to valid probability range
                risk = max(0.05, min(0.95, risk))
                probas[i, 0] = 1 - risk
                probas[i, 1] = risk
                
            return probas
    
    # Basic transformer that does nothing
    class IdentityTransformer:
        def transform(self, X):
            return X
    
    # Create the basic components
    model = BasicModel()
    scaler = IdentityTransformer()
    imputer = IdentityTransformer()
    
    # Use a minimal set of features
    features = [
        '_AGE_G', '_BMI5CAT', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH',
        'EXERANY2', 'SMOKE100', 'WTKG3', 'INCOME3', 'BPHIGH6'
    ]
    
    # Default threshold
    threshold = 0.5
    
    # This is definitely a dummy model
    is_dummy = True
    
    return model, scaler, imputer, features, threshold, is_dummy, None

# -------------------- UI FUNCTIONS --------------------

# Load model components
model, scaler, imputer, features, threshold, is_dummy, error_msg = load_model_files()

# Check if models loaded successfully
if error_msg:
    st.markdown(f"""
    <div class="error-msg">
    <h3>⚠️ Error Loading Models</h3>
    <p>{error_msg}</p>
    <p>Please ensure all required model files are available in the application directory.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()  # Stop execution if models couldn't be loaded

# Show model info
if is_dummy:
    st.markdown("""
    <div class="model-info">
    ⚠️ Using simplified model for demonstration. Results are approximate.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="model-info">
    ✅ Using trained diabetes prediction model
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📋 Risk Assessment", "ℹ️ About"])

with tab1:
    st.markdown("### Enter your health information")
    st.markdown("Fill out the form below to get a diabetes risk assessment.")
    
    # Create form for user input
    user_inputs = {}
    
    # Get subset of features for UI - only show the most important ones
    ui_features = [
        '_AGE_G', '_BMI5CAT', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH',
        'EXERANY2', 'SMOKE100', 'WTKG3', 'INCOME3', 'BPHIGH6'
    ]
    
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
            try:
                # Prepare input data for the model
                # First, create a DataFrame with the features expected by the model
                input_data = {}
                
                # Fill known values from user inputs
                for feature in features:
                    if feature in user_inputs:
                        input_data[feature] = [user_inputs[feature]]
                    else:
                        # For features not directly collected, use sensible defaults
                        input_data[feature] = [0]  # Default to 0 or "No" for most binary features
                
                # Create DataFrame
                input_df = pd.DataFrame(input_data)
                
                # Apply preprocessing
                X = input_df.values
                X_imputed = imputer.transform(X)
                X_scaled = scaler.transform(X_imputed)
                
                # Get prediction
                risk_prob = model.predict_proba(X_scaled)[0, 1]
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
                
            except Exception as e:
                st.error(f"Error calculating risk: {str(e)}")
                st.error("Please try again or check your input values.")
                
            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool provides an approximate risk assessment based on limited factors.
            It is not a diagnostic tool and should not replace professional medical advice.
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### About This Tool")
    st.markdown("""
    This is a tool to assess your potential risk of developing diabetes based on key risk factors identified by health research.
    
    #### How It Works
    The tool uses a trained machine learning model to analyze your health information and estimate your diabetes risk.
    
    #### Key Risk Factors
    - **Age**: Risk increases with age, especially after 45
    - **BMI**: Being overweight or obese significantly increases risk
    - **Physical Activity**: Regular exercise reduces risk
    - **Blood Pressure**: High blood pressure is associated with increased risk
    - **General Health**: Overall health status correlates with diabetes risk
    
    #### Resources
    For more information about diabetes risk, please visit:
    - [American Diabetes Association](https://www.diabetes.org/)
    - [CDC Diabetes Risk Assessment](https://www.cdc.gov/diabetes/risk/)
    """)
    
    # Model information
    st.markdown("### Technical Information")
    if is_dummy:
        st.markdown("""
        This version uses a simplified model for demonstration purposes.
        """)
    else:
        st.markdown("""
        This version uses a trained machine learning model to predict diabetes risk based on a comprehensive
        set of health and demographic factors.
        """) 