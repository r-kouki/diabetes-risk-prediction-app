import streamlit as st
import sys
import os
import subprocess
import importlib.util

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Function to check if a package is installed
def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

# Function to install a package using pip
def install_package(package_name):
    st.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        st.success(f"Successfully installed {package_name}")
        return True
    except Exception as e:
        st.error(f"Failed to install {package_name}: {e}")
        return False

# Check required packages
required_packages = {
    "scikit-learn": "scikit-learn==1.3.0",
    "pandas": "pandas==2.0.3",
    "numpy": "numpy==1.24.3", 
    "joblib": "joblib==1.3.2",
    "matplotlib": "matplotlib==3.7.2",
    "xgboost": "xgboost==1.7.3"
}

# Check if required packages are installed
missing_packages = []
for package, install_spec in required_packages.items():
    if not is_package_installed(package):
        missing_packages.append(install_spec)

# Install missing packages if needed
if missing_packages:
    st.warning("Some required packages are missing. Installing them now...")
    for package in missing_packages:
        install_package(package)
    st.success("All dependencies installed successfully!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("Please click the button below to launch the application")
        if st.button("🚀 Launch Diabetes Risk Prediction App", type="primary", use_container_width=True):
            # Import the app module without running the page config again
            import sys
            import importlib.util
            spec = importlib.util.spec_from_file_location("app", "app.py")
            app_module = importlib.util.module_from_spec(spec)
            sys.modules["app"] = app_module
            spec.loader.exec_module(app_module)
    st.stop()

# Check for model files
required_files = [
    "calibrated_model.pkl",
    "scaler.pkl",
    "imputer.pkl",
    "feature_list.json",
    "optimal_threshold.json"
]

missing_files = [file for file in required_files if not os.path.exists(file)]

# If any model files are missing, run the fix_model script
if missing_files:
    st.warning(f"Missing required model files: {', '.join(missing_files)}")
    st.info("Attempting to fix model files...")
    
    try:
        import fix_model
        success = fix_model.repair_model_files()
        if success:
            st.success("Model files have been fixed.")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.write("Please click the button below to launch the application")
                if st.button("🚀 Launch Diabetes Risk Prediction App", type="primary", use_container_width=True):
                    # Import the app module without running the page config again
                    import sys
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("app", "app.py")
                    app_module = importlib.util.module_from_spec(spec)
                    sys.modules["app"] = app_module
                    spec.loader.exec_module(app_module)
        else:
            st.error("Failed to fix model files. Please check the logs for more information.")
        st.stop()
    except Exception as e:
        st.error(f"Error running fix_model: {e}")
        st.stop()

# If all checks pass, show a welcome screen with a launch button
st.title("✨ Diabetes Risk Prediction Tool")
st.markdown("### Setup Complete")
st.success("All dependencies and model files are available and ready to use.")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.write("Please click the button below to launch the application")
    if st.button("🚀 Launch Diabetes Risk Prediction App", type="primary", use_container_width=True):
        # Import the app module without running the page config again
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        sys.modules["app"] = app_module
        spec.loader.exec_module(app_module) 