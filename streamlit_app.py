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

# Auto-install all required packages without checking first
st.info("Installing required packages...")
for package_name, install_spec in required_packages.items():
    install_package(install_spec)
st.success("Dependencies installation complete!")

# Always run fix_model.py to ensure we have working model files
st.info("Generating model files...")
try:
    import fix_model
    success = fix_model.repair_model_files()
    if success:
        st.success("Model files have been generated and are ready to use.")
    else:
        st.warning("Could not fully repair model files, but will attempt to continue.")
except Exception as e:
    st.error(f"Error generating model files: {e}")

# Launch main application
st.title("✨ Diabetes Risk Prediction Tool")
st.markdown("### Setup Complete")
st.success("Dependencies installed and model files generated.")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.write("Please click the button below to launch the application")
    if st.button("🚀 Launch Diabetes Risk Prediction App", type="primary", use_container_width=True):
        # Import the app module without running the page config again
        import sys
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("app", "app.py")
            app_module = importlib.util.module_from_spec(spec)
            sys.modules["app"] = app_module
            spec.loader.exec_module(app_module)
            st.success("Application loaded successfully!")
        except Exception as e:
            st.error(f"Error loading the application: {e}")
            st.error("Please refresh the page and try again.") 