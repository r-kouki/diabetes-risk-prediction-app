#!/bin/bash

# Make sure pip is up to date
pip install --upgrade pip

# Install the dependencies explicitly
pip install streamlit>=1.32.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn
pip install matplotlib==3.7.2
pip install scipy==1.10.1
pip install xgboost==1.7.3
pip install joblib
pip install pickle5==0.0.12

# Execute the app
streamlit run app.py 