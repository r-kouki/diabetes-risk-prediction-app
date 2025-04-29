# Diabetes Risk Prediction App - Streamlit Cloud Deployment

This repository contains a Streamlit app for diabetes risk prediction that can be deployed on Streamlit Cloud.

## Deployment Instructions

1. Connect this repository to Streamlit Cloud
2. Set the main app file to: `streamlit_app.py`
3. No additional settings required

## Features

- Automatically installs required dependencies
- Self-healing functionality for model files
- User-friendly interface for diabetes risk assessment

## Troubleshooting

If you encounter issues:

1. Check the Streamlit Cloud logs for specific error messages
2. Verify that all files have been properly pushed to the repository
3. The app includes automatic fallback mechanisms if the model files are corrupted

## Required Files

- `streamlit_app.py` - Bootstrap entry point
- `app.py` - Main application code
- `fix_model.py` - Model repair functionality
- `requirements.txt` - Package dependencies

## Contact

For issues or questions, please create an issue in this repository. 