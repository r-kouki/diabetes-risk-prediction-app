import os
import shutil
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def restore_from_backup(file_path):
    """Restore original file from backup if it exists"""
    backup_path = f"{file_path}.bak"
    if os.path.exists(backup_path):
        print(f"Restoring {file_path} from backup...")
        shutil.copy(backup_path, file_path)
        print(f"Successfully restored {file_path}")
        return True
    else:
        print(f"No backup found for {file_path}")
        return False

def check_file(file_path):
    """Check if a file can be loaded correctly"""
    try:
        print(f"Testing {file_path}...")
        data = joblib.load(file_path)
        print(f"Successfully loaded {file_path}")
        
        # Print additional info for debugging
        if file_path == 'calibrated_model.pkl':
            print(f"Model type: {type(data)}")
            if hasattr(data, 'classes_'):
                print(f"Classes: {data.classes_}")
            if hasattr(data, 'estimators_'):
                print(f"Number of estimators: {len(data.estimators_)}")
            elif hasattr(data, 'base_estimator'):
                print(f"Base estimator type: {type(data.base_estimator)}")
                if hasattr(data.base_estimator, 'estimators_'):
                    print(f"Base estimators: {len(data.base_estimator.estimators_)}")
            
            # Try a test prediction
            print("Testing prediction...")
            # Create random test data
            test_data = np.random.rand(1, 38)
            proba = data.predict_proba(test_data)
            print(f"Prediction probabilities: {proba}")
        
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return False

def create_test_case():
    """Create a test case with known features to validate model"""
    try:
        # Load features
        with open('feature_list.json', 'r') as f:
            features = json.load(f)['features']
        
        # Create test data with high risk values
        high_risk_data = {}
        for feature in features:
            if feature == 'GENHLTH':
                high_risk_data[feature] = 5  # Poor health
            elif feature == '_BMI5CAT':
                high_risk_data[feature] = 4  # Obese
            elif feature == '_AGE_G':
                high_risk_data[feature] = 10  # Older age group
            elif feature == 'PHYSHLTH':
                high_risk_data[feature] = 30  # Many days of poor physical health
            elif feature == 'MENTHLTH':
                high_risk_data[feature] = 30  # Many days of poor mental health
            elif feature == 'EXERANY2':
                high_risk_data[feature] = 2  # No exercise
            elif feature in ['CVDINFR4', 'CVDCRHD4', 'CVDSTRK3']:
                high_risk_data[feature] = 1  # Yes to cardiovascular issues
            elif feature == 'BPHIGH6':
                high_risk_data[feature] = 1  # High blood pressure
            elif feature == 'HIGH_RISK':
                high_risk_data[feature] = 1  # High risk (target)
            else:
                high_risk_data[feature] = 1  # Default value
        
        # Create test data with low risk values
        low_risk_data = {}
        for feature in features:
            if feature == 'GENHLTH':
                low_risk_data[feature] = 1  # Excellent health
            elif feature == '_BMI5CAT':
                low_risk_data[feature] = 2  # Normal weight
            elif feature == '_AGE_G':
                low_risk_data[feature] = 1  # Young age group
            elif feature == 'PHYSHLTH':
                low_risk_data[feature] = 0  # No poor physical health days
            elif feature == 'MENTHLTH':
                low_risk_data[feature] = 0  # No poor mental health days
            elif feature == 'EXERANY2':
                low_risk_data[feature] = 1  # Yes to exercise
            elif feature in ['CVDINFR4', 'CVDCRHD4', 'CVDSTRK3']:
                low_risk_data[feature] = 2  # No to cardiovascular issues
            elif feature == 'BPHIGH6':
                low_risk_data[feature] = 2  # No high blood pressure
            elif feature == 'HIGH_RISK':
                low_risk_data[feature] = 0  # Low risk (target)
            else:
                low_risk_data[feature] = 2  # Default value
        
        # Create DataFrame with both test cases
        test_df = pd.DataFrame([high_risk_data, low_risk_data])
        
        # Save test cases
        test_df.to_csv('test_cases.csv', index=False)
        print("Created test cases and saved to test_cases.csv")
        
        return test_df
    except Exception as e:
        print(f"Error creating test cases: {str(e)}")
        return None

def test_model_with_cases(test_df, model, scaler, imputer, features, threshold):
    """Test model with test cases to ensure it works correctly"""
    try:
        # Make predictions on both test cases
        # Ensure features order matches model's expectation
        test_df = test_df[features]
        
        # Apply preprocessing
        imputed_data = imputer.transform(test_df)
        scaled_data = scaler.transform(imputed_data)
        
        # Make predictions
        probas = model.predict_proba(scaled_data)[:, 1]
        predictions = [1 if p >= threshold else 0 for p in probas]
        
        # Print results
        print("\nModel Test Results:")
        print(f"High risk case probability: {probas[0]:.4f} (expected: >0.5)")
        print(f"Low risk case probability: {probas[1]:.4f} (expected: <0.5)")
        print(f"High risk case prediction: {predictions[0]} (expected: 1)")
        print(f"Low risk case prediction: {predictions[1]} (expected: 0)")
        
        # Check if predictions make sense
        if predictions[0] == 1 and predictions[1] == 0 and probas[0] > probas[1]:
            print("✅ Model predictions look reasonable - high risk case has higher probability than low risk case")
        else:
            print("❌ Model predictions may not be reasonable - check if model is working correctly")
        
        return probas, predictions
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return None, None

def main():
    # Files to restore
    files_to_restore = [
        'calibrated_model.pkl',
        'scaler.pkl',
        'imputer.pkl'
    ]
    
    # Restore each file from backup
    restored_count = 0
    for file_path in files_to_restore:
        if restore_from_backup(file_path):
            restored_count += 1
    
    if restored_count == 0:
        print("No files were restored from backup.")
        return
    
    # Check each file
    all_good = True
    for file_path in files_to_restore:
        if os.path.exists(file_path):
            if not check_file(file_path):
                all_good = False
        else:
            print(f"File {file_path} does not exist")
            all_good = False
    
    if all_good:
        print("\nAll files are valid and ready to use.")
        
        # Load components for testing
        model = joblib.load('calibrated_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        
        # Load features
        with open('feature_list.json', 'r') as f:
            features = json.load(f)['features']
        
        # Load threshold
        with open('optimal_threshold.json', 'r') as f:
            threshold = json.load(f)['optimal_threshold']
        
        # Create and test with test cases
        test_df = create_test_case()
        if test_df is not None:
            test_model_with_cases(test_df, model, scaler, imputer, features, threshold)
        
        print("\nYou can now run your Streamlit app: streamlit run app.py")
    else:
        print("\nSome files could not be loaded correctly. Check the error messages above.")

if __name__ == "__main__":
    main() 