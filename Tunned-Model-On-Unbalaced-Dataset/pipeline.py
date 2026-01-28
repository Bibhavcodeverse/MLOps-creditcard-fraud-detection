import os
import argparse
from sklearn.model_selection import train_test_split
from data_loader import load_data, validate_data, split_features_target
from features import build_preprocessing_pipeline
from train import train_model, save_model, cross_validate_model
from evaluate import evaluate_model
import joblib

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument('--data_path', type=str, default='data/raw/creditcard.csv', help='Path to dataset')
    args = parser.parse_args()
    
    # 1. Load and Validate Data
    print("Step 1: Loading and Validating Data...")
    df = load_data(args.data_path)
    validate_data(df)
    
    # 2. Split Data
    print("\nStep 2: Splitting Data...")
    X, y = split_features_target(df)
    # Stratified split to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Feature Engineering
    print("\nStep 3: Feature Engineering...")
    pipeline = build_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Save the preprocessing pipeline
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(pipeline, 'models/preprocessor.pkl')

    # 4. Model Training
    print("\nStep 4: Model Training...")
    model = train_model(X_train_transformed, y_train)
    
    # 5. Cross Validation
    # Note: For CV, we should ideally use a Pipeline(preprocessor, model) to avoid leakage.
    # But for simplicity here, we check on transformed training data (acceptable if transformations are not learned heavily, 
    # but RobustScaler fits on data. So strictly speaking we should use Full Pipeline for CV).
    # Let's do it properly for CV:
    from sklearn.pipeline import Pipeline as SklearnPipeline
    full_pipeline = SklearnPipeline([
        ('preprocessor', build_preprocessing_pipeline()),
        ('model', model)
    ])
    cross_validate_model(full_pipeline, X_train, y_train)
    
    # 6. Evaluation
    print("\nStep 6: Evaluation...")
    evaluate_model(model, X_test_transformed, y_test)
    
    # 7. Model Persistence
    save_model(model)
    
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
