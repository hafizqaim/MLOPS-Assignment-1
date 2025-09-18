# Recreate MLflow experiments with our trained models
import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from datetime import datetime

# Set the working directory to the project directory
os.chdir(r"F:\University\Assignments\7th Semester\MLOps\MLOPS-Assignment-1")

# Setup MLflow experiment tracking
experiment_name = "CIFAR10_Model_Comparison"
mlflow.set_experiment(experiment_name)
mlflow.set_tracking_uri("file:./mlruns")

print(f"MLflow experiment '{experiment_name}' is set up!")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Load the saved models and results
models_dir = "./models"
models_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and 'model' in f]

# Define model results (from our previous training)
results = {
    'Neural Network (MLP)': {
        'accuracy': 0.0850,
        'precision': 0.0660,
        'recall': 0.0850,
        'f1_score': 0.0654,
        'training_time': 0.7994
    },
    'Random Forest': {
        'accuracy': 0.1350,
        'precision': 0.1318,
        'recall': 0.1350,
        'f1_score': 0.1274,
        'training_time': 1.0297
    },
    'Support Vector Machine': {
        'accuracy': 0.1200,
        'precision': 0.0259,
        'recall': 0.1200,
        'f1_score': 0.0425,
        'training_time': 1.4506
    }
}

print("Recreating MLflow experiments...")
print("=" * 50)

for model_name, metrics in results.items():
    print(f"Logging {model_name}...")
    
    # Start MLflow run for this model
    with mlflow.start_run(run_name=f"{model_name}_run"):
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("training_time", metrics['training_time'])
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset", "CIFAR10_synthetic")
        mlflow.log_param("train_samples", 800)
        mlflow.log_param("test_samples", 200)
        mlflow.log_param("n_features", 3072)
        mlflow.log_param("n_classes", 10)
        
        # Log model if file exists
        model_file = None
        for file in models_files:
            if model_name.lower().replace(' ', '_').replace('(', '').replace(')', '') in file.lower():
                model_file = os.path.join(models_dir, file)
                break
        
        if model_file and os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                mlflow.sklearn.log_model(model, f"{model_name.replace(' ', '_').lower()}_model")
                print(f"  ✓ Model artifact logged: {model_file}")
            except Exception as e:
                print(f"  ⚠ Could not log model artifact: {e}")
        
        # Add tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "CIFAR10_synthetic")
        mlflow.set_tag("framework", "scikit-learn")
        
        print(f"  ✓ {model_name} run logged successfully!")

print("\n" + "=" * 50)
print("All experiments recreated successfully!")

# Get the experiment and show summary
experiment = mlflow.get_experiment_by_name("CIFAR10_Model_Comparison")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print(f"\nExperiment Summary:")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Total Runs: {len(runs)}")

if len(runs) > 0:
    print(f"\nRuns Overview:")
    for _, run in runs.iterrows():
        print(f"- {run['tags.model_type']}: Accuracy = {run['metrics.accuracy']:.4f}")
    
    # Register the best model
    best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
    best_run_id = best_run['run_id']
    best_model_name = best_run['tags.model_type']
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_run['metrics.accuracy']:.4f})")
    
    try:
        # Register the best model
        model_name = "CIFAR10_Best_Classifier"
        model_version = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/{best_model_name.replace(' ', '_').lower()}_model",
            name=model_name
        )
        print(f"✅ Best model registered as '{model_name}' version {model_version.version}")
    except Exception as e:
        print(f"⚠ Could not register model: {e}")

print(f"\n🎉 MLflow setup complete! Run 'mlflow ui' to view the experiments.")