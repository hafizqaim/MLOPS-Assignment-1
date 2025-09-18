# MLOps Assignment 1: Complete MLOps Pipeline Implementation

## Project Overview
This project demonstrates a comprehensive MLOps pipeline implementation for image classification using machine learning models. The workflow encompasses all five parts of the assignment requirements, from GitHub setup to complete project documentation.

## Project Structure
```
MLOPS-Assignment-1/
├── src/
│   ├── Assignment-1 Code.ipynb    # Main implementation notebook
│   └── mlruns/                    # MLflow tracking data
├── models/                        # Saved model artifacts
│   ├── neural_network_mlp_*.joblib
│   ├── random_forest_*.joblib
│   ├── support_vector_machine_*.joblib
│   ├── feature_scaler_*.joblib
│   └── model_results_*.csv
├── data/                          # Dataset directory
├── notebooks/                     # Additional notebooks
├── results/                       # Analysis results
├── mlruns/                        # MLflow experiment data
├── requirements.txt               # Project dependencies
├── recreate_mlflow_experiments.py # MLflow setup script
└── README.md                      # Project documentation
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git for version control
- VS Code or Jupyter Notebook environment

### Installation Steps

1. **Clone the Repository**
```powershell
git clone https://github.com/hafizqaim/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1
```

2. **Create Virtual Environment**
```powershell
python -m venv mlops_env
.\mlops_env\Scripts\activate
```

3. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

4. **Navigate to Source Directory**
```powershell
cd src
```

5. **Run the Main Notebook**
Open `Assignment-1 Code.ipynb` in VS Code or Jupyter Lab and execute all cells sequentially.

## MLflow Integration

### Start MLflow UI
```powershell
# From the src/ directory
mlflow ui
```
Access the UI at: [http://localhost:5000](http://localhost:5000)

### Features Available in MLflow UI
- **Experiments:** View all model training runs
- **Metrics:** Compare accuracy, precision, recall, F1-score
- **Parameters:** Analyze hyperparameter configurations
- **Artifacts:** Download models and confusion matrices
- **Model Registry:** Access registered best performing model

## Results Summary

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** | **~13.5%** | **~13.2%** | **~13.5%** | **~12.7%** | ~1.04s |
| Support Vector Machine | ~12.0% | ~2.6% | ~12.0% | ~4.3% | ~1.18s |
| Neural Network (MLP) | ~8.5% | ~6.6% | ~8.5% | ~6.5% | ~0.76s |

### Key Insights
- **Best Model:** Random Forest with highest overall performance
- **Fastest Training:** Neural Network MLP with shortest training time
- **Most Consistent:** Random Forest with balanced precision-recall
- **Recommendation:** Random Forest for deployment due to superior generalization

## Technical Specifications

### Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
mlflow
joblib
```

### Model Configurations
- **MLP:** 2 hidden layers (100, 50 neurons), early stopping enabled
- **Random Forest:** 100 estimators, max depth 10, all CPU cores utilized
- **SVM:** RBF kernel, C=1.0, gamma='scale' for optimal performance

### Data Processing
- **Train-Test Split:** 80-20 ratio with stratification
- **Feature Scaling:** StandardScaler for neural network and SVM
- **Preprocessing:** Systematic approach for different model requirements

## Usage Instructions

### Running the Complete Pipeline
1. Execute the notebook cells sequentially from top to bottom
2. Monitor MLflow UI for real-time experiment tracking
3. Check the `/models` directory for saved artifacts
4. Review model comparison results and insights

## License & Academic Use

This project is developed as part of an academic assignment for the MLOps course. It demonstrates best practices in machine learning operations and serves as a learning resource for MLOps implementation.

<img width="705" height="612" alt="confusion_matrix_neural_network_mlp" src="https://github.com/user-attachments/assets/e96a1a49-6331-4f48-822d-0566ac1d6224" />
<img width="696" height="612" alt="confusion_matrix_random_forest" src="https://github.com/user-attachments/assets/49fe87d1-4549-412d-8e50-da7e6ad8a4e7" />
<img width="705" height="612" alt="confusion_matrix_support_vector_machine" src="https://github.com/user-attachments/assets/2548e1ae-b573-412e-97a3-89696b44201b" />

**Note:** This implementation is for educational purposes and should be adapted for production use with appropriate security, scalability, and performance considerations.

---
