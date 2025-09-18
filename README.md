# MLOps Assignment 1: CIFAR-10 Model Comparison and MLflow Tracking

## Project Description
This project demonstrates a complete MLOps pipeline for image classification using a synthetic CIFAR-10 dataset. The workflow covers data preprocessing, model training, experiment tracking, model comparison, artifact logging, and model registration using MLflow. Three machine learning models are compared: Multi-layer Perceptron (Neural Network), Random Forest, and Support Vector Machine (SVM). The best performing model is registered in the MLflow Model Registry.

**Key Features:**
- Synthetic CIFAR-10 dataset (1000 samples, 10 classes)
- Data preprocessing and visualization
- Training and evaluation of three models (MLP, Random Forest, SVM)
- Comprehensive experiment tracking with MLflow (metrics, parameters, artifacts)
- Model artifact and confusion matrix logging
- Model registration in MLflow Model Registry
- Reproducible and well-documented notebook

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone https://github.com/hafizqaim/MLOPS-Assignment-1.git
cd MLOPS-Assignment-1/src
```

### 2. Create and Activate a Python Environment
It is recommended to use a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r ..\requirements.txt
```

### 4. Run the Notebook
Open `Assignment-1 Code.ipynb` in VS Code or Jupyter and run all cells sequentially.

### 5. Start MLflow UI
In a new terminal, from the `src` directory:
```powershell
mlflow ui
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 6. Explore Results
- Compare model runs, metrics, and artifacts in the MLflow UI
- Download model files and confusion matrices
- View registered models in the Model Registry

## Notes
- All models and artifacts are saved in the `/models` directory
- The notebook is self-contained and reproducible
- For any issues, ensure you are running MLflow UI from the `src` directory where the `mlruns` folder is created
- This is a university assignment project and should not be used for commercial purposes.

---