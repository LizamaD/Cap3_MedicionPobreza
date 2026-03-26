# GEMINI Project Context: Poverty Measurement Analysis

This document provides a comprehensive overview of the "Cap3_MedicionPobreza" project for the Gemini AI assistant.

## Project Overview

This is a machine learning project focused on poverty analysis. The core objective is to build and train a model that can identify and characterize poverty based on data from the National Survey of Household Income and Expenditure (ENIGH).

The primary methodology involves using a **TensorFlow/Keras Autoencoder**. The model is trained exclusively on data from households classified as "non-poor." The trained autoencoder can then be used to analyze "poor" households by measuring the reconstruction error: a high error suggests that the household's characteristics deviate significantly from the "non-poor" patterns learned by the model.

The project is structured as a Python-based data processing and modeling pipeline.

## Directory and File Structure

- **`/data`**: Contains the raw and processed datasets.
  - `/data/raw`: Location for the initial ENIGH CSV files.
  - `/data/processed`: Destination for the cleaned, merged, and split datasets created by the pipeline.
- **`/src`**: The main Python source code for the project.
  - `pipeline.py`: The central script that orchestrates the entire data processing workflow, from reading raw data to generating the final model-ready datasets.
  - `train_final_model.py`: The script responsible for training the final autoencoder model on the processed "non-poor" dataset and saving the trained model artifacts.
  - `tune_autoencoder.py`: A script for hyperparameter optimization of the autoencoder using the Optuna library.
  - Other `.py` files (`poblacion.py`, `viviendas.py`, etc.): Each module is responsible for cleaning, processing, and feature engineering a specific raw data file from the ENIGH survey.
- **`/notebooks`**: Contains Jupyter notebooks for experimentation and execution.
  - `app.ipynb`: The main notebook used to execute the entire data processing pipeline from start to finish.
  - `experimentos.ipynb`: A notebook for experimentation and preliminary analysis.
- **`/results`**: Intended for storing the output of the model, such as the trained model files, evaluation metrics, and prediction results.
- **`requirements.txt`**: A list of all the Python packages required to run this project.

## Development and Execution Workflow

The project follows a two-stage process: data preparation and model training.

### 1. Development Setup

To set up the environment, install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### 2. Data Processing Pipeline

The data pipeline is orchestrated by `src/pipeline.py` and is typically run via the `notebooks/app.ipynb` notebook. The key steps are:

1.  **Load Raw Data**: The pipeline starts by loading multiple raw CSV files (poblacion, viviendas, hogares, etc.) from the `/data/raw` directory.
2.  **Process Modules**: Each dataset is processed by its corresponding module in `src` (e.g., `process_poblacion` in `src/poblacion.py`). This involves cleaning, feature engineering, and data type correction.
3.  **Create Master Table**: The processed dataframes are merged into a single, person-level "master table."
4.  **Impute Missing Values**: Missing data in the master table is imputed using appropriate strategies (median for numerical, `__MISSING__` category for categorical).
5.  **Prepare for Model**: The imputed data is prepared for the autoencoder. This includes one-hot encoding categorical variables and applying a low-variance feature selection filter.
6.  **Split by Poverty**: The final dataset is split into `pobres.csv` and `no_pobres.csv` based on the poverty indicator, and the files are saved to `/data/processed`.

### 3. Model Training

The model training is handled by `src/train_final_model.py`:

1.  **Load Data**: The `no_pobres.csv` dataset is loaded from `/data/processed`.
2.  **Scale Features**: The features are scaled to a [0, 1] range using `MinMaxScaler`. The scaler is saved so the same transformation can be applied later.
3.  **Build and Train**: The autoencoder is built using the best hyperparameters (likely found using `tune_autoencoder.py`) and trained on the scaled "non-poor" data.
4.  **Save Artifacts**: The final trained autoencoder model (`.keras`) and the data scaler (`.joblib`) are saved to the `/results` directory.

To run the full process, you would typically execute the cells in `notebooks/app.ipynb` first, and then run the training script:

```bash
# Step 1: Execute the data pipeline (e.g., by running the app.ipynb notebook)
# This will generate the 'no_pobres.csv' file in 'data/processed/'

# Step 2: Train the final model
python src/train_final_model.py
```
