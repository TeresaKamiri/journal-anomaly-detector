# Journal Anomaly Detector

This project aims to detect anomalies in journal entries combining autoencoder-based unsupervised learning with explainability (SHAP) and benchmarking (Random Forest, KNN). The system identifies unusual patterns in journal entries, helping auditors spot potential errors, fraud, or irregularities.

## Project Structure

```bash
journal-anomaly-detector/
├── data/                                # Data files
│   ├── synthetic_versions/            # Synthetic Raw journal entries dataset
│   │   ├── synthetic_labeled_v1.csv   
│   │   ├── synthetic_labeled_v2.csv
│   │   ├── synthetic_labeled_v3.csv
│   │   ├── synthetic_labeled_v4.csv
│   │   ├── synthetic_labeled_v5.csv
│   ├── journal_entries.csv             # Raw journal entries dataset
├── anomalies/                           # Anomalies detection results
│   ├── anomalies_found.csv             # Detected anomalies
├── evalutations/                       # Benchmark evaluation results
├── models/                              # Saved models and preprocessing artifacts
│   ├── autoencoder_model.keras        # Trained autoencoder model (Keras format)
│   ├── feature_names.pkl              # List of features used in the model
│   ├── scaler.pkl                     # Scaler used for data preprocessing
├── notebooks/                           # Jupyter notebooks for exploratory analysis and preprocessing
│   ├── eda_preprocessing.py           # Data exploration and preprocessing code
├── src/                                 # Source code for anomaly detection and model training
│   ├── benchmark_models.py            # Script for benchmarking model.  runs KNN/RF/Autoencoder comparison
│   ├── detect_anomalies.py            # Script to detect anomalies in journal entries
│   ├── explain_anomalies.py           # Script for explaining detected anomalies. uses SHAP for interpretation
│   ├── generate_synthetic_journals.py # Script to generate synthetic journal entries for testing
│   ├── preprocess.py                  # Script to preprocess and real datasets on retraining
│   ├── render_anomalies.py            # Script to render anomaly data on dashboard
│   ├── train_autoencoder.py           # Script to train the autoencoder model
│   ├── train_on_upload.py             # Script to retrain on uploaded data
│   └── utils.py                       # Utility functions used across scripts
├── dashboard/                           # Streamlit frontend for anomaly detection
│   └── app.py                         # Streamlit app for visualizing anomalies
├── app.log                             # Log file to track app execution
├── step_by_step.md                    # Step-by-step guide for project setup and usage
```

## Project Overview

This project uses an autoencoder-based deep learning model with explainability (SHAP) and benchmarking (Random Forest, KNN) to detect anomalies in journal entries. The model is trained on historical journal data, and the detected anomalies are flagged for further investigation.

### Main Features
- **Anomaly Detection:** Detects irregular patterns or outliers in journal entries using an autoencoder.
- **Explainability:** Uses explainability techniques to help auditors understand why certain entries are flagged as anomalies.
- **Synthetic Data Generation:** Generates synthetic journal entries for testing and model validation.
- **Optional: retrain model on uploaded data**
- **SHAP-based interpretability (force + summary plots)**
- **Benchmark:** comparison (Autoencoder vs. KNN vs. Random Forest), Supervised evaluation (if labels exist)
- **Streamlit Dashboard:** A frontend for visualizing and interacting with anomaly detection results.

## Requirements

You need the following libraries and tools to run this project:

- Python 3.10
- TensorFlow or Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Streamlit
- Other dependencies listed in `requirements.txt` (if provided)

## Setup Instructions

### Step 1: Create and Activate a Virtual Environment

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/journal-anomaly-detector.git
   ```

2. Navigate to the project directory:

   ```bash
   cd journal-anomaly-detector
   ```

3. Create a Python virtual environment:

   ```bash
   python3.10 -m venv ml-venv
   ```

4. Activate the virtual environment:

   - **On Windows:**
     ```bash
     .\ml-venv\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```bash
     source ml-venv/bin/activate
     ```

### Step 2: Install the Required Dependencies

With the virtual environment activated, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your 🧪 Datasets

* Place labeled/unlabeled `.csv` files in `data/`
* If your dataset includes a `label` column (0 = normal, 1 = fraud), supervised evaluation + benchmarking will be activated

If needed, run the preprocessing and exploratory analysis steps in `notebooks/eda_preprocessing.py`.

## Usage 

### Generate Synthetic Data

You can generate synthetic journal entries using:

```bash
python src/generate_synthetic_journals.py
```

### Training the Model

1. Run the `train_autoencoder.py` script to train the autoencoder model:

   ```bash
   python src/train_autoencoder.py
   ```

2. After training, the model will be saved in the `models/` directory as `autoencoder_model.keras`.

### Detecting Anomalies

To detect anomalies in journal entries, run the `detect_anomalies.py` script:

```bash
python src/detect_anomalies.py
```

This will generate a CSV file (`anomalies_found.csv`) in the `anomalies/` folder containing all detected anomalies.

### Explaining Anomalies

To get explanations for why anomalies were detected, run:

```bash
python src/explain_anomalies.py
```
### Benchmark Model

KNN/RF/Autoencoder comparison, run:

```bash
python src/explain_anomalies.py
```

### Streamlit Dashboard

To start the Streamlit dashboard, use the following command:

```bash
streamlit run dashboard/app.py
```

This will launch a web application for visualizing detected anomalies, benchmarks, interacting with the dataset or retraining new datasets.

#### ✅ Hybrid Strategy
Default: Uses pretrained model on synthetic baseline
Optional: Retrain model on uploaded real data with checkbox toggle

#### 📦 Outputs

* `evaluation/*.csv`: benchmark results
* `models/`: trained models + artifacts
* Downloadable anomaly report via UI

## Contributing

Feel free to fork this repository and submit pull requests for any improvements or fixes. If you have suggestions or issues, open an issue, and i will get back to you.
