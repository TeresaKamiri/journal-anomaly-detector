# Journal Anomaly Detector

This project aims to detect anomalies in journal entries using an autoencoder-based model. The system identifies unusual patterns in journal entries, helping auditors spot potential errors, fraud, or irregularities.

## Project Structure

```bash
journal-anomaly-detector/
├── data/                                # Data files
│   ├── journal_entries.csv             # Raw journal entries dataset
├── anomalies/                           # Anomalies detection results
│   ├── anomalies_found.csv             # Detected anomalies
├── models/                              # Saved models and preprocessing artifacts
│   ├── autoencoder_model.h5           # Trained autoencoder model (H5 format)
│   ├── autoencoder_model.keras        # Trained autoencoder model (Keras format)
│   ├── feature_names.pkl              # List of features used in the model
│   ├── scaler.pkl                     # Scaler used for data preprocessing
├── notebooks/                           # Jupyter notebooks for exploratory analysis and preprocessing
│   ├── eda_preprocessing.py           # Data exploration and preprocessing code
├── src/                                 # Source code for anomaly detection and model training
│   ├── detect_anomalies.py            # Script to detect anomalies in journal entries
│   ├── explain_anomalies.py           # Script for explaining detected anomalies
│   ├── generate_synthetic_journals.py # Script to generate synthetic journal entries for testing
│   ├── train_autoencoder.py           # Script to train the autoencoder model
│   └── utils.py                       # Utility functions used across scripts
├── dashboard/                           # Streamlit frontend for anomaly detection
│   └── app.py                         # Streamlit app for visualizing anomalies
├── app.log                             # Log file to track app execution
├── step_by_step.md                    # Step-by-step guide for project setup and usage
```

## Project Overview

This project uses an autoencoder-based deep learning model to detect anomalies in journal entries. The model is trained on historical journal data, and the detected anomalies are flagged for further investigation.

### Main Features
- **Anomaly Detection:** Detects irregular patterns or outliers in journal entries using an autoencoder.
- **Explainability:** Uses explainability techniques to help auditors understand why certain entries are flagged as anomalies.
- **Synthetic Data Generation:** Generates synthetic journal entries for testing and model validation.
- **Streamlit Dashboard:** A frontend for visualizing and interacting with anomaly detection results.

## Requirements

You need the following libraries and tools to run this project:

- Python 3.x
- TensorFlow or Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Streamlit
- Other dependencies listed in `requirements.txt` (if provided)

## Setup Instructions

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/journal-anomaly-detector.git
   ```

2. Navigate to the project directory:

   ```bash
   cd journal-anomaly-detector
   ```

3. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your dataset. Place the `journal_entries.csv` file in the `data/` directory.

5. If needed, run the preprocessing and exploratory analysis steps in `notebooks/eda_preprocessing.py`.

## Usage 
Refer to the step_by_step.md file

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

2. After training, the model will be saved in the `models/` directory as `autoencoder_model.h5` and `autoencoder_model.keras`.

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

### Generate Synthetic Data

You can generate synthetic journal entries using:

```bash
python src/generate_synthetic_journals.py
```

### Streamlit Dashboard

To start the Streamlit dashboard, use the following command:

```bash
streamlit run dashboard/app.py
```

This will launch a web application for visualizing detected anomalies and interacting with the dataset.

## Contributing

Feel free to fork this repository and submit pull requests for any improvements or fixes. If you have suggestions or issues, open an issue, and i will get back to you.
