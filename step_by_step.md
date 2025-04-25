### **Program Flow Overview**

1. **Data Preparation and Model Training**  
   - You **start by training the autoencoder model** if you haven't already done so. This process involves:
     - Preprocessing the journal entries.
     - Training an autoencoder model to reconstruct the input data.
     - Saving the trained model (`autoencoder_model.h5`) and the scaler (`scaler.pkl`) for later use in anomaly detection.
   
2. **Anomaly Detection**  
   - Once the model is trained, the next step is **detecting anomalies** in the journal entries.
   - This is done in the `detect_anomalies.py` script, where you load the saved model and scaler, transform the data, and compute the reconstruction error (MSE). If the reconstruction error exceeds a certain threshold, the data point is labeled as an anomaly.

3. **Explaining Anomalies**  
   - For each detected anomaly, you can use **SHAP** to explain why the anomaly was detected.
   - The `explain_anomalies.py` script loads the autoencoder and SHAP explainer to visualize and explain the model's decision-making process.

4. **Frontend (Streamlit App)**  
   - **Streamlit** is used to create an interactive frontend where users can upload their own journal entries CSV file, run anomaly detection, and visualize the results.
   - The app loads the CSV, runs anomaly detection, and shows a table of detected anomalies, anomaly scores, and visualizations like histograms and SHAP explanations.

---

### **Steps to Run the Program**

Here's a step-by-step guide for running the entire pipeline:

### 1. **Train the Autoencoder Model** (`train_autoencoder.py`)

   Before Model Training:

   Exploratory Data Analysis (EDA): Use the notebook to explore the dataset. You can run some basic analyses like:

   Viewing basic statistics.

   Checking for missing values.

   Plotting distributions, correlation heatmaps, and relationships between variables.

   Preprocessing the Data: After you understand the data, you can preprocess it (handle missing values, normalize/standardize data, encode categorical features) in the notebook.

   After Model Training (for verification):

   You could use the notebook after model training to quickly verify if the data is correctly preprocessed and whether any additional steps are needed before running anomaly detection or explaining anomalies.

If you haven't already trained the model, follow these steps:

- **Run the model training**:
  - In your terminal, navigate to the `src` directory and run the script:
    ```bash
    python train_autoencoder.py
    ```
  - This will:
    - Load the journal entries data from `data/journal_entries.csv`. or the explored dataset `data/processed_journal_entries.csv`
    - Preprocess the data and train the autoencoder.
    - Save the trained model as `models/autoencoder_model.h5` and the scaler as `models/scaler.pkl`.

- **Model files will be saved** in the `models/` folder, which will be used later for anomaly detection and explanation.

### 2. **Perform Anomaly Detection** (`detect_anomalies.py`)

Once the model is trained, you can start **detecting anomalies**:

- **Create a script to detect anomalies** by running the `detect_anomalies` function from `detect_anomalies.py`.
- Example of running it:
  ```python
  from src.detect_anomalies import detect_anomalies

  data_path = "../data/journal_entries.csv"
  detected_df, threshold = detect_anomalies(data_path)

  # The detected_df contains anomaly scores and the "is_anomaly" flag
  print(detected_df.head())
  ```

This function will:
1. Read the `journal_entries.csv` file.
2. Preprocess the data and scale it using the saved scaler.
3. Use the trained autoencoder to detect anomalies by comparing the reconstruction error against a threshold.

The output will be a dataframe (`detected_df`) with columns:
- `anomaly_score`: The reconstruction error.
- `is_anomaly`: A boolean flag indicating whether it's an anomaly or not.

### 3. **Explain Anomalies** (`explain_anomalies.py`)

Once anomalies are detected, you can **explain them using SHAP**:

- **Run the explanation**:
  ```python
  from src.explain_anomalies import explain

  shap_values = explain(detected_df)

  # Visualize SHAP explanations (requires SHAP library)
  shap.initjs()
  shap.force_plot(shap_values.base_values[0], shap_values.values[0], detected_df.iloc[0])
  ```

This will give you an interactive SHAP plot that shows how each feature in the data contributes to the anomaly detection decision.

### 4. **Streamlit App (Frontend)** (`app.py`)

Finally, you can use **Streamlit** to create a user-friendly interface for anomaly detection:

- **Install Streamlit** if you haven’t already:
  ```bash
  pip install streamlit
  ```

- **Run the Streamlit app**:
  - In the terminal, navigate to the `root/` directory and run:
    ```bash
    streamlit run dashboard/app.py
    ```

This will start the Streamlit app and open it in your browser. You can:
- Upload a CSV file (journal entries).
- Specify the threshold for anomaly detection (default is 95th percentile).
- View the detected anomalies, anomaly scores, and visualizations.

### **App Flow (Streamlit)**:
1. **Upload a CSV file**: Users can upload their journal entries file.
2. **Anomaly Detection**: The app will automatically process the data, run anomaly detection, and display results.
3. **Visualization**: The app shows a table of anomalies, histograms of anomaly scores, and SHAP explanations for individual anomalies.

---

### **Directory Structure Recap**:

1. **Training the model**:
   - `train_autoencoder.py`: Trains and saves the autoencoder model and scaler.
   
2. **Anomaly Detection**:
   - `detect_anomalies.py`: Detects anomalies in the journal entries using the trained autoencoder model.

3. **Explaining Anomalies**:
   - `explain_anomalies.py`: Explains why certain data points are flagged as anomalies using SHAP.

4. **Streamlit Frontend**:
   - `app.py`: An interactive interface for anomaly detection using Streamlit.

---

### **End-to-End Example**:

1. **Step 1**: Train the model (if not already trained):
   ```bash
   python src/train_autoencoder.py
   ```

2. **Step 2**: Run the Streamlit app to interact with the system:
   ```bash
   streamlit run dashboard/app.py
   ```

3. **Step 3**: Upload your CSV file in the Streamlit app, and view anomalies and their explanations!
