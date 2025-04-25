import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def render_anomalies(detected_df):
    anomalies_only = detected_df[detected_df['is_anomaly'] == True].copy()

    if anomalies_only.empty:
        st.success("✅ No anomalies detected with the current filters.")
        return

    # Add severity levels
    score_max = anomalies_only['anomaly_score'].max()
    score_min = anomalies_only['anomaly_score'].min()
    score_range = score_max - score_min

    def categorize_severity(score):
        if score > score_min + 0.66 * score_range:
            return "High"
        elif score > score_min + 0.33 * score_range:
            return "Medium"
        else:
            return "Low"

    anomalies_only['severity'] = anomalies_only['anomaly_score'].apply(categorize_severity)

    # Sidebar filters
    st.sidebar.markdown("### Filter Anomalies")
    vendor_options = anomalies_only['vendor'].unique().tolist()
    account_options = anomalies_only['account'].unique().tolist()

    selected_vendor = st.sidebar.selectbox("Vendor", ["All"] + vendor_options)
    selected_account = st.sidebar.selectbox("Account", ["All"] + account_options)

    if 'posting_date' in anomalies_only.columns:
        anomalies_only['posting_date'] = pd.to_datetime(anomalies_only['posting_date'], errors='coerce')
        min_date = anomalies_only['posting_date'].min().date()
        max_date = anomalies_only['posting_date'].max().date()
        date_range = st.sidebar.date_input("Date range", [min_date, max_date])

        if len(date_range) == 2:
            start_date, end_date = date_range
            anomalies_only = anomalies_only[
                (anomalies_only['posting_date'].dt.date >= start_date) &
                (anomalies_only['posting_date'].dt.date <= end_date)
            ]

    if selected_vendor != "All":
        anomalies_only = anomalies_only[anomalies_only['vendor'] == selected_vendor]
    if selected_account != "All":
        anomalies_only = anomalies_only[anomalies_only['account'] == selected_account]

    # Color-coded table
    def highlight_severity(row):
        if row['severity'] == "High":
            return ['background-color: #ff9999'] * len(row)
        elif row['severity'] == "Medium":
            return ['background-color: #fff599'] * len(row)
        else:
            return ['background-color: #ccffcc'] * len(row)

    st.write("🎯 Filtered Anomalies")
    styled_df = anomalies_only.style.apply(highlight_severity, axis=1)
    st.dataframe(styled_df)

    # Download CSV
    csv_anomalies = anomalies_only.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Filtered Anomalies as CSV",
        data=csv_anomalies,
        file_name="filtered_anomalies.csv",
        mime="text/csv"
    )

    # Timeline plot
    if 'posting_date' in anomalies_only.columns:
        st.subheader("📈 Anomalies Over Time")
        fig, ax = plt.subplots()
        anomalies_only.set_index("posting_date") \
            .resample("W")["anomaly_score"].count() \
            .plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of Anomalies")
        ax.set_title("Weekly Anomaly Frequency")
        st.pyplot(fig)
