import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os

def process_timestamps(df):
    """
    Converts timestamp column to datetime format and extracts the year.
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['year'] = df['Timestamp'].dt.to_period("Y")  # Convert to yearly period
    return df

def count_messages_over_time(df):
    """
    Counts the number of messages over time (grouped by year).
    """
    return df.groupby('year').size()

def plot_time_series(df):
    """
    Processes data, runs the time-series analysis, and plots the number of social media posts over time.
    """
    df = process_timestamps(df)
    date_counts = count_messages_over_time(df)

    if date_counts.empty:
        st.warning("⚠️ No data available for time-series analysis.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(date_counts.index.astype(str), date_counts.values, marker='o', linestyle='-')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Number of Social Media Posts Over Time (Yearly)")
    ax.set_xticklabels(date_counts.index.astype(str), rotation=45, ha='right')
    ax.grid()
    st.pyplot(fig)

def plot_monthly_time_series(df, selected_year):
    """
    Plots number of social media posts for each month in the selected year.
    """
    df = process_timestamps(df)
    df = df[df['Timestamp'].dt.year == selected_year]

    if df.empty:
        st.warning("⚠️ No data available for the selected year.")
        return

    df['month'] = df['Timestamp'].dt.month
    monthly_counts = df.groupby('month').size()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_counts.index, monthly_counts.values, marker='o')
    ax.set_title(f"Monthly Social Media Posts in {selected_year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Posts")
    ax.set_xticks(range(1, 13))
    ax.grid(True)
    st.pyplot(fig)

def run_time_series_analysis(preprocessed_file):
    """
    Loads preprocessed data and runs time-series analysis before classification.
    """
    if os.path.exists(preprocessed_file):
        df = pd.read_csv(preprocessed_file)
        plot_time_series(df)  # Ensures processing before plotting
        available_years = sorted(df['Timestamp'].dropna().dt.year.astype(int).unique())
        selected_year = st.selectbox("📅 Select a Year for Monthly Analysis", available_years)

        if st.button("Confirm Year Selection"):
            plot_monthly_time_series(df, selected_year)
    else:
        st.warning("⚠️ No preprocessed data found. Please run preprocessing first.")
