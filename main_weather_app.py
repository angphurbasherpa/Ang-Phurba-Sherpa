import streamlit as st

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
# Load data
df = pd.read_csv("weather_history_bangladesh.csv")

# Set up the web app
st.set_page_config(page_title="Weather Analysis and Classification Web App", page_icon=":partly_sunny:", layout="wide")

# Title and description
st.title("Weather Analysis and Classification App")
st.markdown("This app analyzes weather data and predicts weather History of Bangladesh.")

# Sidebar options for analysis
analysis_options = st.sidebar.multiselect("Select the weather parameters to analyze and predict:",
                                          ["temperature_fahrenheit", "humidity_percentage", "wind_speed_mph", "pressure_in", "avg_temperature"],
                                          ["temperature_fahrenheit", "humidity_percentage"])

# EDA
num_plots = min(5, len(analysis_options))
for i in range(num_plots):
    if analysis_options[i] == "temperature_fahrenheit":
        st.header("Temperature variation over the years")
        df["year"] = pd.DatetimeIndex(df["date"]).year
        fig, ax = plt.subplots()
        sns.lineplot(x="year", y="temperature_fahrenheit", data=df, ax=ax)
        st.pyplot(fig)

        st.header("Probability Density Analysis of Temperature Variation over the Years")
        fig, ax = plt.subplots()
        sns.kdeplot(x="year", hue="temperature_fahrenheit", data=df, shade=True, ax=ax)
        plt.title("Probability Density Analysis of temperature variation over the years")
        st.pyplot(fig)

    elif analysis_options[i] == "humidity_percentage":
        st.header("Distribution of humidity levels")
        fig, ax = plt.subplots()
        sns.histplot(df["humidity_percentage"], kde=True, ax=ax)
        plt.xticks(rotation=90, fontsize=10)
        st.pyplot(fig)

        st.header("Probability Density Analysis of Humidity Levels")
        fig, ax = plt.subplots()
        sns.countplot(x="humidity_percentage", data=df, ax=ax)
        ax.set_title("Probability Density Analysis of humidity levels")
        st.pyplot(fig)


    elif analysis_options[i] == "wind_speed_mph" and "temperature_fahrenheit" in analysis_options:
        st.header("Correlation between wind speed and temperature")
        fig, ax = plt.subplots()
        sns.scatterplot(x="wind_speed_mph", y="temperature_fahrenheit", data=df, ax=ax)
        st.pyplot(fig)

        st.header("Probability Density Analysis of Correlation between Wind Speed and Temperature")
        fig, ax = plt.subplots()
        sns.kdeplot(x="wind_speed_mph", y="temperature_fahrenheit", data=df, shade=True, ax=ax)
        plt.title("Probability Density Analysis of correlation between wind speed and temperature")
        st.pyplot(fig)

    elif analysis_options[i] == "pressure_in" and "temperature_fahrenheit" in analysis_options and "pressure_in" in df.columns:
        st.header("Temperature and pressure variation")
        fig, ax = plt.subplots()
        sns.scatterplot(x="pressure_in", y="temperature_fahrenheit", data=df, ax=ax)
        st.pyplot(fig)

        st.header("Probability Density Analysis of Temperature and Pressure Variation")
        fig, ax = plt.subplots()
        sns.kdeplot(x="pressure_in", y="temperature_fahrenheit", data=df, shade=True, ax=ax)
        plt.title("Probability Density Analysis of temperature and pressure variation")
        st.pyplot(fig)

    elif analysis_options[i] == "avg_temperature" and "humidity_percentage" in analysis_options:
        st.header("Average temperature and humidity during the hottest month of the year")
        df["month"] = pd.DatetimeIndex(df["date"]).month
        max_temp_month = df.groupby("month")["temperature_fahrenheit"].max().idxmax()
        max_temp_month_data = df[df["month"] == max_temp_month]
        avg_temp = max_temp_month_data['temperature_fahrenheit'].mean()
        avg_humidity = max_temp_month_data['humidity_percentage'].mean()

        fig, ax = plt.subplots()
        sns.barplot(x=["avg_temperature", "Average Humidity"], y=[avg_temp, avg_humidity], ax=ax)
        ax.set_ylabel("Percentage/Fahrenheit")
        st.pyplot(fig)

        st.header("Probability Density Analysis of Temperature")
        fig, ax = plt.subplots()
        sns.kdeplot(x="temperature_fahrenheit", data=df, shade=True, ax=ax)
        plt.title("Probability Density Analysis of temperature")
        st.pyplot(fig)




