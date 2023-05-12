import streamlit as st

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("breast-cancer.csv")

# Set up the web app
st.set_page_config(page_title="Breast Cancer Analysis and Classification Web App", page_icon=":woman-health-worker:",
                   layout="wide")

# Title and description
st.title("Breast Cancer Analysis and Classification App")
st.markdown("This app analyzes breast cancer data and predicts the diagnosis (M = malignant, B = benign).")

# Sidebar for user input
st.sidebar.title("User Input")
st.sidebar.markdown("Select the parameters to analyze and predict:")

analysis_options = st.sidebar.multiselect("Select parameters for analysis:",
                                          ["radius_mean", "texture_mean", "concavity_mean", "area_mean", "diagnosis"],
                                          ["radius_mean"])


# EDA for selected options
if "diagnosis" in analysis_options:
    # 1. How is the diagnosis distributed?
    st.header("Distribution of Diagnosis")
    sns.countplot(x="diagnosis", data=df)
    plt.title("Distribution of diagnosis")
    st.pyplot()

    # PDA of Diagnosis
    st.header("Probability Density Analysis of Diagnosis")
    fig, ax = plt.subplots()
    for diagnosis in ["M", "B"]:
        sns.kdeplot(data=df[df["diagnosis"] == diagnosis], x="area_mean", label=diagnosis, ax=ax)
    plt.title("Probability Density Analysis of diagnosis")
    st.pyplot()

if "radius_mean" in analysis_options:
    # 2. What is the distribution of the mean radius?
    st.header("Distribution of Mean Radius")
    sns.histplot(df["radius_mean"], kde=True)
    st.pyplot()

    # PDA of mean radius
    st.header("Probability Density Analysis of Mean Radius")
    sns.kdeplot(x="radius_mean", data=df, shade=True)
    plt.title("Probability Density Analysis of mean radius")
    st.pyplot()

if "radius_mean" in analysis_options and "texture_mean" in analysis_options:
    # 3. Is there a correlation between mean radius and mean texture?
    st.header("Correlation between Mean Radius and Mean Texture")
    sns.scatterplot(x="radius_mean", y="texture_mean", data=df)
    st.pyplot()

    # PDA of correlation between mean radius and mean texture
    st.header("Probability Density Analysis of Mean Radius and Mean Texture correlation")
    sns.kdeplot(x="radius_mean", y="texture_mean", data=df, shade=True)
    plt.title("Probability Density Analysis of mean radius and mean texture correlation")
    st.pyplot()

if "concavity_mean" in analysis_options:
    # 4. How does the diagnosis vary with respect to mean concavity?
    st.header("Diagnosis with respect to Mean Concavity")
    sns.boxplot(x="diagnosis", y="concavity_mean", data=df)
    plt.title("Variation of diagnosis with respect to mean concavity")
    st.pyplot()

    # PDA of diagnosis vary with respect to mean concavity
    st.header("Probability Density Analysis of Diagnosis with respect to Mean Concavity")
    sns.kdeplot(x="concavity_mean", hue="diagnosis", data=df, shade=True)
    plt.title("Probability Density Analysis of diagnosis with respect to mean concavity")
    st.pyplot()

if "area_mean" in analysis_options:
    # 5. How does the area of the tumor vary with respect to the diagnosis?
    st.header("Tumor area with respect to Diagnosis")
    sns.violinplot(x="diagnosis", y="area_mean", data=df)
    plt.title("Variation of tumor area with respect to diagnosis")
    st.pyplot()

    # PDA of area of the tumor vary with respect to the diagnosis
    st.header("Probability Density Analysis of Tumor area with respect to Diagnosis")
    sns.kdeplot(x="area_mean", hue="diagnosis", data=df, shade=True)
    plt.title("Probability Density Analysis of tumor area with respect to diagnosis")
    st.pyplot()


# Classification
# Prepare the data
X = df.drop(["id", "diagnosis"], axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy score
st.write(f"Accuracy: {accuracy}")
