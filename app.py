# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Iris Dataset EDA", layout="wide")

# Title
st.title("🌼 Iris Dataset - Exploratory Data Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Iris CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Data Preview")
    st.dataframe(df.head())

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop unnecessary columns
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Summary stats
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.drop(columns=['Species']).corr(), annot=True, cmap='coolwarm', ax=ax1)
    st.pyplot(fig1)

    # Boxplot
    st.subheader("📦 Boxplot - Petal Length by Species")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Species', y='PetalLengthCm', data=df, ax=ax2)
    st.pyplot(fig2)

    # Pairplot
    st.subheader("🔁 Pairplot of All Features")
    st.text("Note: May take a few seconds to load")
    pairplot_fig = sns.pairplot(df, hue='Species')
    st.pyplot(pairplot_fig)
    
    # Scatter plot
    st.subheader("🌱 Sepal Length vs Sepal Width")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df, ax=ax4)
    st.pyplot(fig4)

    # Insights
    st.subheader("🧠 My Takeaways")
    st.markdown("""
    - Setosa is super easy to identify — its petals are way smaller than the others.  
    - Petal length and width are strongly related — they rise together.  
    - Sepal width is kinda random and not very helpful in spotting differences.  
    - Versicolor and Virginica look similar, but Virginica usually has longer petals.  
    - Petal size overall gives the best clues about the species.
    """)

else:
    st.warning("👆 Please upload the `iris.csv` file to start exploring.")

