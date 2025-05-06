# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the healthcare dataset
df = pd.read_csv("healthcare_data.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values and data types
print(df.info())
print(df.isnull().sum())

# Summary statistics of the dataset
print(df.describe())

# Visualize the distribution of key variables
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Diabetes', y='BMI', data=df, palette='Set3')
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('Diabetes')
plt.ylabel('BMI')
plt.show()

# Handle categorical variables
df['Gender'] = df['Gender'].astype('category')
df['Diabetes'] = df['Diabetes'].astype('category')

# One-Hot Encoding for Smoking_Status
df = pd.get_dummies(df, columns=['Smoking_Status'], drop_first=True)

# Label Encoding for binary categorical variables
df['Gender'] = df['Gender'].cat.codes
df['Diabetes'] = df['Diabetes'].cat.codes

# Display the transformed dataset
print(df.head())

# Advanced Visualization: Interactive scatter plot using Plotly
fig = px.scatter(df, x='Age', y='BMI', color='Diabetes',
                 title='Age vs BMI by Diabetes Status',
                 labels={'Age': 'Age', 'BMI': 'BMI', 'Diabetes': 'Diabetes Status'})
fig.show()

# Geospatial Analysis (if location data is available)
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Diabetes",
                            mapbox_style="carto-positron", zoom=3,
                            title="Geospatial Distribution of Diabetes")
    fig.show()
