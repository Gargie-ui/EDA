# Exploratory Data Analysis (EDA) Project
# Dataset Example: Titanic Dataset

# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2. Load Dataset
# =========================
# Replace with your dataset path
df = pd.read_csv("titanic.csv")

# =========================
# 3. Basic Information
# =========================
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 4. Statistical Summary
# =========================
print("\nStatistical Summary:")
print(df.describe())

# =========================
# 5. Data Cleaning
# =========================

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# =========================
# 6. Univariate Analysis
# =========================

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender Count
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# =========================
# 7. Bivariate Analysis
# =========================

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Passenger Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()

# Age vs Fare
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title("Age vs Fare")
plt.show()

# =========================
# 8. Correlation Heatmap
# =========================

# Select only numeric columns
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 9. Key Insights
# =========================

print("\nKey Insights:")
print("1. Female passengers had a higher survival rate.")
print("2. First-class passengers survived more often.")
print("3. Fare and Passenger Class show correlation with survival.")
print("4. Younger passengers had slightly better survival chances.")

# =========================
# 10. Save Cleaned Dataset
# =========================
df.to_csv("cleaned_titanic.csv", index=False)

print("\nEDA Completed Successfully!")
