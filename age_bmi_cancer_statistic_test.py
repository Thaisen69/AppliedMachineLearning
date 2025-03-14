import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load data
df = pd.read_csv("dataR2.csv")

# Rename Classification values for better readability
df['Classification'] = df['Classification'].map({1: "Healthy", 2: "Cancer"})

# Boxplot: Age vs Cancer
def plot_boxplot(feature, title):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Classification', y=feature, data=df, palette=["#1f77b4", "#ff7f0e"])
    plt.title(title)
    plt.show()

plot_boxplot("Age", "Age Distribution by Cancer Status")
plot_boxplot("BMI", "BMI Distribution by Cancer Status")

# KDE Plots (Density estimation)
def plot_kde(feature, title):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[df['Classification'] == "Healthy"][feature], label="Healthy", shade=True)
    sns.kdeplot(df[df['Classification'] == "Cancer"][feature], label="Cancer", shade=True)
    plt.title(title)
    plt.legend()
    plt.show()

plot_kde("Age", "Age Distribution Density by Cancer Status")
plot_kde("BMI", "BMI Distribution Density by Cancer Status")

# T-tests
age_healthy = df[df['Classification'] == "Healthy"]["Age"]
age_cancer = df[df['Classification'] == "Cancer"]["Age"]
bmi_healthy = df[df['Classification'] == "Healthy"]["BMI"]
bmi_cancer = df[df['Classification'] == "Cancer"]["BMI"]

age_ttest = ttest_ind(age_healthy, age_cancer)
bmi_ttest = ttest_ind(bmi_healthy, bmi_cancer)

print("T-test for Age: p-value =", age_ttest.pvalue)
print("T-test for BMI: p-value =", bmi_ttest.pvalue)
