import pandas as pd

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import itertools

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove rows with outliers in this column
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def visualize_boxplot(df, column):
    plt.figure(figsize=(6, 4))
    df.boxplot([column])
    plt.title(f'Boxplot for {column}')
    plt.ylabel(column)
    plt.show()

def main():
    df = pd.read_csv("water_potability.csv")

    # ----- Data Cleaning -----

    # -- Handle duplicates
    duplicate_rows = (df.duplicated().sum()) # Check for duplicates

    print(f"Number of duplicate rows: {duplicate_rows}")

    df = df.drop_duplicates() # Drop duplicates if there's any

    print(f"Number of rows after removing duplicates: {df.shape[0]}")

    # Check for outliers using boxplot:
    # for column in df.select_dtypes(include=['float64', 'int64']).columns:
    #     visualize_boxplot(df, column)

    # Remove outliers using IQR
    df = remove_outliers_iqr(df, "ph")
    df = remove_outliers_iqr(df, "Hardness")
    df = remove_outliers_iqr(df, "Solids")
    df = remove_outliers_iqr(df, "Chloramines")
    df = remove_outliers_iqr(df, "Sulfate")
    df = remove_outliers_iqr(df, "Conductivity")
    df = remove_outliers_iqr(df, "Organic_carbon")
    df = remove_outliers_iqr(df, "Trihalomethanes")
    df = remove_outliers_iqr(df, "Turbidity")

    print(f"Number of rows after removing outliers: {df.shape[0]}")

    # -- Handle missing data
    missing_data = df.isna().sum()

    print(missing_data)

    df.to_csv("cleaned_data.csv", index=False)

    # ------------ EDA ------------

    # Summar statistics
    df.describe().to_csv("summary_statistics.csv")

    # Univariate analysis
    cols = [i for i in df.columns if i not in ['Potability']]

    for column in cols:
         # Plot the histogram with KDE
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=column, kde=True, bins=20)
        plt.title(f'Distribution of {column}')
        plt.show()


    # Multi-variate analysis
    # sns.pairplot(df, diag_kind = 'kde', corner = True);
    # plt.show()

    # Correlation matrix via heatmap
    # Check the Correlation
    print(df.corr())



if __name__ == "__main__":
    main()