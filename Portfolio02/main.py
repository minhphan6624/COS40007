import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove rows with outliers in this column
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def main():
    df = pd.read_csv("water_potability.csv")

    print(df.head())

    print(df.shape)

    # ----- Data Cleaning -----

    # -- Handle duplicates
    duplicate_rows = (df.duplicated().sum()) # Check for duplicates

    print(f"Number of duplicate rows: {duplicate_rows}")

    df = df.drop_duplicates() # Drop duplicates if there's any

    print(f"Number of rows after removing duplicates: {df.shape[0]}")

    # Handle outliers:
    # Loop through each numerical column and create a box plot
    

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

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        df.boxplot([column])
        plt.title(f'Boxplot for {column}')
        plt.ylabel(column)
        plt.show()

    
    # Handle missing data
    missing_data = df.isna().sum()

    print(missing_data)





if __name__ == "__main__":
    main()