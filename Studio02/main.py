import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 

# Convert the strength to a categorical value
def convert_strength(strength):
    if strength < 20:
        return 1
    elif strength >= 20 and strength < 30:
        return 2
    elif strength >= 30 and strength < 40:
        return 3
    elif strength >= 40 and strength < 50:
        return 4
    elif strength >= 50:
        return 5


# Plot the distribution of the concrete strength classes
def plot_distribution(df):
    # Plot the distribution as a bar chart
    class_counts = df["converted_strength"].value_counts().sort_index() # Count the number of instances in each class and sort by index

    plt.figure(figsize=(10,6))  # 10 units wide by 6 units tall
    class_counts.plot(kind='bar', color='skyblue')

    # Add labels and title
    plt.title("Distribution of Concrete Strength Classes")
    plt.xlabel("Strength Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation = 0) # Rotate the x-axis labels to be horizontal

    plt.show() # display the plot

# Convert the age to a categorical value
def convert_age(df):
    
    #Count the unique values in the age column,
    unique_ages = np.sort(df["age"].unique())
    unique_age_counts = len(unique_ages)

    # print(f"There are {unique_age_counts} unique age values: {unique_ages}")

    age_mapping = {age: i+1 for i, age in enumerate(unique_ages)}

    df["converted_age"] = df["age"].map(age_mapping)

    print("converted age")
    print(df["converted_age"].head()) 

    df[['cement', 'water', 'superplastic', 'converted_age', 'converted_strength']].to_csv('selected_converted_concrete.csv', index=False)

# Normalize the 7 features
def normalize_features(df):

    features_to_normalize = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']
    scaler = MinMaxScaler()

    normalized_values = scaler.fit_transform(df[features_to_normalize])

    new_column_names = [f"{col}_normalized" for col in features_to_normalize]

    for feature in features_to_normalize:
        df[f"{feature}_normalized"] = normalized_values[:, features_to_normalize.index(feature)]

    print("Normalized features")
    print(df.head())

    df[new_column_names + ["converted_age", "converted_strength"]].to_csv('normalized_concrete.csv', index=False)


def create_composite_features(df):
    df['cement_slag'] = df[['cement_normalized', 'slag_normalized']].cov().iloc[0, 1]  # Covariance between 'cement' and 'slag'
    df['cement_ash'] = df[['cement_normalized', 'ash_normalized']].cov().iloc[0, 1]    # Covariance between 'cement' and 'ash'
    df['water_fineagg'] = df[['water_normalized', 'fineagg_normalized']].cov().iloc[0, 1]  # Covariance between 'water' and 'fineagg'
    df['ash_superplastic'] = df[['ash_normalized', 'superplastic_normalized']].cov().iloc[0, 1]  # Covariance between 'ash' and 'superplastic'

    print("Composite features")
    print(df.head())
    
    new_features = ["converted_age", 'cement_normalized', 'slag_normalized', 'ash_normalized', 'water_normalized', 'superplastic_normalized', 'coarseagg_normalized', 'fineagg_normalized'] + ["cement_slag", "cement_ash", "water_fineagg", "ash_superplastic", "converted_strength"]

    df[new_features].to_csv('features_concrete.csv', index = False)
    print(df[new_features].head())

def filter_features(df):

    columns_to_keep = [
        'cement_normalized', 'water_normalized', 'superplastic_normalized', "converted_age",
        "cement_slag", "cement_ash", "water_fineagg", "ash_superplastic",
        "converted_strength"
    ]

    df_filtered = df[columns_to_keep]
    print("Filtered features")
    print(df_filtered.head())

    df_filtered.to_csv('selected_features_concrete.csv', index = False)

def main():
    # Read original data
    df = pd.read_csv("concrete.csv")

    # Convert the strength to a categorical value
    df["converted_strength"] = df["strength"].apply(convert_strength)
    df.to_csv('converted_concrete.csv' ,index = False)
    
    # plot_distribution(df)

    convert_age(df)

    normalize_features(df)

    create_composite_features(df)

    filter_features(df)


if __name__ == "__main__":
    main()
    