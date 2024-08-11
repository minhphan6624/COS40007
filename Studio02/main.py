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
def plot_distribution(dataframe):
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

    print(f"There are {unique_age_counts} unique age values: {unique_ages}")

    age_mapping = {age: i+1 for i, age in enumerate(unique_ages)}

    df["converted_age"] = df["age"].map(age_mapping)

    #convert age to categorical
    df["age"] = df["age"].astype("category")

    print(df[['age', 'converted_age']].head())

# Normalize the 7 features
def normalize_features(df):
    features_to_normalize = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']
    scaler = MinMaxScaler()

    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    df.to_csv('normalized_concrete.csv', index = False)
    print(df[features_to_normalize].head())


def main():
    df = pd.read_csv("concrete.csv")

    print(df.head())

    df["converted_strength"] = df["strength"].apply(convert_strength)
    print(df.shape)    

    #Write to a new file
    df.to_csv('converted_concrete.csv' ,index = False)

    #plot_distribution(df)
    
    # -- Activity 2 --
    convert_age(df)

    normalize_features(df)


if __name__ == "__main__":
    main()
    