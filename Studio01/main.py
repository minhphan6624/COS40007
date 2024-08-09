import pandas as pd

def main():
    df = pd.read_csv('data/wdbc.csv', header=None)

    rows, cols = df.shape

    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")

    #Column types
    column_types = df.dtypes
    print(f"Column types: {column_types}")

    duplicate_removed = df.drop_duplicates()
    

if __name__ == "__main__":
    main()