import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("concrete.csv")

    for index, row in df.iterrows():
        print(row)



if __name__ == "__main__":
    main()