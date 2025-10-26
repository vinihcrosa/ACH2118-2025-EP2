import pandas as pd

def main():
    file_path = './data/ep2-train.csv'
    data = pd.read_csv(
        file_path,
        encoding='ISO-8859-1',
        sep=';',
        decimal=','
        )
    print(data.head())


if __name__ == "__main__":
    main()
