import pandas as pd

def read_data():
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = read_data()
    print(f'shape df_train: {train_df.shape}')
    print(f'shape df_test: {test_df.shape}')


