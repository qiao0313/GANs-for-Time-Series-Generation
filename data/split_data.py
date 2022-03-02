import pandas as pd


def split_data(data_path, train_path, test_path):
    df = pd.read_csv(data_path)
    action_id = df['action_id']

    lst_id = list(set(action_id))
    train_len = int(len(lst_id) * 0.8)
    lst_train = lst_id[:train_len]
    lst_test = lst_id[train_len:]
    with open(train_path, 'w') as f1:
        f1.write('action_id,x,y,time_stamp\n')
        for index in lst_train:
            action = df.loc[df['action_id'] == index]
            action.to_csv(f1, sep=',', index=0, header=0)

    with open(test_path, 'w') as f2:
        f2.write('action_id,x,y,time_stamp\n')
        for index in lst_test:
            action = df.loc[df['action_id'] == index]
            action.to_csv(f2, sep=',', index=0, header=0)


if __name__ == '__main__':
    data_path = 'data.csv'
    train_path = 'train_data.csv'
    test_path = 'test_data.csv'
    split_data(data_path, train_path, test_path)

