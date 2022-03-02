import pandas as pd
import os
from math import floor


def create_equidistant_actions(data_path, save_path):
    df = pd.read_csv(data_path, header=None)
    with open(save_path, 'w') as f:
        for index, row in df.iterrows():
            dx_dy = row.values.tolist()
            start_x = dx_dy[0]
            start_y = dx_dy[1]
            end_x = dx_dy[2]
            end_y = dx_dy[3]
            n = 64
            dx = floor((end_x - start_x) / n)
            dy = floor((end_y - start_y) / n)
            # dx = round((end_x - start_x) / n, 1)
            # dy = round((end_y - start_y) / n, 1)

            dx_list = [dx] * n
            dy_list = [dy] * n

            d_list = []
            for i in range(len(dx_list)):
                d_list.append(dx_list[i])
                d_list.append(dy_list[i])

            d_list = [str(ele) for ele in d_list]
            f.write(','.join(d_list) + '\n')


if __name__ == '__main__':
    train_data_path = '../features/train_raw_features_dx_dy.csv'
    train_save_path = 'train_equidistant.csv'
    create_equidistant_actions(train_data_path, train_save_path)

    test_data_path = '../features/test_raw_features_dx_dy.csv'
    test_save_path = 'test_equidistant.csv'
    create_equidistant_actions(test_data_path, test_save_path)
