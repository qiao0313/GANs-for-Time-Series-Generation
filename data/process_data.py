import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_data(raw_data_path, new_data_path):
    data = np.load(raw_data_path, allow_pickle=True)
    with open(new_data_path, 'w', encoding='utf-8') as f:
        f.write('action_id,x,y,time_stamp\n')
        action_id = 0
        for i in range(len(data['human50'])):
                for tra in data['human50'][i]:  # each: [[], [], ...]
                    if len(tra) <= 64:
                        for j in range(len(tra)):
                            f.write(str(action_id) + ',' + str(tra[j][1]) + ',' + str(tra[j][2]) + ',' + str(tra[j][3]) + '\n')
                        action_id += 1
        print(action_id)


if __name__ == '__main__':
    raw_data_path = 'human50.npz'
    new_data_path = 'data.csv'
    process_data(raw_data_path, new_data_path)
