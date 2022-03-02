import os
import pandas as pd
import numpy as np


def file2action(file_path, file_out, max_len, feature_type):
    df = pd.read_csv(file_path)
    action_id = df['action_id']

    lst_id = []
    for index in action_id:
        if index not in lst_id:
            lst_id.append(index)
    # print(lst_id)

    with open(file_out, 'w') as f:
        for act_id in lst_id:
            action = df.loc[df['action_id'] == act_id]
            features = action2rawfeatures(action, max_len, feature_type)
            features = [str(feature) for feature in features]
            features.append(str(1))
            f.write(','.join(features) + '\n')


def action2rawfeatures(action, max_len, feature_type):
    action = action.drop(columns=['action_id'])
    features = []
    first_action = action.iloc[[0]]
    last_action = action.iloc[[-1]]
    start_x, start_y = first_action['x'].values.tolist()[0], first_action['y'].values.tolist()[0]
    end_x, end_y = last_action['x'].values.tolist()[0], last_action['y'].values.tolist()[0]
    end_x, end_y = end_x - start_x, end_y - start_y
    features.append(0)
    features.append(0)
    features.append(end_x)
    features.append(end_y)
    df = action.diff()
    first_row = df.index[0]
    df = df.drop(first_row)
    dx = df['x']
    dy = df['y']
    dt = df['time_stamp']

    dx = dx.values.tolist()
    dy = dy.values.tolist()
    dt = dt.values.tolist()

    if len(dx) < max_len:
        for i in range(max_len - len(dx)):
            dx.append(0)
            dy.append(0)
            dt.append(0)
    else:
        dx = dx[0:max_len]
        dy = dy[0:max_len]
        dt = dt[0:max_len]
    for i in range(len(dx)):
        features.append(dx[i])
        features.append(dy[i])
        if feature_type == 'dx_dy_dt':
            features.append(dt[i])

    return features


if __name__ == '__main__':
    train_file_path = 'train_data.csv'
    train_file_out = '../features/train_raw_features_dx_dy.csv'
    max_len = 64
    feature_type = 'dx_dy'
    file2action(train_file_path, train_file_out, max_len, feature_type)

    test_file_path = 'test_data.csv'
    test_file_out = '../features/test_raw_features_dx_dy.csv'
    file2action(test_file_path, test_file_out, max_len, feature_type)
