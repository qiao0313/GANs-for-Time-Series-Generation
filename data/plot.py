import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(point_x, point_y, save_path, action_id):
    print('Start generating {} trajectory'.format(action_id))
    plt.cla()
    plt.title('trajectory_{}'.format(action_id))
    plt.plot(point_x, point_y, marker=".", markersize='6')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(save_path, 'trajectory_{}.png'.format(action_id)))


def plot_main(plot_data_path, save_path):
    df = pd.read_csv(plot_data_path)
    df_action_id = df['action_id']
    action_id = list(set(df_action_id.values.tolist()))
    print(len(action_id))
    for act_id in action_id:
        df_tmp = df.loc[df['action_id'] == act_id]
        point_x = df_tmp['x'].values.tolist()
        point_y = df_tmp['y'].values.tolist()
        plot(point_x, point_y, save_path, act_id)


if __name__ == '__main__':
    plot_data_path = 'test_data.csv'
    save_path = 'images'
    plot_main(plot_data_path, save_path)
