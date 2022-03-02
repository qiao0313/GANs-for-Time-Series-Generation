import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(x, y, action_id, trajectory_plot_path):
    print('Start generating {} trajectory'.format(action_id))
    plt.cla()
    plt.title('trajectory_{}'.format(action_id))
    # plt.plot(x, y, marker=".", markersize='6')
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(trajectory_plot_path, 'trajectory_{}.png'.format(action_id)))


def plot_trajectory_main(action_id_path, action_data_path, trajectory_points_path, trajectory_plot_path):
    df_action_id = pd.read_csv(action_id_path)
    df = pd.read_csv(action_data_path, header=None)
    action_id = df_action_id['action_id'].values.tolist()
    with open(trajectory_points_path, 'w') as f:
        for index, row in df.iterrows():
            act_id = action_id[index]
            start_x = 0
            start_y = 0
            x = [start_x]
            y = [start_y]
            dx_dy = row
            dx = dx_dy[0::2]
            dy = dx_dy[1::2]

            for i in range(len(dx)):
                x.append(start_x + sum(dx[:i]))
                y.append(start_y + sum(dy[:i]))
            plot_trajectory(x, y, act_id, trajectory_plot_path)

            point_x = [str(i) for i in x]
            point_y = [str(i) for i in y]
            f.write(','.join(point_x) + ',' + ','.join(point_y) + '\n')


if __name__ == '__main__':
    action_id_path = '../data/test_start_end_xy.csv'
    action_data_path = 'test_actions_dx_dy.csv'
    trajectory_points_path = 'trajectory_points_dx_dy.csv'
    trajectory_plot_path = './images/dx_dy'
    plot_trajectory_main(action_id_path, action_data_path, trajectory_points_path, trajectory_plot_path)
