import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(x, y, index, trajectory_plot_path):
    print('Start generating {} trajectory'.format(index))
    plt.cla()
    plt.title('trajectory_{}'.format(index))
    plt.plot(x, y, marker=".", markersize='6')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(trajectory_plot_path, 'trajectory_{}.png'.format(index)))


def plot_trajectory_main(points_path, trajectory_plot_path):
    df = pd.read_csv(points_path, header=None)
    for index, row in df.iterrows():
        start_x = 0
        start_y = 0
        x = [0]
        y = [0]
        dx_dy = row
        dx = dx_dy[0::2]
        dy = dx_dy[1::2]

        for i in range(len(dx)):
            x.append(start_x + sum(dx[:i]))
            y.append(start_y + sum(dy[:i]))
        plot_trajectory(x, y, index, trajectory_plot_path)


if __name__ == '__main__':
    points_path = 'dx_dy/points_8000.csv'
    trajectory_plot_path = 'images'
    plot_trajectory_main(points_path, trajectory_plot_path)
