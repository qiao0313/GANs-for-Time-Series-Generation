import pandas as pd


def extract_start_point(data_path, start_end_xy_path):
    df = pd.read_csv(data_path)
    df_action_id = df['action_id']
    action_id = list(set(df_action_id.values.tolist()))

    with open(start_xy_path, 'w') as f:
        f.write('action_id,start_x,start_y,end_x,end_y\n')
        for act_id in action_id:
            df_tmp = df.loc[df['action_id'] == act_id]
            f.write(str(act_id) + ',' + str(df_tmp['x'].values.tolist()[0]) + ',' + str(df_tmp['y'].values.tolist()[0]) +
                    ',' + str(df_tmp['x'].values.tolist()[-1]) + ',' + str(df_tmp['y'].values.tolist()[-1]) + '\n')


if __name__ == '__main__':
    data_path = 'test_data.csv'
    start_xy_path = 'test_start_end_xy.csv'
    extract_start_point(data_path, start_xy_path)
