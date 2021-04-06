
import os
import shutil
import pandas as pd
import numpy as np

def save_pos_site(result_txt, source_folder, target_folder):
    df = pd.read_table(result_txt)
    df_pos = df.loc[df['phish'] == 1]
    # df = [x.strip().split('\t') for x in open(result_txt).readlines()]
    # df_pos = [x for x in df if x[2] == '1']
    print('Number of reported positive: {}'.format(len(df_pos)))

    if len(df_pos) == 0:
        return
    os.makedirs(target_folder, exist_ok=True)
    for folder in list(df_pos['folder']):
    # for folder in [x[0] for x in df_pos]:
        try:
            shutil.copytree(os.path.join(source_folder, folder),
                        os.path.join(target_folder, folder))
        except FileExistsError:
            continue

def get_diff(bigger_folder, smaller_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for folder in os.listdir(bigger_folder):
        if folder not in os.listdir(smaller_folder):
            try:
                shutil.copytree(os.path.join(bigger_folder, folder),
                                os.path.join(target_folder, folder))
            except FileExistsError:
                continue

def get_runtime(result_txt):
    df = pd.read_table(result_txt)
    runtime_list = list(df['runtime (layout detector|siamese|crp classifier|login finder)'])
    breakdown = [list(map(float, x.split('|'))) for x in runtime_list]
    breakdown_df = pd.DataFrame(breakdown)
    breakdown_df.columns = ['layout', 'siamese', 'crp', 'dynamic']
    breakdown_df = breakdown_df.replace(0, np.NaN)
    print(breakdown_df.mean())
    print(breakdown_df.median())
    print(breakdown_df.min())
    print(breakdown_df.max())
    # print(breakdown_df)

def get_total_runtime(result_txt):
    df = pd.read_table(result_txt)
    runtime = list(df['total_runtime'])
    print(np.mean(runtime))
    print(np.median(runtime))
    print(np.min(runtime))
    print(np.max(runtime))


if __name__ == '__main__':
    date = '2021-04-06'
    # for phishpedia
    save_pos_site('./{}_pedia.txt'.format(date), 'E:\\screenshots_rf\\{}'.format(date),
                  './datasets/PhishDiscovery/Phishpedia/{}'.format(date))

    # for phishintention
    save_pos_site('./{}.txt'.format(date), 'E:\\screenshots_rf\\{}'.format(date),
                  './datasets/PhishDiscovery/PhishIntention/{}'.format(date))

    # get phishpedia - phishintention
    get_diff('./datasets/PhishDiscovery/Phishpedia/{}'.format(date), './datasets/PhishDiscovery/PhishIntention/{}'.format(date),
             './datasets/PhishDiscovery/pedia_intention_diff/{}'.format(date))

    get_total_runtime('./{}.txt'.format(date))