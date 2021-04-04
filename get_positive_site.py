
import os
import shutil
import pandas as pd

def save_pos_site(result_txt, source_folder, target_folder):
    df = pd.read_table(result_txt)
    df_pos = df.loc[df['phish'] == 1]
    print('Number of reported positive: {}'.format(len(df_pos)))
    if len(df_pos) == 0:
        return
    os.makedirs(target_folder, exist_ok=True)
    for folder in list(df_pos['folder']):
        try:
            shutil.copytree(os.path.join(source_folder, folder),
                        os.path.join(target_folder, folder))
        except FileExistsError:
            continue


if __name__ == '__main__':
    # save_pos_site('./2021-04-02_pedia.txt', 'E:\\screenshots_rf\\2021-04-02', './datasets/PhishDiscovery/Phishpedia/2021-04-02')

    save_pos_site('./2021-04-02.txt', 'E:\\screenshots_rf\\2021-04-02', './datasets/PhishDiscovery/PhishIntention/2021-04-02')