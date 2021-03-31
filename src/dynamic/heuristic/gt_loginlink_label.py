

from src.util.chrome import *
from src.credential_classifier.bit_pytorch.utils import read_xml
import os
import pandas as pd
import time
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException

legitimate_urls = list(pd.read_csv('./datasets/alexa.csv', header=None).iloc[:, 0])
write_txt_path = './datasets/gt_loginurl_for600.txt'

# with open(write_txt_path, 'w+') as f:
#     f.write('folder\turl\n')

for url in tqdm(legitimate_urls):
    domain_name = url.split('//')[-1]
    if domain_name in open(write_txt_path).read():
        continue
    if domain_name + '.xml' in os.listdir('D:/ruofan/git_space/phishpedia/benchmark/600_legitimate'):

        # get url
        orig_url = url
        try:
            print("getting url")
            driver.set_page_load_timeout(30)
            driver.get(orig_url)
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
            time.sleep(1)
        except TimeoutException as e:
            print(str(e))
            continue
        except Exception as e:
            print(str(e))
            print("no alert")

        _, gt_coords = read_xml(os.path.join('D:/ruofan/git_space/phishpedia/benchmark/600_legitimate', url.split('//')[-1] + '.xml'))
        for bbox in gt_coords:
            x1, y1, x2, y2 = bbox
            center = ((x1+x2)/2, (y1+y2)/2)
            click_point(center[0], center[1])
            current_url = driver.current_url
            if current_url != orig_url:
                with open(write_txt_path, 'a+') as f:
                    f.write(domain_name+'\t'+current_url+'\n')

            try:
                print("getting url")
                driver.set_page_load_timeout(30)
                driver.get(orig_url)
                alert_msg = driver.switch_to.alert.text
                driver.switch_to.alert.dismiss()
                time.sleep(1)
            except TimeoutException as e:
                print(str(e))
                continue
            except Exception as e:
                print(str(e))
                print("no alert")

