
from phishintention.phishintention_main import *
import time
import datetime
import sys
from datetime import datetime, timedelta, time
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':

    result = subprocess.run("conda run -n myenv pip show phishintention | grep Location | awk '{print $2}'", capture_output=True, shell=True)
    package_location = result.stdout.decode('utf-8').strip()

    AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, \
        SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(f"{package_location}/phishintention/configs.yaml",
                                                                            device='cuda')
    print('Number of protected logos = ', len(LOGO_FEATS)) # (3064, )

    date = datetime.today().strftime('%Y-%m-%d')
    print('Today is:', date)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder",
                        default='phishintention/datasets/test_sites',
                        help='Input folder path to parse')
    parser.add_argument('-r', "--results", default=date + '.txt',
                        help='Input results file name')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--no_repeat', action='store_true')
    args = parser.parse_args()
    print(args)
    runit(args.folder, args.results, AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL,
          SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH) # if running phishintention
    print('Process finish')

