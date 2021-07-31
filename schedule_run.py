
import time
import datetime
import sys
from datetime import datetime, timedelta, time
import argparse
# from phishintention_main import *
from phishpedia_main import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':

    while True:
        date = '2021-07-28'
        # date = datetime.today().strftime('%Y-%m-%d')
        print('Today is:', date)
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', "--folder",
                            default='E:\\screenshots_rf\\{}'.format(date),
                            help='Input folder path to parse')
        parser.add_argument('-r', "--results", default=date + '.txt',
                            help='Input results file name')
        args = parser.parse_args()
        print(args)
        # runit(args.folder, args.results) # if running phishintention
        runit_pedia(args.folder, args.results) # if running phishpedia
        print('Process finish')

