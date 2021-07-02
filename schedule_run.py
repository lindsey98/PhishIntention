
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
        print('Today is:', datetime.today().strftime('%Y-%m-%d'))

        parser = argparse.ArgumentParser()
        parser.add_argument('-f', "--folder",
                            default='E:\\screenshots_rf\\{}'.format(datetime.today().strftime('%Y-%m-%d')),
                            help='Input folder path to parse')
        parser.add_argument('-r', "--results", default=datetime.today().strftime('%Y-%m-%d') + '.txt',
                            help='Input results file name')
        args = parser.parse_args()
        print(args)
        # runit(args.folder, args.results)
        runit_pedia(args.folder, args.results)
        print('Process finish')

