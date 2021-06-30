
import schedule
import time
import datetime
import sys
from datetime import datetime, timedelta, time
import argparse
from phishintention_main import *

if __name__ == '__main__':

    print('Today is:', datetime.today().strftime('%Y-%m-%d'))
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", default='E:\\screenshots_rf\\{}'.format(datetime.today().strftime('%Y-%m-%d')), help='Input folder path to parse')
    parser.add_argument('-r', "--results", default=datetime.today().strftime('%Y-%m-%d')+'.txt', help='Input results file name')
    args = parser.parse_args()
    print(args)

    schedule.every(5).hours.until(timedelta(hours=5)).do(runit(args)) # run it every 5 hours, kill it after running 5 hours


