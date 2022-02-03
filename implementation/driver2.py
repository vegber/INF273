import multiprocessing as mp
import time
from numpy import mean
from utils_code.pdp_utils import *
import os


def file_list():
    return sorted((os.listdir('../utils_code/pdp_utils/data/pd_problem/')))


def extract_values(filename: str) -> tuple:
    vals = [s for s in filename[:-4].split("_")]
    vals = [int(i) for i in vals if i.isdigit()]
    return vals[0], vals[1]


def file_data():
    file_info = {}
    data_files = file_list()
    # for x in range(len(data_files)):
    #    file_info[x] = []
    print(data_files)


file_data()
