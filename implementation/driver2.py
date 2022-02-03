import multiprocessing as mp
import time
from numpy import mean
from utils_code.pdp_utils import *
import os
from natsort import natsorted, ns
path = '../utils_code/pdp_utils/data/pd_problem/'


def file_list():
    return natsorted(os.listdir(path), key=lambda y:y.lower())


def extract_values(filename: str) -> tuple:
    vals = [s for s in filename[:-4].split("_")]
    vals = [int(i) for i in vals if i.isdigit()]
    return vals[0], vals[1]


