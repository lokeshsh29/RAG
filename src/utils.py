import os
import sys
import dill
import pandas as pd
import numpy as np
import scipy.sparse as sp

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object_sparse(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        sp.save_npz(file_path, obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object_sparse(file_path):
    try:
        return sp.load_npz(file_path)
    except Exception as e:
        raise CustomException(e, sys)