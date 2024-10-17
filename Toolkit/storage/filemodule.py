import pickle
import csv
from datetime import datetime
import os
from collections.abc import Iterable

def saveAsCSV(content, file_path="./auto_generated.csv"):
    with open(file_path, 'w', newline='') as f:
        write = csv.writer(f)
        for row in content:
            if is_NestedIterable(row):
                write.writerow(row)
            else:
                write.writerow([row])

def is_NestedIterable(target):
    if isIterable(target) and all(isIterable(target) for sublist in target):
        return True
    else:
        return False

def isIterable(target):
    return True if isinstance(target, Iterable) else False

def saveAsBinary(content, file_path="./auto_generated.pickle"):
    with open(file_path,"wb") as fw:
        pickle.dump(content, fw)

def splitFileName(string):
    if string.find(".") > -1:
        name, ext = os.path.splitext(string)
    else:
        name = string
        ext = "pickle" #default
    return name, ext

def saveToFile(content, file_name, prefix="", format=None, time_stamp=True):
    file_name, ext = splitFileName(file_name)
    format = format if format is not None else ext

    file_full_path =\
          f"{prefix}/{file_name}{datetime.now().strftime('%m_%d_%H_%M_%S') if time_stamp is True else ''}.{format}"
    saver_dict = {"csv":saveAsCSV, "pickle":saveAsBinary, 
                  ".csv":saveAsCSV, ".pickle":saveAsBinary}

    if saver_dict.get(format) is not None:
        saver = saver_dict.get(format)
        saver(content, file_full_path)
    return os.path.exists( file_full_path )