from calculate.utils import Uility
from object.objectHelper import ObjectHelper
import glob
import numpy as np
from tqdm import tqdm
import json

class JsonClass(Uility):
    def __init__(self, file_path):
        super().__init__()
        files = glob.glob(file_path)
        self.file_list = self.loadPath(files)

    def loadPath(self, filepaths):
        sorted_list = self.sort_filenames_by_number(filepaths, criteria='Test_')
        return sorted_list

    def loadJsonData(self, filepaths):
        print("--load json data--")

        json_list = []
        for path in tqdm(filepaths):
            with open(path, "r") as js:
                data = json.load(js)
                json_list.append( data )    
        return json_list
