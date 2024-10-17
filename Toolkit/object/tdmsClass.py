from calculate.utils import Uility
from .objectHelper import ObjectHelper
import glob
import numpy as np
from nptdms import TdmsFile
from tqdm import tqdm

# tdms를 object화 하기
class TdmsObjectClass():
    def __init__(self, data):
        self.data = data
    def __call__():
        return 0
    
    def getChannelData(self):
        return 0
    def checkTdmsObject(self, data):
        return isinstance(data, self)
    def checkTdms(self, data):
        return isinstance(data, TdmsFile)


class TdmsClass(Uility):
    def __init__(self, file_path):
        super().__init__()
        files = glob.glob(file_path)
        self.file_list = self.loadPath(files)

    def loadPath(self, filepaths):
        sorted_list = self.sort_filenames_by_number(filepaths, criteria='test_')
        return sorted_list

    def loadTdmsData(self, filepaths):
        print("--load tdms data--")
        tdms_list = []
        for path in tqdm(filepaths):
            tdms_list.append( TdmsFile(path) )
        return tdms_list

    def getChannelData(self, tdms_datas, group, channel):
        if ObjectHelper.isIterable(tdms_datas) is False:
            tdms_datas = [tdms_datas]

        lists = []
        error_count = 0
        for tdms in tdms_datas:
            try:
                lists.append( tdms[str(group)][str(channel)][:] )
            except Exception as e:
                lists.append( [] )
                print(error_count, " => ", e)
                if error_count == 0:
                    self.showGroupinfo(tdms)
                error_count = error_count + 1
        return lists

    def showGroupinfo(self, tdms_file):
        group_list = tdms_file.groups()
        print("\n Groups in TDMS:", group_list)
        return group_list

    def showChannelInfo(self, tdms_file):
        group_list = tdms_file.groups()
        for group in group_list:
            print("\n Channels in Group:", tdms_file[group.name].channels())