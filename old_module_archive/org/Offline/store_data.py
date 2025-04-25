import json
import pickle
import bz2
import os
from os import path
import queue


class StoreData:
    def __init__(self,STORAGE_LOCATION,SENSOR_ADDRESS,FILE_NAME,FILE_FORMAT):
        self.STORAGE_LOCATION=STORAGE_LOCATION
        self.SENSOR_ADDRESS=SENSOR_ADDRESS
        self.FILE_NAME=FILE_NAME
        self.FILE_FORMAT=FILE_FORMAT
        self.FULL_PATH=self.STORAGE_LOCATION+"\\"+self.SENSOR_ADDRESS+"\\"+self.FILE_NAME+"."+self.FILE_FORMAT
    def check_path(self):
        if(path.exists(self.STORAGE_LOCATION+"\\"+self.SENSOR_ADDRESS)!=True):    
            os.mkdir(self.STORAGE_LOCATION+"\\"+self.SENSOR_ADDRESS)
        return
    def store_as_json(self,payload):        
        self.check_path()
        
        f = open(self.FULL_PATH, "a")#FILE_FORMAT=json
        f.write(json.dumps(payload))
        f.close()
        return
    def store_as_pickle(payload):
        self.check_path()
        
        geeky_file = open(self.FULL_PATH, 'wb')#FILE_FORMAT=pickle
        pickle.dump(payload, geeky_file)    
        geeky_file.close()
        return
    def store_as_pickle_bz2(self, payload):
        self.check_path()
        
        ofile = bz2.BZ2File(self.FULL_PATH,'wb') #FILE_FORMAT=pickle.bz2
        pickle.dump(payload,ofile)
        ofile.close()     
        return
    def convert_pickle_bz2_2_json(self):
        ifile = bz2.BZ2File(self.FULL_PATH,'rb')
        pickle_data = pickle.load(ifile)
        ifile.close()
        json_data=json.loads(json.dumps(pickle_data, default=str))
        return json_data
    def convert_pickle_2_json(self):
        with open(self.FULL_PATH, 'rb') as infile:
          obj = pickle.load(infile)
        json_obj = json.loads(json.dumps(obj, default=str))
        return json_obj
    def remove_from_disk(self):
        if os.path.exists(self.FULL_PATH):
            os.remove(self.FULL_PATH)
        else:
            print("The "+self.FULL_PATH+" does not exist!")
        return
