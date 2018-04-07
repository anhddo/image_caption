import pickle
import os
import matplotlib.pyplot as plt
import math

class TrainLog:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.epoch = 0

    def load(self, path):
        if not os.path.exists(path):
            return
        with open(path , 'rb') as f:
            obj = pickle.load(f) 
            self.__dict__.update(obj.__dict__)

    def save(self, path):
        with open(path , 'wb') as f:
            pickle.dump(self, f) 
