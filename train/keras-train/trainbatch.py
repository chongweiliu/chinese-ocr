# -*- coding: utf-8 -*-
import model
import train
import dataset

import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler

import numpy as np

model,basemodel = model.get_model(height=train.imgH, nclass=train.nclass)
import os
modelPath = '../pretrain-models/keras.hdf5'
if os.path.exists(modelPath):
       basemodel.load_weights(modelPath)
        
batchSize = 32
train_loader = torch.utils.data.DataLoader(
    train.train_dataset, batch_size=batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(train.workers),
    collate_fn=dataset.alignCollate(imgH=train.imgH, imgW=train.imgW, keep_ratio=train.keep_ratio))

testSize = 16
#print test_dataset[0]
test_loader = torch.utils.data.DataLoader(
    train.test_dataset, batch_size=testSize,
    shuffle=True, num_workers=int(train.workers)    )

j = 0
crrentLoss = 1000
loss = 1000
interval  = 50
for i in range(3):
    for X,Y in train_loader:
                X = X.numpy()
                X = X.reshape((-1,train.imgH,train.imgW,1))
                Y = np.array(Y)
                
                Length = int(train.imgW/4)-2
                batch = X.shape[0]
                X,Y = [X, Y, np.ones(batch)*Length, np.ones(batch)*train.n_len], np.ones(batch)
                model.train_on_batch( X,Y)  
                if j%interval==0 :
                   X,Y  =  next(iter(test_loader))
                   X = X.numpy()
                   X = X.reshape((-1,train.imgH,train.imgW,1))
                   Y = Y.numpy()
                   Y = np.array(Y)
                   batch = X.shape[0]
                   X,Y = [X, Y, np.ones(batch)*Length, np.ones(batch)*train.n_len], np.ones(batch)
                   
                   crrentLoss = model.evaluate(X,Y)
                   print("step:{},loss:{},crrentLoss:{}".format(j,loss,crrentLoss))
                   if crrentLoss<loss:
                        loss = crrentLoss
                        path = 'save_model/model{}.h5'.format(loss)
                        print( "save model:".format(path))
                        basemodel.save(path)

                j+=1
                
