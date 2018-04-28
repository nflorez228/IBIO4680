#!/usr/local/bin/ipython

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net0(nn.Module):
    def __init__(self):
	super(Net0, self).__init__()

    def forward(self, x):
        return F.dropout(x, 0.5, training=self.training)

class Net1(nn.Module):
    def __init__(self):
	super(Net1, self).__init__()
	self.dropout=nn.Dropout()

    def forward(self, x):
        return self.dropout(x)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arange', type=int, default=20, choices=[5,10,15,20])
    parser.add_argument('--net', type=str, default='0', choices=['0','1'])
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    if args.net=='0':
        model = Net0()
    else:
        model = Net1()
        
    if args.test: model.eval()
    else: model.train()
    
    x = torch.arange(args.arange)
    print(model(x).data.numpy().flatten())
    
    
    

