from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F



class classify(nn.Module):
    def __init__(self, in_channels,rh1, ch1, sent):
        super(classify, self).__init__()
        
        num_classes = 2
        
        self.p3 = nn.Sequential(
                            nn.Conv1d(in_channels = in_channels, out_channels = 128, kernel_size = 7),
                            nn.ReLU()
                            )
        self.r3 = nn.Sequential(
                            nn.Conv1d(in_channels = in_channels, out_channels = 64, kernel_size = 5),
                            nn.ReLU()
                            )
    
        self.s1 = nn.Linear(4*sent,rh1)

        self.l1 = nn.Linear(64+128, ch1)
        
        self.l3 = nn.Linear(2*100,100)
        self.l4 = nn.Linear(100, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.7)
        
    def forward(self, paper, review, sentiment):  
        batch_size = paper.shape[0]
        out_p3 = self.p3(paper)
        out_p3 = F.max_pool1d(out_p3, out_p3.shape[2])

        out_r3 = self.r3(review)
        out_r3 = F.max_pool1d(out_r3, out_r3.shape[2])    #out_p/r shape = (batch_size, #filters, 1)
        
        out = torch.cat((out_p3, out_r3), dim = 1)         #out shape = (batch_size, num_filters*kernels, 1)
        
        r = self.s1(sentiment.reshape(batch_size, -1))
        r = self.dropout(r)
        
        out = self.l1(out.reshape(batch_size, -1))
        out = self.dropout(out)

        out = self.l3(torch.cat((out, r), dim = 1))
        out = self.dropout(out)
        out = self.relu(out)
        out = self.l4(out)
    
        return out,r