import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN2(nn.Module):
    def __init__(self, rh1, ch1):
        super(CNN2, self).__init__()
        
        self.num_filters = 3
        self.kernel_shape = [3,4,5]
        self.kernels = 512
        
#         self.p1 = nn.Sequential(
#                             nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 3),
#                             nn.ReLU()
#                           )
#         self.p2 = nn.Sequential(
#                             nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 4),
#                             nn.ReLU()
#                             )
        self.p3 = nn.Sequential(
                            nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 5),
                            nn.ReLU()
                            )
#         self.r1 = nn.Sequential(
#                             nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 3),
#                             nn.ReLU()
#                             )
#         self.r2 = nn.Sequential(
#                             nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 4),
#                             nn.ReLU()
#                             )
        self.r3 = nn.Sequential(
                            nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 5),
                            nn.ReLU()
                            )
    
        self.s1 = nn.Linear(4*98,rh1)
        self.s2 = nn.Linear(rh1, 1)
#       self.s3 = nn.Linear(rh2, 1)

        self.l1 = nn.Linear(256, ch1)
        self.l2 = nn.Linear(ch1, 1)
        
        self.l3 = nn.Linear(2,1)
#         self.r1 = nn.Linear
#         self.l3 = nn.Linear(192, 48)
#         self.l4 = nn.Linear(48, 12)
#         self.l5 = nn.Linear(12,1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        
    def forward(self, paper, review, sentiment):   #paper = N*512*666, #review = N*512*98
        batch_size = paper.shape[0]
#         review = torch.cat((review, sentiment), dim = 1)
#         out_p1 = self.p1(paper)
#         out_p1 = F.max_pool1d(out_p1, out_p1.shape[2])
#         out_p2 = self.p2(paper)
#         out_p2 = F.max_pool1d(out_p2, out_p2.shape[2])
        #out_p3 = self.p3(paper)
        #out_p3 = F.max_pool1d(out_p3, out_p3.shape[2])
#         out_r1 = self.r1(review)
#         out_r1 = F.max_pool1d(out_r1, out_r1.shape[2])
#         out_r2 = self.r2(review)
#         out_r2 = F.max_pool1d(out_r2, out_r2.shape[2])
        out_r3 = self.r3(review)
        out_r3 = F.max_pool1d(out_r3, out_r3.shape[2])    #out_p/r shape = (batch_size, #filters, 1)
        
#         out_p = torch.cat((out_p1, out_p2, out_p3), dim = 1)
#         out_r = torch.cat((out_r1, out_r2, out_r3), dim = 1)
        #out = torch.cat((out_p3, out_r3), dim = 1)         #out shape = (batch_size, num_filters*kernels, 1)
    	out = out_r3
        r = self.s1(sentiment.view(batch_size, -1))
        r = self.s2(r)
#         r = self.s3(r)
        
        out = self.l1(out.view(batch_size, -1))
        out = self.dropout(out)
        out = self.relu(out)
        out = self.l2(out)
        
        
        output = self.l3(torch.cat((out,r), dim = 1))
        
        return output, r
