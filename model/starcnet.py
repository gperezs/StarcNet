import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a1 = 1 # filters multiplier
        self.a21 = 1
        self.a2 = 1
        self.a3 = 1
        n = 8 # for groupnorm
        self.sz1 = 32 # size of input image (sz1 x sz1)
        self.sz = 10 # for secon input size (2*sz x 2*sz) default: 10
        self.conv1 = nn.Conv2d(5, 128, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(n,128)
        self.relu1 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(n,128)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.GroupNorm(n,128)
        self.relu4 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.GroupNorm(n,128)
        self.relu7 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.GroupNorm(n,128)
        self.relu8 = nn.LeakyReLU()
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn11 = nn.GroupNorm(n,128)
        self.relu11 = nn.LeakyReLU()
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn12 = nn.GroupNorm(n,128)
        self.relu12 = nn.LeakyReLU()
        self.fc1 = nn.Linear(128*int(self.sz1/2)*int(self.sz1/2), 128)
        self.bn13 = nn.GroupNorm(n,128)
        self.relu13 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(384, 4)
        
        self.aconv1 = nn.Conv2d(5, 128, kernel_size=3, padding=1)
        self.abn1 = nn.GroupNorm(n,128)
        self.arelu1 = nn.LeakyReLU()
        self.aconv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.abn3 = nn.GroupNorm(n,128)
        self.arelu3 = nn.LeakyReLU()
        self.aconv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.abn4 = nn.GroupNorm(n,128)
        self.arelu4 = nn.LeakyReLU()
        self.aconv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.abn7 = nn.GroupNorm(n,128)
        self.arelu7 = nn.LeakyReLU()
        self.apool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.aconv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.abn8 = nn.GroupNorm(n,128)
        self.arelu8 = nn.LeakyReLU()
        self.aconv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.abn11 = nn.GroupNorm(n,128)
        self.arelu11 = nn.LeakyReLU()
        self.aconv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.abn12 = nn.GroupNorm(n,128)
        self.arelu12 = nn.LeakyReLU()
        self.afc1 = nn.Linear(128*int(self.sz1/2)*int(self.sz1/2), 128)
        self.abn13 = nn.GroupNorm(n,128)
        self.arelu13 = nn.LeakyReLU()
        self.adropout1 = nn.Dropout()
         
        self.aaconv1 = nn.Conv2d(5, 128, kernel_size=3, padding=1)
        self.aabn1 = nn.GroupNorm(n,128)
        self.aarelu1 = nn.LeakyReLU()
        self.aaconv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.aabn3 = nn.GroupNorm(n,128)
        self.aarelu3 = nn.LeakyReLU()
        self.aaconv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.aabn4 = nn.GroupNorm(n,128)
        self.aarelu4 = nn.LeakyReLU()
        self.aaconv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.aabn7 = nn.GroupNorm(n,128)
        self.aarelu7 = nn.LeakyReLU()
        self.aapool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.aaconv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.aabn8 = nn.GroupNorm(n,128)
        self.aarelu8 = nn.LeakyReLU()
        self.aaconv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.aabn11 = nn.GroupNorm(n,128)
        self.aarelu11 = nn.LeakyReLU()
        self.aaconv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.aabn12 = nn.GroupNorm(n,128)
        self.aarelu12 = nn.LeakyReLU()
        self.aafc1 = nn.Linear(128*int(self.sz1/2)*int(self.sz1/2), 128)
        self.aabn13 = nn.GroupNorm(n,128)
        self.aarelu13 = nn.LeakyReLU()
        self.aadropout1 = nn.Dropout()
        
        self.resize = nn.Upsample(size=(self.sz1,self.sz1))
        
    def forward(self, x):
        x0 = x[:,:,:,:]#.unsqueeze(1)
        x1 = x[:,:,int(self.sz1/2-self.sz):int(self.sz1/2+self.sz), \
                int(self.sz1/2-self.sz):int(self.sz1/2+self.sz)] #.unsqueeze(1)
        x2 = x[:,:,int(self.sz1/2-self.sz/2):int(self.sz1/2+self.sz/2+1), \
                int(self.sz1/2-self.sz/2):int(self.sz1/2+self.sz/2+1)]#.unsqueeze(1)
        x1 = self.resize(x1)
        x2 = self.resize(x2)

        out = self.relu1(self.bn1(self.conv1(x0)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.relu4(self.bn4(self.conv4(out)))
        out = self.pool2(self.relu7(self.bn7(self.conv7(out))))
        out = self.relu8(self.bn8(self.conv8(out)))
        out = self.relu11(self.bn11(self.conv11(out)))
        out = self.relu12(self.bn12(self.conv12(out)))
        out = self.dropout1(self.relu13(self.bn13(self.fc1(out.view(-1, 128*int(self.sz1/2)*int(self.sz1/2))))))
        
        out2 = self.arelu1(self.abn1(self.aconv1(x1)))
        out2 = self.arelu3(self.abn3(self.aconv3(out2)))
        out2 = self.arelu4(self.abn4(self.aconv4(out2)))
        out2 = self.apool2(self.arelu7(self.abn7(self.aconv7(out2))))
        out2 = self.arelu8(self.abn8(self.aconv8(out2)))
        out2 = self.arelu11(self.abn11(self.aconv11(out2)))
        out2 = self.arelu12(self.abn12(self.aconv12(out2)))
        out2 = self.adropout1(self.arelu13(self.abn13(self.afc1(out2.view(-1, 128*int(self.sz1/2)*int(self.sz1/2))))))
         
        out3 = self.aarelu1(self.aabn1(self.aaconv1(x2)))
        out3 = self.aarelu3(self.aabn3(self.aaconv3(out3)))
        out3 = self.aarelu4(self.aabn4(self.aaconv4(out3)))
        out3 = self.aapool2(self.aarelu7(self.aabn7(self.aaconv7(out3))))
        out3 = self.aarelu8(self.aabn8(self.aaconv8(out3)))
        out3 = self.aarelu11(self.aabn11(self.aaconv11(out3)))
        out3 = self.aarelu12(self.aabn12(self.aaconv12(out3)))
        out3 = self.aadropout1(self.aarelu13(self.aabn13(self.aafc1(out3.view(-1, 128*int(self.sz1/2)*int(self.sz1/2))))))
        
        out = torch.cat((out,out2,out3),1)
        
        out = self.fc2(out)
        return out


