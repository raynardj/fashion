
# coding: utf-8

# # Fashion dataset
# # DenseNet Application of Image Classification
# ## Multi label problem
# The data is from the kaggle competition [iMaterialist Challenge (Fashion) at FGVC5](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018)

# In[1]:


from torchvision.models.densenet import densenet121 as feature_extractor
from torch import nn
import torch


# In[2]:


import os


# ## Config args

# In[3]:


SCALE = 256
TRAIN = "/data/fashion/img/train/"
VALID = "/data/fashion/img/valid/"
CUDA = torch.cuda.is_available()
DENSE_FEATURE = 1024
BS = 40
# VERSION = "0.0.1" # bias =True for last linear
VERSION = "0.0.2"
CATE_LEN = 228

# Mean and standard for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# In[4]:


conv_model = feature_extractor(pretrained=True)


# ### Split ConvLayer to 2 parts, conv0~transtition3, (not gonna train), denseblock4~norm5 (train)

# In[5]:


dense_conv1 = nn.Sequential(*[getattr(conv_model.features,nn_name) for nn_name in ["conv0","norm0","relu0","pool0","denseblock1","transition1",
                                                                                   "denseblock2","transition2","denseblock3","transition3",]])

dense_conv2 = nn.Sequential(*[getattr(conv_model.features,nn_name) for nn_name in ["denseblock4","norm5"]])


# ### Loading data

# In[6]:


from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
from PIL import Image


# In[7]:


transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine([-10,10]), 
                                transforms.Resize((SCALE,SCALE)),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN,STD),
                               ])


# Specific data generator

# In[8]:


class fashion_data(Dataset):
    def __init__(self,img_folder,cate_len,transform):
        super(fashion_data,self).__init__()
        self.img_folder = img_folder
        self.fnames = os.listdir(self.img_folder)
        self.cate_len = cate_len
        self.transform = transform
        
    def __len__(self):
        return len(self.fnames)
    
    def get_cate(self,url):
        zr = torch.zeros(228)
        zr[torch.LongTensor(list(int(i[1:])-1 for i in str(url).split(".")[0].split("_")[1:]))]=1
        return zr
    
    def __getitem__(self,idx):
        img = Image.open(self.img_folder+self.fnames[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.get_cate(self.fnames[idx])


# #### Train /Valid dataset

# In[ ]:


trn = fashion_data(TRAIN,CATE_LEN,transform = transform)
val = fashion_data(VALID,CATE_LEN,transform = transform)
# dl = DataLoader(trn,batch_size=4,shuffle=4,)


# In[10]:


# CLASS_TO_IDX = trn.class_to_idx
# IDX_TO_CLASS = dict((v,k) for k,v in CLASS_TO_IDX.items())
# print(IDX_TO_CLASS)


# In[11]:


# data_gen = iter(DataLoader(trn,shuffle=True))

# next(data_gen)


# ### Top half of the model, with fully connected layers

# In[12]:


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        bs = x.size()[0]
        return x.view(bs,-1)
    
def convlayer(f_in,f_out,ks,stride=1):
    return nn.Sequential(*[
        nn.Conv2d(f_in, f_out, ks, stride = stride, padding = ks//2,bias = True),
        nn.BatchNorm2d(f_out),
        nn.LeakyReLU(inplace = True),
    ])


# In[13]:


fl = Flatten()
FEATURE_WIDTH = dense_conv2(dense_conv1(torch.rand(2,3,SCALE,SCALE))).size()[1]
print(FEATURE_WIDTH)


# In[14]:


class top_half(nn.Module):
    def __init__(self):
        super(top_half,self).__init__()
        self.top_ = nn.Sequential(*[
                                    convlayer(FEATURE_WIDTH,FEATURE_WIDTH//2,3,2),
                                    convlayer(FEATURE_WIDTH//2,FEATURE_WIDTH//4,3,2),
                                    Flatten(),
                                    nn.Linear(FEATURE_WIDTH,DENSE_FEATURE,bias=False),
                                    nn.BatchNorm1d(DENSE_FEATURE),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Dropout(p=.5),
                                    nn.Linear(DENSE_FEATURE,CATE_LEN,bias=False),
                                   ])
    def forward(self,x):
        return self.top_(x)


# ### Construct Model, optimizer,train function

# In[15]:


top_half_ = top_half()


# In[16]:


if CUDA:
    torch.cuda.empty_cache()
    top_half_.cuda()
    dense_conv1.cuda()
    dense_conv2.cuda()
    
from torch.optim import Adam

opt = Adam(list(dense_conv2.parameters())+list(top_half_.parameters()),amsgrad=True)
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCEWithLogitsLoss() # Binary cross entropy with logit


# In[17]:


from p3self.matchbox import Trainer,argmax,accuracy,supermean,f1_score,save_model,load_model

def save_():
    save_model(dense_conv2,"/data/weights/dense_conv2.%s.pkl"%(VERSION))
    save_model(top_half_,"/data/weights/top_half.%s.pkl"%(VERSION))
    
def load_():
    load_model(dense_conv2,"/data/weights/dense_conv2.%s.pkl"%(VERSION))
    load_model(top_half_,"/data/weights/top_half.%s.pkl"%(VERSION))
    


# In[18]:


trainer = Trainer(trn, val_dataset = val, 
                  batch_size = BS, print_on = 5, )


# In[20]:


# what happened on each step of training
def action(*args,**kwargs):
    x,y = args[0]
    if CUDA:
        x,y = x.cuda(),y.cuda()
    x = x[:,:3,...]
    opt.zero_grad()
    y_ = top_half_(dense_conv2(dense_conv1(x)))
    
    loss = loss_func(y_,y)
    acc,recall,precision,f1 = f1_score(y_,y)
    
    loss.backward()
    opt.step()
    if kwargs["ite"]%10==9:
        save_()
    return {
        "loss":loss.item(),
        "acc":acc.item(),
        "recall":recall.item(),
        "precision":precision.item(),
        "f1":f1.item(),
    }
# what happened on each step of valid   
def val_action(*args,**kwargs):
    x,y = args[0]
    if CUDA:
        x,y = x.cuda(),y.cuda()
    x = x[:,:3,...]
    y_ = top_half_(dense_conv2(dense_conv1(x)))
    
    loss = loss_func(y_,y)
    acc,recall,precision,f1 = f1_score(y_,y)

    return {
        "loss":loss.item(),
        "acc":acc.item(),
        "recall":recall.item(),
        "precision":precision.item(),
        "f1":f1.item(),
    }


# In[22]:


trainer.action = action
trainer.val_action = val_action


# ### Training
# 
# Comment load_() if nothing to load

# In[21]:


load_()


# In[23]:


trainer.train(20)


# In[ ]:


torch.cuda.empty_cache()

