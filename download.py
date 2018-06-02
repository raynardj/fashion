
# coding: utf-8

# # Download Dataset

# In[1]:


import pandas as pd

import os


# In[2]:


os.system("mkdir -p /data")
os.system("mkdir -p /data/fashion")
os.system("mkdir -p /data/fashion/img")
os.system("mkdir -p /data/fashion/img/train")
os.system("mkdir -p /data/fashion/img/valid")


# In[3]:


COMP  = "/data/fashion/"
TRAIN = COMP + "train.csv"
VALID = COMP + "validation.csv"


# In[27]:


# from json import load as loadjs

# train_ = loadjs(open(TRAIN))

# [train_.keys()]

# img_df = pd.DataFrame(train_["images"])
# img_df.head()

# ann_df = pd.DataFrame(train_["annotations"])
# ann_df.head()

# img_df["label"] = ann_df["labelId"]

# img_df.to_csv("/data/fashion/train.csv",index=False)

# from json import load as loadjs

# valid_ = loadjs(open(VALID))

# [valid_.keys()]

# img_df = pd.DataFrame(valid_["images"])
# img_df.head()

# ann_df = pd.DataFrame(valid_["annotations"])
# ann_df.head()

# img_df["label"] = ann_df["labelId"]

# img_df.to_csv("/data/fashion/validation.csv",index=False)


# In[38]:


train_df = pd.read_csv(TRAIN)


# In[39]:


print(train_df.sample(5))


# In[40]:


train_list = train_df.as_matrix().tolist()


# In[61]:


from PIL import Image


# In[79]:


def download(element):
    try:
        iid,url,lbl = element
        fn = "/data/fashion/img/train/%s"%(str(iid)+"p_c"+"_c".join(eval(lbl))+".jpg")
        cmd = "wget -O %s %s"%(fn,url)
        os.system(cmd)
        img = Image.open(fn)
        img.resize((320,320)).save(fn)
        return 1
    except:
        return 0


# In[80]:


from multiprocessing import Pool


# In[81]:


p=Pool(5)


# In[82]:


result = p.map(download,train_list)
print(sum(result))

