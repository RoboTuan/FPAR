import torch 
i=0

k2 = list(torch.load("model_twoStream_state_dict.pth",map_location=torch.device('cpu')).keys())
for k in (torch.load("twoStream_4_CrossMod_dict.pth",map_location= torch.device('cpu'))):
    if(k!=k2[i]):
        print(k,k2[i])
    i +=1 

print(k2)


import torch


print(torch.cuda.device_count())   # --> 0
print(torch.cuda.is_available())   # --> False
print(torch.version.cuda)          # --> 9.0.176
print(torch.backends.cudnn.enabled)