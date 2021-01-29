import copy
import torch 

def change_dict(rgb_model_dict, outDir):
    state_dict = torch.load(rgb_model_dict, map_location=torch.device('cpu'))
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        key2=key
        keys = key.split(".")
        if keys[0] == 'resNet':
            keys[1]="cm_rgb_" + keys[1]
            key2=''.join([str(el) + '.' for el in keys])[:-1]
            
        state_dict_v2[key2] = state_dict_v2.pop(key)
    torch.save(state_dict_v2,outDir)
    return state_dict_v2
state_dict_v2 = change_dict("D:\POLITECNICO\Machine_learning\model_rgb_state_dict.pth","rgb_4_CrossMod_dict.pth")

for key,value in state_dict_v2.items():
    print(key,value.size())