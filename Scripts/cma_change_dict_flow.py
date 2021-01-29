import copy
import torch 

def change_dict(flow_model_dict, outDir):
    state_dict = torch.load(flow_model_dict, map_location=torch.device('cpu'))
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        state_dict_v2["cm_fl_" + key] = state_dict_v2.pop(key)
    torch.save(state_dict_v2,outDir)
    return state_dict_v2
state_dict_v2 = change_dict("D:\POLITECNICO\Machine_learning\FPAR\cma\model_flow_state_dict.pth","flow_4_CrossMod_dict.pth")

for key in state_dict_v2:
    print(key)