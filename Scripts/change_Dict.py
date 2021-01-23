import copy
import torch 

def change_dict(twoStream_model_dict, outDir):
    state_dict = torch.load(twoStream_model_dict, map_location=torch.device('cpu'))
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        keys=key.split(".")

        if (keys[0] == 'flowModel'):
            key_new = ''.join([str(el)+'.' for el in keys[1:]])[:-1]
            key_new = "Model.resNet.cm_fl_" + key_new

        elif (keys[0] == 'frameModel'):

            if(keys[1] == 'resNet'):
                key_new = ''.join([str(el)+'.' for el in keys[2:]])[:-1]
                key_new = "Model." + keys[1] + ".cm_rgb_" + key_new

            elif(keys[1] == 'lstm_cell'):
                key_new = ''.join([str(el) + '.' for el in keys[1:]])[:-1]
                key_new = "Model." + key_new
            else:
                state_dict_v2.pop(key)
                continue
        else:
            key_new = key
        print(key_new)
        state_dict_v2[key_new] = state_dict_v2.pop(key)
        torch.save(state_dict_v2, outDir)

state_dict_v2 = change_dict("D:\POLITECNICO\Machine_learning\model_twoStream_state_dict.pth","twoStream_4_CrossMod_dict.pth")