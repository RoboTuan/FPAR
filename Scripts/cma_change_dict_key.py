import torch
from ML_DL_Project.Scripts.resnetMod import *
from ML_DL_Project.Scripts.flow_resnet import *
import copy 
def change_key_names(flow_model_path,frame_model_path,out_path_flow,out_path_frame):
    
    flowModel = flow_resnet34(False, channels=10, num_classes=61)
    flowModel.load_state_dict(torch.load(flow_model_path),strict = False)
    flowModelB = flow_resnet34(False, channels=10, num_classes=61)
    flowModel.load_state_dict(torch.load(flow_model_path),strict = False)

    state_dict = flowModel.state_dict()
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
      state_dict_v2['cm_fl_' + key] = state_dict_v2.pop(key)

    flowModelB.load_state_dict(state_dict_v2,strict=False)

    torch.save(flowModelB.state_dict(), out_path_flow)

    
    

    frameModel = resnet34(True, True)

    frameModelB = resnet34(True,True)
    

    state_dict = frameModel.state_dict()
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
      state_dict_v2['cm_rgb_'+ key] = state_dict_v2.pop(key)

    frameModelB.load_state_dict(state_dict_v2,strict=False)

    torch.save(frameModelB.state_dict(), out_path_frame)

