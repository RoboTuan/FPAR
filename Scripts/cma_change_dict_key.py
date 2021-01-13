import torch
from ML_DL_Project.Scripts.resnetMod import *
from ML_DL_Project.Scripts.flow_resnet import *

def change_key_names(flow_model_path,frame_model_path,out_path_flow,out_path_frame):

    flowModel = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
    flowModel.load_state_dict(torch.load(flow_model_path))

    flowModelB = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
    flowModel.load_state_dict(torch.load(flow_model_path))
            
    state_dict = flowModel.state_dict()
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if 'conv1' in key:
            pre, post = key.split('.')
            state_dict_v2[pre+'_cm_fl'+'.'+post] = state_dict_v2.pop(key)

    flowModelB.load_state_dict(state_dict_v2,strict=False)

    torch.save(flowModelB.state_dict(), out_path_flow)

    frameModel = attentionModel(num_classes, memSize)
    frameModel.load_state_dict(torch.load(frame_model_path))
    frameModelB = attentionModel(num_classes, memSize)
    frameModel.load_state_dict(torch.load(frame_model_path))

    state_dict = frameModel.state_dict()
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if 'conv1' in key:
            pre, post = key.split('.')
            state_dict_v2[pre+'_cm_rgb'+'.'+post] = state_dict_v2.pop(key)

    frameModelB.load_state_dict(state_dict_v2,strict=False)

    torch.save(flowModelB.state_dict(), out_path_frame)

