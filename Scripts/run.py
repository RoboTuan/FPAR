import ML_DL_Project.Scripts.cma_main_run_twoStream as cmr

cross_mod_att = ["--trainDatasetDir", "./GTEA61",
              "--outDir", "outDir",
              "--valDatasetDir", "./GTEA61_val",
              "--stackSize", "5",
              "--seqLen", "7",
              "--numEpochs", "3",
              "--flowModel", "./drive/MyDrive/model_flow_state_dict_300.pth",
              "--rgbModel", "./drive/MyDrive//model_rgb_state_dict.pth",  
]

cmr.__main__(cross_mod_att)