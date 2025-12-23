from models.ACSFormer import ACSFormer

def get_model(modelname="Unet", img_size=256, img_channel=1, classes=9, assist_slice_number=4):
   
    if modelname == "ACSFormer":
        model = ACSFormer(n_channels=img_channel, n_classes=classes)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model