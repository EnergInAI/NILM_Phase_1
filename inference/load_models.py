import torch
from models.seq2point_cnn import Seq2PointCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    model = Seq2PointCNN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def load_all_models():
    return {
        "fridge": load_model("models/fridge_best.pth"),
        "ac": load_model("models/ac_best.pth"),
        "tv": load_model("models/tv_best.pth"),
        "wm": load_model("models/wm_best.pth"),
    }
