import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kymatio.torch import Scattering2D


#Scattering
device = "cuda"

scattering_J5 = Scattering2D(J=5, L=4, shape=(336,336)).to(device)
scattering_J4 = Scattering2D(J=4, L=4, shape=(336,336)).to(device)
scattering_J3 = Scattering2D(J=3, L=4, shape=(336,336)).to(device) 

def scattering_features_torch(x,scattering):
    # x: (1,3,336,336), RGB in [0,1]
    gray = 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]   # (1,H,W)
    S = scattering(gray)                                # (1,C,h,w)
    S_mean = S.mean(dim=(-2,-1))                        # (1,C)
    S_std  = S.std(dim=(-2,-1))                         # (1,C)
    X = torch.cat([S_mean, S_std], dim=1)
    return S_mean #X  

#----------------

print("loading features ...")

J=3

if J==3:
    AI_features = np.load("stats/AI_scattering_J3_L4.npy")
    NGA_features = np.load("stats/NGA_scattering_J3_L4.npy")
    AI_ext_features = np.load("stats/AI_ext_scattering_J3_L4.npy")
    NGA_ext_features = np.load("stats/NGA_ext_scattering_J3_L4.npy")
    AI_wikiart = np.load("stats/AI_wikiart_scattering_J3_L4.npy")
    Human_wikiart = np.load("stats/Human_wikiart_scattering_J3_L4.npy")
    AI_wikiart_10K = np.load("stats/AI_wikiart_10K_scattering_J3_L4.npy")
    Human_wikiart_10K = np.load("stats/Human_wikiart_10K_scattering_J3_L4.npy")

if J==4:
    AI_features = np.load("stats/AI_scattering_J4_L4.npy")
    NGA_features = np.load("stats/NGA_scattering_J4_L4.npy")
    AI_ext_features = np.load("stats/AI_ext_scattering_J4_L4.npy")
    NGA_ext_features = np.load("stats/NGA_ext_scattering_J4_L4.npy")
    AI_wikiart = np.load("stats/AI_wikiart_scattering_J4_L4.npy")
    Human_wikiart = np.load("stats/Human_wikiart_scattering_J4_L4.npy")
    AI_wikiart_10K = np.load("stats/AI_wikiart_10K_scattering_J4_L4.npy")
    Human_wikiart_10K = np.load("stats/Human_wikiart_10K_scattering_J4_L4.npy")

if J==5:
    AI_features = np.load("stats/AI_scattering_J5_L4.npy")
    NGA_features = np.load("stats/NGA_scattering_J5_L4.npy")
    AI_ext_features = np.load("stats/AI_ext_scattering_J5_L4.npy")
    NGA_ext_features = np.load("stats/NGA_ext_scattering_J5_L4.npy")
    AI_wikiart = np.load("stats/AI_wikiart_scattering_J5_L4.npy")
    Human_wikiart = np.load("stats/Human_wikiart_scattering_J5_L4.npy")
    AI_wikiart_10K = np.load("stats/AI_wikiart_10K_scattering_J5_L4.npy")
    Human_wikiart_10K = np.load("stats/Human_wikiart_10K_scattering_J5_L4.npy")

AI_features_train = AI_wikiart_10K
Human_features_train = Human_wikiart_10K

AI_features_val = np.concatenate([AI_ext_features,AI_wikiart])
Human_features_val = np.concatenate([NGA_ext_features,Human_wikiart])

AI_features_test = AI_features
Human_features_test = NGA_features

# loading target points

ai_points = np.load("embeddings/pca_AI.npy")
nga_points = np.load("embeddings/pca_NGA.npy")

ai_ext_points = np.load("embeddings/pca_AI_extension_oldspace.npy")
nga_ext_points = np.load("embeddings/pca_NGA_extension_oldspace.npy")

ai_wikiart_points = np.load("./embeddings/ai_wikiart_points.npy")
human_wikiart_points = np.load("./embeddings/human_wikiart_points.npy")

ai_wikiart_10K_points = np.load("./embeddings/ai_wikiart_10K_points.npy")
human_wikiart_10K_points = np.load("./embeddings/human_wikiart_10K_points.npy")

ai_points_train = ai_wikiart_10K_points
human_points_train = human_wikiart_10K_points

ai_points_val = np.concatenate([ai_ext_points,ai_wikiart_points])
human_points_val = np.concatenate([nga_ext_points,human_wikiart_points])

ai_points_test = ai_points
human_points_test = nga_points 

X_train = np.concatenate([AI_features_train, Human_features_train])
X_val = np.concatenate([AI_features_val, Human_features_val])
X_test = np.concatenate([AI_features_test, Human_features_test])

y = 0  #first component; set to 1 for second component
y_train = np.concatenate([ai_points_train[:,y], human_points_train[:,y]])
y_val = np.concatenate([ai_points_val[:,y], human_points_val[:,y]])
y_test = np.concatenate([ai_points_test[:,y], human_points_test[:,y]])

print(y_train.shape)

######################################
# build a torch mlp regressor
######################################
                       
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class MLPRegressorTorch(nn.Module):
    def __init__(self, in_dim, hidden=(128,64), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

#training
def load_mlp_regressor(
    checkpoint_file,
    in_dim = 181, #61, 113, 181  depending on J
    hidden=(128,64),
    dropout=0.3
):
    ckpt = torch.load(checkpoint_file)
    #print("keys: ", ckpt.keys())

    model = MLPRegressorTorch(
        in_dim=in_dim,
        hidden=hidden,
        dropout=dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters())
    
    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    opt.load_state_dict(
        ckpt["optimizer_state_dict"]
    )
    x_scaler = ckpt["x_scaler"]
    y_scaler = ckpt["y_scaler"]
    return model, x_scaler, y_scaler

    
def train_mlp_regressor(
    X_train, y_train,
    X_val, y_val,
    hidden=(128,64),
    dropout=0.3,
    weight_decay=1e-3,
    lr=1e-3,
    batch_size=64,
    epochs=300,
    patience=20,
    device="cuda",
    checkpoint_file=None            
):
    print("Training model ...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr = x_scaler.fit_transform(X_train)
    Xva = x_scaler.transform(X_val)

    ytr = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    yva = y_scaler.transform(y_val.reshape(-1,1)).ravel()

    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva = torch.tensor(yva, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True
    )

    model = MLPRegressorTorch(
        in_dim=Xtr.shape[1],
        hidden=hidden,
        dropout=dropout
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    loss_fn = nn.MSELoss()

    best_r2 = -np.inf
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva).cpu().numpy()
            pred = y_scaler.inverse_transform(pred.reshape(-1,1)).ravel()
            r2 = r2_score(y_val, pred)

        if r2 > best_r2:
            best_r2 = r2
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            if checkpoint_file:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "best_r2": best_r2,
                        "x_scaler": x_scaler,
                        "y_scaler": y_scaler
                    },
                    #"best_pc1_model_84_L4.pt"
                    checkpoint_file
                )

                print("saved", best_r2)
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    model.load_state_dict(best_state)

    return model, x_scaler, y_scaler, best_r2

training_mode = False

if training_mode:  #train
    mlp_model, x_scaler, y_scaler, r2_val = train_mlp_regressor(
        X_train, y_train,
        X_val, y_val,
        hidden=(128,64),
        dropout=0.3,
        weight_decay=1e-3, 
        lr=1e-3,
        epochs=200,
        patience=20,
        device=device,
        checkpoint_file="best_pc1_mlp_J3_L4_prova.pt"
    )

else:
    #mlp_model, x_scaler, y_scaler = load_mlp_regressor("pretrained_models/best_pc1_mlp_J5_L4.pt",in_dim = 181)
    #mlp_model, x_scaler, y_scaler = load_mlp_regressor("pretrained_models/best_pc1_mlp_J4_L4.pt",in_dim = 113)
    mlp_model, x_scaler, y_scaler = load_mlp_regressor("pretrained_models/best_pc1_mlp_J3_L4.pt",in_dim = 61) 

mlp_model.eval()

#cannot use x_scaler.transform inside a differentiable Torch graph
def get_mean_and_scale(scaler):
    s_mean = torch.tensor(
        scaler.mean_,
        dtype=torch.float32,
        device=device
    )
    s_scale = torch.tensor(
        scaler.scale_,
        dtype=torch.float32,
        device=device
    )
    return s_mean, s_scale

x_mean,x_scale = get_mean_and_scale(x_scaler)
y_mean,y_scale = get_mean_and_scale(y_scaler)

def score_pc1_from_image(x,model,scattering,x_mean,x_scale,y_mean,y_scale):
    model.eval()
    S = scattering_features_torch(x,scattering)
    Z = (S - x_mean) / x_scale
    y_scaled = model(Z)
    return y_scaled * y_scale + y_mean

def eval_model(model,X_test):
    Xte = x_scaler.transform(X_test)
    yte = y_scaler.fit_transform(y_test.reshape(-1,1)).ravel()
    Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte = torch.tensor(yte, dtype=torch.float32, device=device)

    y_pred = model(Xte).cpu().detach().numpy()
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()
    r2 = r2_score(y_test, y_pred)
    print(f"test score = {r2}")
