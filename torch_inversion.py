import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################################
# Inversion
#######################################################

# you need to provide access to the datasets of images you want to invert

AIpastiche = pd.read_parquet("ai_pastiche.parquet", engine="pyarrow")
#print(len(AIpastiche))
#print(AIpastiche.columns)

NGD =  pd.read_parquet("ngd.parquet", engine="pyarrow")
#print(len(NGD))
#print(NGD.columns)

ai_image_paths = []
for _, d in AIpastiche.iterrows():
    ai_image_paths.append(f"data/generated_images/{d['generated_image']}")

nga_image_paths = []
for _, d in NGD.iterrows():
    nga_image_paths.append(f"data/national_gallery_images/{d['objectid']}.jpg")

# some functions used during inversion

# regularizer
def total_variation(x):
    return (
        (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() +
        (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    )

# edge mask is meant to constrain the perturbation to occur mainly where the image already has structure:
def edge_mask(x0, eps=1e-6):
    gray = 0.299*x0[:,0:1] + 0.587*x0[:,1:2] + 0.114*x0[:,2:3]

    kx = torch.tensor(
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        dtype=x0.dtype, device=x0.device
    ).view(1,1,3,3) / 8.0

    ky = torch.tensor(
        [[-1,-2,-1],[0,0,0],[1,2,1]],
        dtype=x0.dtype, device=x0.device
    ).view(1,1,3,3) / 8.0

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    mag = torch.sqrt(gx**2 + gy**2 + eps)

    mask = mag / (mag.amax(dim=(-2,-1), keepdim=True) + eps)

    return mask.detach()

def smoothing(mask,kernel_size=21,sigma=5.0):
    coords = torch.arange(kernel_size, device=mask.device) - kernel_size//2
    g = torch.exp(-(coords**2)/(2*sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    kernel = kernel[None,None]
    mask = F.conv2d(
        mask,
        kernel,
        padding=kernel_size//2
    )
    mask = mask / mask.max()
    return mask

##########################

# load the model you want to invert.
# The model must provide a function "score_pc1_from_image(x)"
# taking in input a torch tensor (1,336,336,3), or the input size of
# CLIP version you are considering, and returing a prediction of
# the PC1 coordinate.
# We make an example using CLIP itself.

device = 'cuda'

import clip
clipmodel='ViT-L/14@336px'
model, preprocess = clip.load(clipmodel, device="cuda")

#projection in PCA space
import joblib
pca = joblib.load("pca.pkl")

pc1_vec = torch.tensor(pca.components_[0], dtype=torch.float32, device=device)
#pc2_vec = torch.tensor(pca.components_[1], dtype=torch.float32, device=device)
pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)

def score_pc1_from_image(x):
    # x: (1,3,336,336), values in [0,1]

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device).view(1,3,1,1)

    xn = (x - mean) / std

    emb = model.encode_image(xn)
    emb = emb / emb.norm(dim=-1, keepdim=True)

    centered = emb - pca_mean

    pc1 = centered @ pc1_vec

    return pc1[0]

# invert: main inversion function
# it calls an external function score_pc1_from_image(x)

def invert(n,score_pc1_from_image,kind='ai',show=True,verbose=1):
    # n is the id of the image to invert
    if kind=='ai':
        image_path = ai_image_paths[n]
        direction = -1 # AI -> Human
    else:
        image_path = nga_image_paths[n]
        direction = 1 #Human -> AI

    img_orig = Image.open(image_path).convert("RGB")
    h,w = img_orig.size
    shape = h/w

    img = img_orig.resize((int(336*shape), 336))
    #testing
    #img = np.random.normal(size=(336,336,3),loc=200,scale=40).astype(np.uint8)
    if show: #show image
        plt.imshow(img)
        plt.tight_layout()
        plt.axis("off")
        plt.show()

    img = img_orig.resize((336, 336))
    x0 = torch.tensor(np.asarray(img) / 255.0, dtype=torch.float32, device=device)
    x0 = x0.permute(2,0,1).unsqueeze(0)

    x_init = x0.clone().detach()[0].permute(1,2,0).cpu().numpy()
    x = x0.clone().detach().requires_grad_(True)

    #loop
    mask = edge_mask(x0).detach()
    mask = smoothing(mask,sigma=5.0,kernel_size=31)
        
    lambda_img = 0.5  
    lambda_tv = 0.001 
    
    eps = 0.05 # max change per channel, in [0,1] units
    u = torch.zeros_like(x0, requires_grad=True)
    
    opt = torch.optim.Adam([u], lr=1e-2) #1e-3 
    traj = []
    
    for t in range(100): #150
        opt.zero_grad()

        delta = eps * torch.tanh(u)
        
        if False: #t%5 == 0:
            delta = torch.nn.functional.avg_pool2d(delta,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1
                                                   )
        #x = (x0 + delta).clamp(0, 1)
        x = (x0 + mask * delta).clamp(0,1)
        
        s = score_pc1_from_image(x)
        
        if t%5==0 and verbose==1:
            print(f"iteration: {t}; s: {s}")
        #print("any nan in s:", torch.isnan(s).any().cpu().numpy())
        #print("scores is ", s)

        loss = direction * s
        loss += 0.01 * total_variation(delta)
        loss.backward()
        #print("grad nan:", torch.isnan(u.grad).any())
        #print("grad min/max:", u.grad.min(), u.grad.max())
        opt.step()
        #print("u nan:", torch.isnan(u).any())
        #print("u min/max:", u.min(), u.max())
    
        with torch.no_grad():
            x.clamp_(0, 1)
            traj.append(float(s.detach().cpu()))
            
    #print(traj[0], traj[-1])
    x_final = x.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    
    if show:
        x_final_img = Image.fromarray((255*x_final).astype(np.uint8))
        plt.imshow(x_final_img.resize((int(336*shape), 336)))
        plt.tight_layout()
        plt.axis("off")
        plt.show()

        x_diff = x_final - x_init
        #print(f"min_diff = {x_diff.min()} max_diff = {x_diff.max()}")
        x_diff_norm = (x_diff - x_diff.min())/(x_diff.max()-x_diff.min())
        #print(x_diff_norm.shape)
        x_diff_norm_img = Image.fromarray((255*x_diff_norm).astype(np.uint8))
        #x_diff_norm_img.save("diff.png")
        plt.imshow(x_diff_norm_img.resize((int(336*shape), 336)))
        plt.tight_layout()
        plt.axis("off")
        plt.show()

    return(x_init,x_final) #numpy array

######################################

#reimbedding the resulting image to compute distance from the original

def embed_image(img):
    imgT = torch.tensor(
        img,
        dtype=torch.float32,
        device=device
    )

    imgT = imgT.permute(2,0,1).unsqueeze(0)   # HWC -> BCHW

    # CLIP normalization
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=device).view(1,3,1,1)

    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=device).view(1,3,1,1)

    imgT = (imgT - mean) / std

    with torch.no_grad():
        img_emb = model.encode_image(imgT)

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    img_emb = img_emb.cpu().numpy()
    return(img_emb)

#print(AIpastiche.columns)

def distance(x_init,x_final):
    #print(x_init.shape)
    pc_old = pca.transform(embed_image(x_init))
    #print(pc_old[0,0], pc_old[0,1])
    #print(x_final.shape)
    pc_new = pca.transform(embed_image(x_final))
    #print(pc_new[0,0], pc_new[0,1])
    dist = np.sqrt((pc_new[0,0]-pc_old[0,0])**2 + (pc_new[0,1]-pc_old[0,1])**2)
    print(f"distance: {dist}")
    return(pc_old[0,0],pc_old[0,1],pc_new[0,0],pc_new[0,1],dist)

#this is a comparative file
df_old = pd.read_csv('aipastiche_results_global_scattering_J4.csv') #best scattering
print(len(df_old))

assert False

rows = []
partial_new = 0
partial_old = 0

#we iterate on a given number of images

for n in range(10):  #len(AIpastiche)):
    img_init,img_final = invert(n,score_pc1_from_image,show=False,verbose=0)
    x_init,y_init,x_final,y_final,dist = distance(img_init,img_final)
    partial_new = partial_new + (x_final-x_init)
    rows.append({
        "position":n,
        "x_init": x_init,
        "y_init": y_init,
        "x_final": x_final,
        "y_final": y_final,
        "distance": dist
    })
    if df_old is not None:
        diff_old = df_old.iloc[n]["x_final"] - df_old.iloc[n]["x_init"]
        partial_old = partial_old + diff_old
        print(f"diff at {n} = {partial_new/(n+1)} vs {partial_old/(n+1)}")
    else:
        print(f"diff at {n} = {partial_new/(n+1)}")
    })

df = pd.DataFrame(rows)
#df.to_csv('aipastiche_results_clip_10.csv', index=False)  #save

df_res['xdiff'] = df_res['x_final']-df_res['x_init']
sorted_df = df_res.sort_values("xdiff",ascending=False)
print(sorted_df.head(10))

print(f"mean distance: {np.mean(sorted_df['distance'])}, std:{np.std(sorted_df['distance'])}")
print(f"mean x diff: {np.mean(sorted_df['xdiff'])}, std:{np.std(sorted_df['xdiff'])}")
