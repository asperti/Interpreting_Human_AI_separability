import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

image_features_AI = np.load("embeddings/AI-Pastiche_emb.npy") 
image_features_AI_gray = np.load("embeddings/AI-Pastiche_emb_gray.npy")
image_features_AI_blur = np.load("embeddings/AI-Pastiche_emb_blur.npy")

image_features_NGAD = np.load("embeddings/NGAD_emb.npy")
image_features_NGAD_gray = np.load("embeddings/NGAD_emb_gray.npy")
image_features_NGAD_blur = np.load("embeddings/NGAD_emb_blur.npy")

def cos_sim(X,Y):
    # X, Y: shape (n_samples, n_features)
    dot = np.sum(X * Y, axis=1)
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)
    sim = dot / (norm_X * norm_Y)
    return sim

print(f"sim ai vs gray: {np.mean(cos_sim(image_features_AI,image_features_AI_gray))}")
print(f"ngad ai vs gray: {np.mean(cos_sim(image_features_NGAD,image_features_NGAD_gray))}")

print(f"sim ai vs blur: {np.mean(cos_sim(image_features_AI,image_features_AI_blur))}")
print(f"sim ngad vs blur: {np.mean(cos_sim(image_features_NGAD,image_features_NGAD_blur))}")


#------------------------------------------------------------
# Data points
ai_points = np.load("embeddings/pca_AI.npy")
nga_points = np.load("embeddings/pca_NGA.npy")

ai_points_gray = np.load("embeddings/pca_AI_gray_oldspace.npy")
nga_points_gray = np.load("embeddings/pca_NGA_gray_oldspace.npy")

ai_points_blur = np.load("embeddings/pca_AI_blur_oldspace.npy")
nga_points_blur = np.load("embeddings/pca_NGA_blur_oldspace.npy")

def euclidean(X,Y):
    return np.sqrt(np.sum((X-Y)**2,axis=1))

print(f"euclidean distance ai vs gray: {np.mean(euclidean(ai_points,ai_points_gray))}")

print(f"euclidean distance ngad vs gray: {np.mean(euclidean(nga_points,nga_points_gray))}")

print(f"euclidean distance ai vs blur: {np.mean(euclidean(ai_points,ai_points_blur))}")

print(f"euclidean distance ngad vs blur: {np.mean(euclidean(nga_points,nga_points_blur))}")

#------------------------------------------------------------


#ai_saturation = np.load("Saturation_AI.npy")
#ai_lstar = np.load("AI_Lstar.npy")
#print(f"shape : {ai_lstar.shape}")
#nga_saturation = np.load("Saturation_NGA.npy")
#nga_lstar = np.load("NGA_Lstar.npy")

# load datasets

csv_file_name = "data/metadata.csv"
AIpastiche = pd.read_csv(csv_file_name)
print(len(AIpastiche))

# upload the National Gallery dataset from CSV file
NGD = pd.read_csv('data/national_gallery_dataset_expanded_with_style.csv')
NGD = NGD[NGD["media"] == "Painting"]
NGD = NGD.sample(len(AIpastiche), random_state=43)
print(len(NGD))
image_indices = NGD.index.tolist()

#create image paths
ai_image_paths = []
for _, d in AIpastiche.iterrows():
    ai_image_paths.append(f"data/generated_images/{d['generated_image']}")
    
nga_image_paths = []
for _, d in NGD.iterrows():
    nga_image_paths.append(f"data/national_gallery_images/{d['objectid']}.jpg")

#interactive plot

ai_points = ai_points_gray
nga_points = nga_points_gray

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_aspect(0.5)

sc_ai = ax.scatter(ai_points[:, 0], ai_points[:, 1], s=30, color="tomato")
sc_nga = ax.scatter(nga_points[:, 0], nga_points[:, 1], s=30, color="royalblue")

ax.legend()

# We'll track the current inset axes globally
axins = None

def on_click(event):
    global axins  # this fixes the SyntaxError
    if event.inaxes != ax:
        return

    # Remove any existing inset
    if axins is not None:
        axins.remove()
        axins = None

    #for scatter, points, image_paths, saturation, lstar  in [(sc_ai, ai_points, ai_image_paths, ai_saturation, ai_lstar), (sc_nga, nga_points, nga_image_paths, nga_saturation, nga_lstar)]:

    for scatter, points, image_paths in [(sc_ai, ai_points, ai_image_paths), (sc_nga, nga_points, nga_image_paths)]:
        
        contains, ind = scatter.contains(event)
        if contains:
            i = ind["ind"][0]
            print(f"selected image {i}") # with saturation {saturation[i]}")
            #print(f"Lstar mean = {lstar[i,0]}")
            #print(f"Lstar std = {lstar[i,1]}")
            #print(f"Local contrast = {lstar[i,2]}")
            #print(f"Edge contrast = {lstar[i,3]}")
            
            img = mpimg.imread(image_paths[i])

            # Create inset so that its bottom-left corner is anchored on the point
            axins = inset_axes(
                ax,
                width=3,              # size relative to parent axes
                height=3,
                loc="lower left",         # which corner touches the point
                bbox_to_anchor=(points[i, 0], points[i, 1]),
                bbox_transform=ax.transData,
                borderpad=0.0,
            )
            axins.imshow(img)
            axins.axis("off")
            fig.canvas.draw_idle()
            return  # stop after handling the first hit
    # If click was not on any scatter, hide any image
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

ax.set_title("Click a point: image’s bottom-left corner sits on the point")
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(-0.55, 0.55)
#ax.set_aspect("equal")
plt.show()
