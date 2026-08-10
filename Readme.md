# Interpreting the separability of Human and AI images in CLIP's embedding space

This notebook is a companion repository of the article "On the Separation of Human and AI-Generated
Images in CLIP Embedding Space". 

We address a previously unreported phenomenon in CLIP representations:
human and AI-generated paintings spontaneously separate along the dominant
principal directions of their joint embedding distribution, without any
supervised objective designed to distinguish the two classes. Our aim is to interpret it:
we seek to identify the visual information underlying the separation and to
trace it back from the embedding space to the image domain.

<p align="center">
  <img src="2D_PCA_projection.png" width="800" title="2D separation">
</p>

We pursue this objective through a progressive investigation combining
interpretable image representations with gradient-based inversion, used
systematically as an experimental probe of the relationships identified in
feature space. Robustness experiments and increasingly expressive
statistical descriptors progressively rule out several intuitive
explanations based on global image properties and simple local statistics,
and point instead to distributed multiscale image structure. Multiscale
scattering provides the most informative interpretable representation
considered, but offers only a partial account of the phenomenon.

The following picture describes the distribution of scattering features realtive
to AI-generated images of AI-Pastiche, and Human paintings of the National Gallery
of Art of Washington.

<p align="center">
  <img src="scattering_comparison_J3.png" width="800" title="scattering distributions for AI and Human paiintings">
</p>

Direct inversion provides a complementary and striking observation:
substantial displacements along the dominant CLIP directions can be induced
by image perturbations that remain nearly imperceptible to human observers,
showing that the directions involved in the separation are highly sensitive
to image variations with very low perceptual salience for humans. 

<p align="center">
  <img src="clip_movements.png" width="800" title="movements in the 2D PCA space induced through gradient ascent inversion">
</p>

Taken
together, these results reveal a significant difference between the visual
evidence reflected in CLIP representations and that readily accessible to
human perception, raising broader questions about the relationship between
artificial and human vision and, ultimately, between artificial and human
aesthetic judgment.

<hr/>

## Code and Notebooks

* <a href="Detection_of_AI_generated_paintins.ipynb">Detection_of_AI_generated_paintings.ipynb</a>. This notebook illustrate the problem, using AI-generated images taken from <a href="https://www.kaggle.com/datasets/asperticsuniboit/deepfakedatabase">AI-Pastiche</a>, and Human paintings from the National Gallery of Art of Washington. It contains code for downloading the datasets, creating the CLIPs embeddings (for ViT-L/14@336px), computing the 2d PCA projection and visualizing it.

* <href="interactive_analysis.py">interactive_analysis.py</a>. This file support interactive inspection of the points in the PCA space (on click visualization). It exploits pre-computed CLIP's embeddings, provided below. 



