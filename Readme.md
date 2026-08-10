# Interpreting the separability of Human and AI images in CLIP's embedding space

We identify a previously unreported phenomenon in CLIP representations:
human and AI-generated paintings spontaneously separate along the dominant
principal directions of their joint embedding distribution, without any
supervised objective designed to distinguish the two classes. Rather than
exploiting this phenomenon for detection, our objective is to interpret it:
we seek to identify the visual information underlying the separation and to
trace it back from the embedding space to the image domain.

We pursue this objective through a progressive investigation combining
interpretable image representations with gradient-based inversion, used
systematically as an experimental probe of the relationships identified in
feature space. Robustness experiments and increasingly expressive
statistical descriptors progressively rule out several intuitive
explanations based on global image properties and simple local statistics,
and point instead to distributed multiscale image structure. Multiscale
scattering provides the most informative interpretable representation
considered, but offers only a partial account of the phenomenon.

Direct inversion provides a complementary and striking observation:
substantial displacements along the dominant CLIP directions can be induced
by image perturbations that remain nearly imperceptible to human observers,
showing that the directions involved in the separation are highly sensitive
to image variations with very low perceptual salience for humans. Taken
together, these results reveal a significant difference between the visual
evidence reflected in CLIP representations and that readily accessible to
human perception, raising broader questions about the relationship between
artificial and human vision and, ultimately, between artificial and human
aesthetic judgment.

<hr/>

## Notebooks



