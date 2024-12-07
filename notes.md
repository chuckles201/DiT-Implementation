Inspiration from [this video](https://www.youtube.com/watch?v=KAYYo3lNOHY&t=1931s&ab_channel=ExplainingAI).

## DiT Notes

### Attention vs. CNNs
The attention mechanism in each layer can be seen as some type of semantic-level, in which the kqv projection weights guide the input to interact with the other features. Then this is fed to the next layer and repeated.

Convolutions are more straight forward; we look for features with our kernels, and slowly lower the spatial dimensions, while growing the feature-dimensions.

### Weight intialization and 'tricks'

Normalization is used so the the statistics in general are steady; think of having relative meaning within the neurons, but no large divergent values.

Xavier initialization is used sometimes over he initializtion because:
- accounts for stability in backward pass when using non-RELU activations functions that don't zero-out half of values
- allows

### ***ViT***

ViT introduced a model that showcased how transformers could beat CNNs:
1. Introduced 'patch-embedding' to encode information about individual HxW patches of information, which would attend to eachother as seperate embeddings/vectors
2. Used a 'class' embedding which bascially 'came along for the ride', and was evaluated at the end with the loss function

This was basically the entire architecture, which just demonstrated the (suprising) superiority of attention even in the image-space.


### ***DiT***

The DiT change the position embedding to be 2 dimensional, with elements with the same x and y coordinates sharing a D/2 position embedding, so each pixel will have a corresponding D/2 embedding for its x and y coordinate, which is concatenated and added to it. Sinusodial positional embeddings can be used.

The architecture also proposes multiple ways to represent the text-guidance, and the timestep-embedding (used for de-noising at specified state)

1. 'In-context': simply processing the time and text embedding as more tokens
2. Cross-attention: attending to the time and text tokens (2 seperate) with each patch (once for each transformer block)
3. Adaptive layer-norm: the text tokens and time embeddings are added and fed through an MLP (one per layer). Then, this gives a 4-d embedding which corresponds to scale/shift parameters of the layer norm
4. Adaptive layernorm zero: We do the same as AdaLN, however we have two extra parameters that are generated through our timestep and class: scaling our attention layer, and scaling our MLP layer. This basically gives us more control over the outputs of our transformer block through our guidance and timestep embedding. The scales are initially zero, and are gradually learned over time (the transformer is an identity mapping until learned otherwise).

### ***Scaling***

The authors test different parameter-counts, by changing the amount of heads per attn. layer with the embedding-dimension size, while also using variable amounts of layers typically found in transformers. Furthermore, they experience with different numbers of 'patches' for the latent-image.

The metric they use to compare the generative quality of the images is the FID benchmark, which measures the 'distance' between the distributions of the generated images and the real images. Basically, it uses a powerful model to extract feature-vectors from the real distribution of images, and the generated one, and measures their difference. Speecifically, we actually just get the mean and variance of each dimension of the vector, and compute the difference between these two. This is a rough approximation of how similar our generated images are to real-world ones (assuming they mostly follow a gaussian).

### ***Model architecture/Results***

The authors use *class* conditioning, in that they only wish to generate images belonging to a class (not a true T2I model), and find that the adaLn Zero works the best. Furthermore, they confirm the scaling properties of growing the paremeters with increased quality of image generation.

Here are some other things the authors find:
1. As we scale up the transformer (across all patch-sizes), it does better
2. Scaling the patch-size, while keeping the model-size constant, the model fpreforms better (flops scale, but not parameters)
3. Strong correlation between FLOPS and FID score. This makes sense, the more compute we give to the problem, the better it preforms.

Furthermore, the authors found that investing more compute in sampling with the diffusion models could not compensate for a lack of compute invested in the model during training.