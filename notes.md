Inspiration from [this video](https://www.youtube.com/watch?v=KAYYo3lNOHY&t=1931s&ab_channel=ExplainingAI).

## DiT Notes


### ***ViT***

ViT introduced a model that showcased how transformers could beat CNNs:
1. Introduced 'patch-embedding' to encode information about individual wxh patches of information, which would attend to eachother as seperate embeddings/vectors
2. Used a 'class' embedding which bascially 'came along for the ride', and was evaluated at the end with the loss function

This was basically the entire architecture, which just demonstrated the (suprising) superiority of attention even in the image-space.