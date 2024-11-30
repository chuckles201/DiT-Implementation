# Data Loading


Here is the strategy for loading our data:
1. Save all of the tensors into an accesible file
2. Write a function that can access the images, and reshape/resize them, and return the image tensor along with the class-tensor (guidance)
3. Write a function that can save the latents of each image tensor passed through a VAE, and then be able to map them back with a dictionary.