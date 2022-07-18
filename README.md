# Synthesising Images from text Descriptions

#### ABSTRACT
In the last years, image synthesis from text descriptions has become one of the most important research fields of computer vision. The aim of this field is to understand spatial relations between described objects and their position in the image, composing realistic images from these relationships. In this project, state-of-the-art text-to-image synthesis methods are studied and evaluated using the FashionGen dataset in order to study their behaviour, creating a fashion generator application as a result. We propose a text-to-image synthesis method to automatically generate clothing fashion images from natural language descriptions. The image generation is based on a Generative Adversarial Network trained using the FashionGen dataset, while the input is based on natural language descriptions processed by a text encoder with an attention mechanism and the data fusion is performed by the use of Conditional Batch Normalization in order to condition the image generation by text features. We purpose a new methodology as a benchmark for text-to-image synthesis by testing our method into three different GAN architectures and making it accessible and extendable to the rest of GANs by just a small change in their architectures.
#### FASHIONGEN
This dataset is created aiming to assist the task of fashion designers to share ideas with others by translating verbal descriptions to images. Thus, given a description of a particular item, images of clothes and accessories are generated matching the description. This dataset is composed by:

![alt text](https://github.com/SarrocaGSergi/TFM/blob/main/VisualResults/Samples_Dataset.png)
- It consist of 293.008 images:
  - 260.480 images for training
  - 32.528 for validation
  - 32.528 for test
- Full HD images photographed under consistent studio conditions.
- All fashion items are photographed from 1 to 6 different angles depending on the category of the item.
- Each product belongs to a main category and a more fine-grained category.
- Each fashion item is paired with paragraph-length descriptive captions sourced from experts.
- Metadata is provided for each item. Also, the color distribution extracted from the text description presented.

![alt text](https://github.com/SarrocaGSergi/TFM/blob/main/VisualResults/Dataset.png)

#### CBN AND FINAL NETWORK STRUCTURE
![alt text](https://github.com/SarrocaGSergi/TFM/blob/main/VisualResults/CBN-Slide.png)
The main idea of this technique is to, at each normalization layer introduced in a convolutional network, normalize vision features and text features in order to achieve a common feature vector that describes the features extracted from both modalities. To do so, the key point of this technique is to calculate the mean and the standard deviation from a statistical distribution (in our case, text sequences) and store them into a learnable variable in order to update this information at every step of the network. With this, we achieve to “remember”, at each layer, the network which is the distribution that should follow during the creation of each image, arriving with that to condition the generation of the images.

Therefore, the experiments realized with this technique started by changing the batch normalization layers of original architectures by the conditional batch normalization layers. After this, some dimensionality adjustments should be performed in order to match these new layers to the conditions of the network.

As CBN uses two different MLPs to calculate the mean and the standard deviation, we need to adjust the input, the hidden layers, and the output of each of the MLPs. Either for mean or std. deviation we adjusted the dimension in the same way. First, the input is set to the number of classes that we are introducing to the network. Then the hidden layers are set to 64 (this number has no mathematical relation). Finally, the output of the MLPs will be the same size as the vision feature vector calculated on the last convolution layer. This will make the different features calculated to match in dimensionality and therefore we can perform batch normalization on them.

Due to the nature of the networks, we can observe that the setting of the output features from the CBN layer will increase by the power of 2 in the Discriminator and decrease by the power of 2 in the Generator. Furthermore, in these experiments we introduced a new way of extracting text features. In this case we are no longer creating an embedding of labels, we are using the last hidden state of a pretrained BERT transformer enconder which has an output of 768. That means that now the
number of classes we have is 768.

The first experiments performed with this technique were using just a word for conditioning the image generation. This means, that every word was encoded as a 768 embedding by a transformer and then normalized with the vision corresponding embedding.

Finally, we tested this same experiment but by the use of text descriptions that contain much more information of how the images should be generated. In this case the transformer had to encode each of the descriptions in a batch into a 768 embedding that contains the latent information of the description. Therefore, the input of the first normalization layer is [64, 128] from the vision modality and [64, 768] from the text modality and the output is a feature vector with shape [64, 128].


> Google Slides: https://docs.google.com/presentation/d/1nCwbefOx-rrPQlFhCvmk8oVeEe0K-IX3fToiJn9An2c/edit?usp=sharing
