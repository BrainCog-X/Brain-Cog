from braincog.base.safety.distortion import ag_distort_28, ag_distort_224, ag_distort_silhouette, save_image, get_silhouette_data
from braincog.datasets import get_mnist_data

# An example of Abutting Grating Distortion applied on MNIST
train_loader, test_loader, _, _ = get_mnist_data(batch_size=100)
for images, labels in train_loader:
    images = ag_distort_28(images, interval=4, phase=2, direction=(1,0))
    save_image(images[0], 'test_ag_mnist.png')
    break

# An example to generate Abutting Grating distorted MNIST of resolution 224x224
for images, labels in train_loader:
    images = ag_distort_224(images, interval=8, phase=4, direction=(1,0))
    save_image(images[0], 'test_high_res_ag_mnist.png')
    break

# An example of Abutting Grating Distortion applied on silhouettes of 16-class-ImageNet
'''
The silhouette images can be downloaded from https://github.com/rgeirhos/texture-vs-shape
'''
dataset = get_silhouette_data('./silhouettes')
for images, labels in dataset:
    images = ag_distort_silhouette(images, interval=16, phase=8)
    
    save_image(images[0], 'test_ag_silhouettes.png')
    break