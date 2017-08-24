# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 06:56:05 2017

@author: eric yuan
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from vis.visualization import get_num_filters
from matplotlib import pyplot as plt
from vis.input_modifiers import Jitter

model = load_model('my_cifar10_ep10.h5')
model.summary()
# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
#layer_idx = utils.find_layer_idx(model, 'preds')
# Swap softmax with linear


plt.rcParams['figure.figsize'] = (50, 50)


# The name of the layer we want to visualize
# You can see this in the model definition.
layer_name = 'preds'
layer_idx = utils.find_layer_idx(model, layer_name)
# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)
# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))


#model.layers[layer_idx].activation = activations.linear
#model = utils.apply_modifications(model)

#This is the output node we want to maximize.
#Generate input image for each filter.
#for output_idx in filters[0:10]:
#    # Lets turn off verbose output this time to avoid clutter and just see the output.
#    img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.),input_modifiers=[Jitter(16)])
#    plt.figure()
#    plt.title('Networks perception of {}'.format(output_idx))
#    plt.imshow(img[...,0])
#plt.show()


#img = visualize_activation(model,layer_idx,filter_indices =7,max_iter=500,input_range=(0., 1.),input_modifiers=[Jitter(16)])
#plt.figure()
#plt.imshow(img)
#plt.show()

vis_images = []
for idx in filters:
    img = visualize_activation(model,layer_idx,filter_indices= idx,tv_weight = 0,lp_norm_weight=0.1,input_modifiers=[Jitter(0.05)])
    vis_images.append(img)

new_vis_image=[]
for i,idx in enumerate(filters):
    img = visualize_activation(model,layer_idx,filter_indices= idx,max_iter=1000,tv_weight =0.3,lp_norm_weight = 0.4,seed_input = vis_images[i],input_modifiers=[Jitter(0.05)])
    
    new_vis_image.append(img)
    
stitched = utils.stitch_images(new_vis_image)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()



#image_modifiers = [Jitter(16)]
## Generate input image for each filter.
#vis_images = []
#for idx in filters:    
#    img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=500, input_modifiers=image_modifiers)
#    
#    # Reverse lookup index to imagenet label and overlay it on the image.
#    #img = utils.draw_text(img, utils.get_imagenet_label(idx))
#    vis_images.append(img)
#
## Generate stitched images with 5 cols (so it will have 3 rows).
#plt.rcParams['figure.figsize'] = (50, 50)
#stitched = utils.stitch_images(vis_images)
#plt.axis('off')
#plt.imshow(stitched)
#plt.show()
