import numpy as np
from keras import activations
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from matplotlib import pyplot as plt
import scipy
from vis.input_modifiers import Jitter


model = load_model('../best_epoch_res-24.h5')
print('Model Loaded')

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
last_layer = utils.find_layer_idx(model, 'dense_1')

# Swap softmax with linear
model.layers[last_layer].activation = activations.linear
model = utils.apply_modifications(model)


layer_name = 'res2a_branch2a'
layer_idx = utils.find_layer_idx(model, layer_name)

filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:15]

vis_images = []
for idx in filters:
    print(idx)
    img = visualize_activation(model, layer_idx, filter_indices=idx)
    img = utils.draw_text(img, 'Filter {}'.format(idx))
    vis_images.append(img)

stitched = utils.stitch_images(vis_images, cols=5)
scipy.misc.toimage(stitched, cmin=0.0, cmax=255.0).save('results/{}.jpg'.format(layer_name))
print('Done')
