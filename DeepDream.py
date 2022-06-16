''' 
This module contain DeepDream class, and DefaultDream class.

to instantiate a DeepDream object 4 arguments should be given,
arguments:
img_path: the path to your image
model: a keras model containing convolutional layers.
preprocessing_function: (optional, default None) your model may expect data of some sort, in such case
this function should be provided to process the input image before feeding it to the model.
deprocessing_function: (optional, default None) needed only and only if processing_function is not None
it's simply a function that should undo and implement the reverse preprocessing.

after that you may need to call get_conv_layers property or plot_model property to envisage wich layers
to consider, then you have to make a dictionary with keys as names of these layers, and values as floats
serving as coefficient designating the contribution of the corresponding layer, afterwards you have to 
call the set_layers_settings method on that dictionary.

during the training process we are going to maximize given layers' outputs (activations) "f(x)" with regard to
the given image "x" using gradient ascent with the aid of Tensorflow's autodifferentiation, in the middle of
this we are going to consider num_octaves stages, at each stage we are going to upscale the image (octave)
and apply gradient ascent for iterations_per_octave numbers.
arguments:
iterations_per_octave: number of iterations per octave.
learning_rate: learning rate.
num_octaves: number of octaves.
octave_scale: the scale at which to increase the image size in each octave step.
max_gain: (optional, Default None) define a maximum gain that should not be exceeded.

the plot_final_image property plots actually the final edited (dreamt) image.

the dream method yields an mp4 video file enchaining all the versions of the input image after 
each training step (x[n]).
argmuents:
output_file: the desired path to the output file.
fps: (optional, Default 5) frame per second. 



DefaultDream function gives an instantiated DeepDream object with default parameters,
arguments:
output_file: the path to the output video
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

class DeepDream():
  def __init__(self, img_path, model, 
               preprocessing_function=None, deprocessing_function=None):
    self.img_path = img_path                 # path to the input image
    self.model = model                       # keras model
    self.preprocessing_function = preprocessing_function        # to process the input image, if necessary
    self.deprocessing_function = deprocessing_function          # undo the preprocessing
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    if preprocessing_function is not None:
      img = preprocessing_function(img)
    self.img = img                                   # image as array
    self.successive_results = []                     # here we save the image array after each training step
  
  @property
  def get_conv_layers(self,):
    # this method yields all the conv layers to give an idea to pick potential desired layers
    corresponding_layers_classes = tuple([keras.layers.Conv2D,
                                          keras.layers.SeparableConv2D,
                                          keras.layers.Convolution2DTranspose,
    ])
    conv_layers = list()
    for layer in self.model.layers:
      if isinstance(layer, corresponding_layers_classes):
        conv_layers.append(layer.name)
    
    if len(conv_layers) == 0:
      raise Exception("The given model does not contain any convolutional layer")
    
    else:
      print("The following are the model's convolutional layers ordered as they originally are.")
      return conv_layers
  
  @property
  def plot_model(self,):
    # plotting the model architecture
    return keras.utils.plot_model(self.model)
  
  def set_layers_settings(self, layers_settings):
    # setting a dictionary of layers_names and the desired coefficients(contirbutions)
    self.layers_settings = layers_settings
    outputs_dict = dict()
    for layer_name in layers_settings.keys():
      outputs_dict[layer_name] = self.model.get_layer(layer_name).output
    self.feature_extractor = keras.Model(inputs = self.model.inputs,
                                         outputs = outputs_dict)
    #feature extractor is a multioutput model, it has as many outputs as the specified layers 

  def compute_gain(self, img):
    # compute the gain based on weighted sum of the feature_extractor's outputs
    features = self.feature_extractor(img)
    gain = tf.zeros(shape=()) # 0
    for name in features.keys():
      coef = self.layers_settings[name]
      activation = features[name]
      gain += coef * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :])) # scalar [2:-2] to avoid atifacts
      # add the layer gain
    return gain
  
  @tf.function
  def gradient_ascent_step(self, img, learning_rate):
    with tf.GradientTape() as tape:
      # autodifferentiation
      tape.watch(img)                          # img is not a variable tape has to watch it
      gain = self.compute_gain(img)            # f(x)
    grads = tape.gradient(gain, img)           # compute the gradients
    grads = tf.math.l2_normalize(grads)        # normalize the gradient before assigning it
    img += learning_rate * grads               # ascending step
    return gain, img                           # f(x), x

  def train(self, iterations_per_octave, learning_rate, 
            num_octaves, octave_scale, max_gain=None):
    original_img = self.img                       # here we keep the img intact 
    img_shape = original_img.shape[1:3]           # our img shape
    successive_shapes = [img_shape]               # each octave's shape
    for i in range(1, num_octaves):
      scale = octave_scale ** i                   # sacle
      shape = (int(img_shape[0]/scale), int(img_shape[1]/scale))    # new shape after scaling
      successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]                     # reverse, lowest first
    self.successive_shapes = successive_shapes

    dreamt_img = tf.identity(original_img)                # copy of the original img, this goes through training 
    S = tf.image.resize(original_img, successive_shapes[0])  # original downscaled to the previous shape

    for octave, shape in enumerate(successive_shapes):
      print(f"Processing octave {octave + 1}... with shape {shape}")
      dreamt_img = tf.image.resize(dreamt_img, shape)     # new octave, new shape, upscale

      S_upscaled = tf.image.resize(S, shape)              # original downscaled to the previous shape, now is upscaed to the current shape
      original_downscaled = tf.image.resize(original_img, shape) # original downscaled to the current step
      lost_details = original_downscaled - S_upscaled            # details lost after upscaling the dreamt_img
      lost_details_final_shape = original_img - tf.image.resize(S, successive_shapes[-1]) # same idea to store each x[n] with a common size

      for i in range(iterations_per_octave):
        # for this specific octave run the gradient step iterations_per_octave times
        gain, dreamt_img = self.gradient_ascent_step(dreamt_img, learning_rate) # runnung the gradient ascent
        if max_gain is not None and gain > max_gain:
          # stop if gain exceeds max_gain
          break
        print(f"\titeration n°{i + 1}: gain is {gain:.2f}")

        result = dreamt_img + lost_details  # may or may not be relevant here
        result = tf.image.resize(dreamt_img, successive_shapes[-1])   # stock with a common size 
        result += lost_details_final_shape
        result = self.deprocessing_function(result.numpy())
        self.successive_results.append(result)

      dreamt_img += lost_details   # inject lost details
      S = tf.image.resize(original_img, shape)    # new S
    
    self.dreamt_img = self.deprocessing_function(dreamt_img.numpy())      # last output
    
  @property
  def plot_final_image(self,):
    # plot the last output image
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.imshow(self.dreamt_img)
    plt.show()
  
  def dream(self, output_file, fps=5):
    # output a video
    size = self.successive_shapes[-1]
    size = (size[1], size[0])
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for img in self.successive_results:
        out.write(img[:, :, [2,1,0]])
    out.release()












# here are some default parameters

# Inception has been proven the best
model = keras.applications.inception_v3.InceptionV3(include_top=False)

# inputs should be preprocessed to be digestible
preprocessing_function = keras.applications.inception_v3.preprocess_input

# undo the preprocessing 
def deprocessing_function(img):
  img = img.reshape((img.shape[1], img.shape[2], 3))     # squeez the data from (1, width, height, 3) to (width, height, 3)
  img /= 2.0
  img += 0.5
  img *= 255.
  img = np.clip(img, 0, 255).astype("uint8")
  return img


layers_settings = {
"mixed3": 3.0,
"mixed4": 2.0,
"mixed5": 1.0,
}

learning_rate = 10.
num_octaves = 5
octave_scale = 1.4
iterations_per_octave = 50
max_gain = 50.


def DefaultDream(img_path, output_file):
  deepdream = DeepDream(img_path=img_path, model=model, 
                        preprocessing_function=preprocessing_function,
                        deprocessing_function=deprocessing_function)
  deepdream.set_layers_settings(layers_settings)
  deepdream.train(iterations_per_octave=iterations_per_octave, learning_rate=learning_rate,
                  num_octaves=num_octaves, octave_scale=octave_scale, max_gain=max_gain)
  deepdream.dream(output_file=output_file)
  # return DeepDream object
  return deepdream