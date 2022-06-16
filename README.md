# DeepDream

DeepDream is a Deep Learning technique that gives neaural networks ability to dream "not literally", it's an artistic image-modification technique, it generates pictures similar to the visual artifacts induced by the human's brain under psychedelic circumstances.

![alt text](https://github.com/TheMagicShop/DeepDream/blob/main/Examples/ex_default_pic1.png)



# How it works:
`DeepDream.py` file contains a `DeepDream` class, it takes a convolutional network (keras model), an input image and some settings, then it can be trained to "dream-ify" the image, it also involve a `DefaultDream` function that provides you with a `DeepDream` object with some default parameters, exempting you from coming up with a strategy and liberate you from the pain of tweaking parameters.

<br />

<br />


to instantiate a DeepDream object 4 arguments should be given.\
`
dd = DeepDream(img_path, model, preprocessing_function=None, deprocessing_function=None)
`\
arguments:\
`img_path`: the path to your image\
`model`: a keras model containing convolutional layers\
`preprocessing_function`: (optional, default `None`) your model may expect data of some sort, in such case\
this function should be provided to process the input image before feeding it to the model\
`deprocessing_function`: (optional, default `None`) needed only and only if processing_function is not `None`
it's simply a function that should undo and implement the reverse preprocessing

<br />

after that you may need to call `get_conv_layers` property or `plot_model` property to envisage wich layers
to consider, then you have to make a dictionary with keys as names of these layers, and values as floats
serving as coefficient designating the contribution of the corresponding layer, afterwards you have to 
call the `set_layers_settings` method on that dictionary
`dd.set_layers_setting({'layer_1_name': coef_1, ..})`

<br />

during the training process we are going to maximize given layers' outputs (activations) "$f(x)$" with regard to the given image "$x$" using gradient ascent with the aid of Tensorflow's autodifferentiation, in the middle of
this we are going to consider `num_octaves` stages, at each stage we are going to upscale the image (octave)
and apply gradient ascent for `iterations_per_octave numbers`.\
`
dd.train(iterations_per_octave, learning_rate, num_octaves, octave_scale, max_gain=None)
`\
arguments:\
`iterations_per_octave`: number of iterations per octave\
`learning_rate`: learning rate\
`num_octaves`: number of octaves\
`octave_scale`: the scale at which to increase the image size in each octave step\
`max_gain`: (optional, Default `None`) define a maximum gain that should not be exceeded

<br />

the `plot_final_image` property plots actually the final modified (dreamt) image.

<br />

the dream method yields an mp4 video file enchaining all the versions of the input image after each training step "$x_n$".\
`
dd.dream(output_file, fps=5)
`\
argmuents:\
`output_file`: the desired path to the output file\
`fps`: (optional, Default 5) frames per second

<br />

<br />

`DefaultDream` function gives an instantiated DeepDream object with default parameters\
arguments:\
`output_file`: the path to the output video
