## AWESOME, WORKING INSTRUCTION, as of 2021-12-17
https://coremltools.readme.io/docs/introductory-quickstart
# ROOT: https://coremltools.readme.io/docs/installation#


## NOTE: I had to
pip3 install tensorflow h5py pillow
ALSO install the *PRERELEASE VERSION* of coremltools,
## because I was a feature had not yet been
you may be able to just do:
%pip3 install --pre coremltools

but i had to uninstall coremltools and install the pre-release version of it. YMMV>
% pip3 uninstall coremltools
% pip3 install --pre coremltools

# ======================================================================

* I wanted to create a coreML filter that could be used to knock out the background while recording a "selfie video" (i.e., a "talking head" video)

* I wanted to run this in Apple's "Swift Playground"

# H3
# I found instructions for how to create a coreML here
# These instructions are for creating an imageclassifer
# that can be then used in CoreML SwiftPlayground
#
# After I got this to work, I followed similar steps
# for getting the bytedance "knockout image background" to work
#
# https://coremltools.readme.io/docs/introductory-quickstart

# I followed this build instructions
# BY FEEDING THESE STEPS, IN STAGES, TO THE PYTHON INTERPRETER

# 0. % python3

# SEE: https://coremltools.readme.io/docs/introductory-quickstart

# ======================================================================
# A: DOWNLOAD THE MODEL

import tensorflow as tf # TF 2.2.0

# Download MobileNetv2 (using tf.keras)
keras_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3,),
    classes=1000,
)


# ======================================================================
# B: Download class labels (from a separate file)

import urllib
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
class_labels = urllib.request.urlopen(label_url).read().splitlines()
class_labels = class_labels[1:] # remove the first class which is background
assert len(class_labels) == 1000

# make sure entries of class_labels are strings
for i, label in enumerate(class_labels):
  if isinstance(label, bytes):
    class_labels[i] = label.decode("utf8")

# ======================================================================
# C: CONVERT THE MODEL

import coremltools as ct

# Define the input type as image,
# set pre-processing parameters to normalize the image
# to have its values in the interval [-1,1]
# as expected by the mobilenet model
image_input = ct.ImageType(shape=(1, 224, 224, 3,),
                           bias=[-1,-1,-1], scale=1/127)

# set class labels
classifier_config = ct.ClassifierConfig(class_labels)

# Convert the model using the Unified Conversion API
model = ct.convert(
    keras_model, inputs=[image_input], classifier_config=classifier_config,
)

# ======================================================================
# D: SET THE MODEL METADATA
# Set feature descriptions (these show up as comments in XCode)
model.input_description["input_1"] = "Input image to be classified"
model.output_description["classLabel"] = "Most likely image category"

# Set model author name
model.author = '"Original Paper: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen'

# Set the license of the model
model.license = "Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for the original source of the model."

# Set a short description for the Xcode UI
model.short_description = "Detects the dominant objects present in an image from a set of 1001 categories such as trees, animals, food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%."

# Set a version for the model
model.version = "2.0"


# ======================================================================
# E: MAKE PREDICTIONS
# Use PIL to load and resize the image to expected size
from PIL import Image
example_image = Image.open("daisy.png").resize((224, 224))

# Make a prediction using Core ML
out_dict = model.predict({"input_1": example_image})

# Print out top-1 prediction
print(out_dict["classLabel"])



# ======================================================================
# F: SAVE AND LOAD THE MODEL
# Save model in the Core ML format
model.save("MobileNetV2.mlmodel")

# Load the saved model
loaded_model = ct.models.MLModel("MobileNetV2.mlmodel")

# ======================================================================
# F-ALT: If you converted the model to the more advanced ML program model type,
# you must save the model as a Core ML model package by specifying .mlpackage

## Save model as a Core ML model package
#model.save("MobileNetV2.mlpackage")
## Load the saved model
#loaded_model = ct.models.MLModel("MobileNetV2.mlpackage")

# ======================================================================
# G: USE THE MODEL WITH XCODE





1. create a new playground
2. copy my model (here: "MobileNetV2" into the "Resources" folder inside the playground)
3.

   copy the compiled model (.mlmodelc) into the playground

% NAME_OF_APP='HeadRoomApp_v0_0_2'
% NAME_OF_MACHINE_LEARNING_MODEL='MobileNetV2'
cd ~/Library/Developer/Xcode/DerivedData/Build/Products/Debug-iphonesimulator/$NAME_OF_APP-BLAHBLAHBLAH/Build/Products/Debug-iphonesimulator/$NAME_OF_MACHINE_LEARNING_MODEL.mlmodelc"


# IN MY CASE:
cd ~/Library/Developer/Xcode/DerivedData/HeadRoomApp_v0_0_2-bdjrwjwfkgktsyfjipeqcbywidzx/Build/Products/Debug-iphonesimulator

// ======================================================================

HERE'S WHAT MY SESSION LOOKED LIKE:


~/Downloads/RobustVideoMatting-coreml ᐅ python3                                                                                                                                                15:12:54
Python 3.9.9 (main, Nov 21 2021, 03:23:44)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> import tensorflow as tf # TF 2.2.0

# Download MobileNetv2 (using tf.keras)
keras_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3,),
    classes=1000,
)

>>>
>>> # Download MobileNetv2 (using tf.keras)
>>> keras_model = tf.keras.applications.MobileNetV2(
...     weights="imagenet",
...     input_shape=(224, 224, 3,),
...     classes=1000,
... )
2021-12-17 15:13:15.476809: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
14540800/14536120 [==============================] - 2s 0us/step
14548992/14536120 [==============================] - 2s 0us/step
>>>
>>>
>>>
>>>
>>> import urllib
>>> label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
>>> class_labels = urllib.request.urlopen(label_url).read().splitlines()
>>> class_labels = class_labels[1:] # remove the first class which is background
>>> assert len(class_labels) == 1000
>>>
>>>
>>>
>>>
>>> # make sure entries of class_labels are strings
>>> for i, label in enumerate(class_labels):
...   if isinstance(label, bytes):
...     class_labels[i] = label.decode("utf8")
...
>>>
>>> image_input = ct.ImageType(shape=(1, 224, 224, 3,),
...                            bias=[-1,-1,-1], scale=1/127)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'ct' is not defined
>>>
>>> # set class labels
>>> classifier_config = ct.ClassifierConfig(class_labels)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'ct' is not defined
>>>
>>> # Convert the model using the Unified Conversion API
>>> model = ct.convert(
...     keras_model, inputs=[image_input], classifier_config=classifier_config,
... )
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'ct' is not defined
>>> import coremltools as ct
WARNING:root:TensorFlow version 2.7.0 has not been tested with coremltools. You may run into unexpected errors. TensorFlow 2.5.0 is the most recent version that has been tested.
WARNING:root:Keras version 2.7.0 has not been tested with coremltools. You may run into unexpected errors. Keras 2.2.4 is the most recent version that has been tested.
>>> image_input = ct.ImageType(shape=(1, 224, 224, 3,),
...                            bias=[-1,-1,-1], scale=1/127)
>>>
>>> # set class labels
>>> classifier_config = ct.ClassifierConfig(class_labels)
>>>
>>> # Convert the model using the Unified Conversion API
>>> model = ct.convert(
...     keras_model, inputs=[image_input], classifier_config=classifier_config,
... )
2021-12-17 15:14:21.935206: I tensorflow/core/grappler/devices.cc:75] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
2021-12-17 15:14:21.965850: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1149] Optimization results for grappler item: graph_to_optimize
  function_optimizer: function_optimizer did nothing. time = 0.015ms.
  function_optimizer: function_optimizer did nothing. time = 0.002ms.

2021-12-17 15:14:22.894072: I tensorflow/core/grappler/devices.cc:75] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
2021-12-17 15:14:23.174646: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1149] Optimization results for grappler item: graph_to_optimize
  constant_folding: Graph size after: 427 nodes (-262), 698 edges (-262), time = 142.104ms.
  dependency_optimizer: Graph size after: 426 nodes (-1), 435 edges (-263), time = 11.541ms.
  debug_stripper: debug_stripper did nothing. time = 1.14ms.
  constant_folding: Graph size after: 426 nodes (0), 435 edges (0), time = 27.491ms.
  dependency_optimizer: Graph size after: 426 nodes (0), 435 edges (0), time = 9.296ms.
  debug_stripper: debug_stripper did nothing. time = 0.66ms.

2021-12-17 15:14:23.744020: I tensorflow/core/grappler/devices.cc:75] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
2021-12-17 15:14:23.753038: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1149] Optimization results for grappler item: graph_to_optimize
  function_optimizer: function_optimizer did nothing. time = 0.005ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.

2021-12-17 15:14:24.534110: I tensorflow/core/grappler/devices.cc:75] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
2021-12-17 15:14:24.733901: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1149] Optimization results for grappler item: graph_to_optimize
  constant_folding: Graph size after: 427 nodes (-262), 698 edges (-262), time = 94.845ms.
  dependency_optimizer: Graph size after: 426 nodes (-1), 435 edges (-263), time = 9.687ms.
  debug_stripper: debug_stripper did nothing. time = 0.872ms.
  constant_folding: Graph size after: 426 nodes (0), 435 edges (0), time = 27.722ms.
  dependency_optimizer: Graph size after: 426 nodes (0), 435 edges (0), time = 7.375ms.
  debug_stripper: debug_stripper did nothing. time = 0.675ms.

Running TensorFlow Graph Passes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  9.90 passes/s]
Converting Frontend ==> MIL Ops: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 426/426 [00:00<00:00, 600.28 ops/s]
Running MIL Common passes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [00:01<00:00, 33.38 passes/s]
Running MIL Clean up passes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 120.07 passes/s]
Translating MIL ==> NeuralNetwork Ops: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 487/487 [00:00<00:00, 980.45 ops/s]
>>> model.input_description["input_1"] = "Input image to be classified"
>>> model.output_description["classLabel"] = "Most likely image category"
>>>
>>> # Set model author name
>>> model.author = '"Original Paper: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen'
>>>
>>> # Set the license of the model
model.license = "Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for the original source of the model."

# Set a short description for the Xcode UI
model.short_description = "Detects the dominant objects present in an image from a set of 1001 categories such as trees, animals, food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%."

# Set a version for the model
model.version = "2.0"
>>> model.license = "Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for the original source of the model."
>>>
>>> # Set a short description for the Xcode UI
>>> model.short_description = "Detects the dominant objects present in an image from a set of 1001 categories such as trees, animals, food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%."
>>>
>>> # Set a version for the model
>>> model.version = "2.0"
>>>
>>>
>>>
>>>
>>>
>>>
>>> # E: MAKE PREDICTIONS
>>> # Use PIL to load and resize the image to expected size
>>> from PIL import Image
>>> example_image = Image.open("daisy.png").resize((224, 224))
>>>
>>> # Make a prediction using Core ML
>>> out_dict = model.predict({"input_1": example_image})

# Print out top-1 prediction
print(out_dict["classLabel"])

>>>
>>> # Print out top-1 prediction
>>> print(out_dict["classLabel"])
daisy
>>>
>>> model.save("MobileNetV2.mlmodel")
>>>
>>> # Load the saved model
>>> loaded_model = ct.models.MLModel("MobileNetV2.mlmodel")
>>>


# ======================================================================


// NB: SEE: https://www.createwithswift.com/using-an-object-detection-machine-learning-model-in-swift-playgrounds/

### I SUCCESSFULLY RECEIVED the message "daisy", which indicated the classifier
### had successfully classified my image of a daisy

### Next, I make this model accessible and working in a swift playground.
### Remember, I will later adjust this workflow to use the bytedance code
### for knocking out a background.  But first, I want to make sure this
### simple image-classifier works.



#HERE'S HOW I GOT THE IMAGE CLASSIFIER MODEL TO WORK WITHIN A COREML SWIFT PLAYGROUND
##

Create a new playground.
Drag the "MobileNetV2.mlmodel" class that we just created into the "Resources" folder inside the playground.

==> THIS PART IS NON-OBVIOUS <==
click on the "MobileNetV2.mlmodel" in the project navigator,
then click on the tiny "C" next to where it says "Model Class // Automatically generated Swift model class"

THis will load the automatically generated siwft class.
You'll want to copy and paste this content into it's own swift file
(such as "ModelNetV2.swift")

Now you can access these functions in your playground.

# FINALLY, following instructions on bottom part of this page, re: what to type into the playground itself
https://www.createwithswift.com/using-an-object-detection-machine-learning-model-in-swift-playgrounds/



# ======================================================================

## OK, NOW THAT THAT WORKS,
## I TAILORED THE ABOVE SCRIPT TO CREATE A .mlmodel that works with the ByteDance code

##
## B/c for my purposes, I desire to knockout the background on a video feed with
## dimensions 1080x1920, ... I use those values in my configuration
## (I have the model automatically scale any image to 1/4th that dimension, becuase
## the smaller sample size to work with the better, as long as accuracy is unaffected)
