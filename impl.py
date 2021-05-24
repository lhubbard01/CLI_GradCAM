import argparse

from PIL import Image
from keras.applications import imagenet_utils as imagenet

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import torch.autograd as A
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv




## Helper utilities for image processing through keras
def toTensor(image, preproc: bool = True):
  if preproc:
    prep= imgnet.preprocess_input(image)
    return torch.tensor((prep/255).transpose(2,0,1))

  return torch.tensor(image.astype(float)/255)

def toPred(tensor, top: int = 5):
  return imgnet.decode_predictions(tensor.detach().numpy(), top = top)

def loadimg(path):
  """loads image from specified location, expected as a single file.
  The images are loaded from PIL and immediately converted to numpy arrays.

  returns the image as a numpy array of uint8, the default format upon conversion from PIL"""

  p = Image.open(path)
  return np.array(p)

def loadall(opts): 
  """loads all images contained in specified directory, 
  opts is expected as a dict from cli parsing. if no images found, raises FileNotFoundError
  If opts does not contain a key for images_dir or the value does not correspond to a directory
  then a value error is raised.
  The images are all preprocessed by keras imagenet utilities, done according to the paper

  returns a list of PyTorch tensors, in pytorch format (color channels, w, h)"""

  if opts["images_dir"] and os.path.isdir(opts["images_dir"]):
    images = []
    for image in os.listdir(opts["images_dir"]): 
      images.append(toTensor(loadimg(image)))
    
    if len(images) == 0:
      raise FileNotFoundError("No files loaded from specified image directory!")

    return images

  raise ValueError("No image directory specified! No images loaded to runtime.")


## GradCAM helpers and method
def gettop(img, model, n: int = 5):
  """returns the top n predictions from the model through keras imagenet utility."""
  try: out = model(img)
  except: out = model(img.reshape(1,*img.shape))
  return out, toPred(out, top = n)




def register_for_backprop_hook(self, Input, Output):
  #cache the feature maps for use in backward hook call to display
  # NOTE this may be done more easily through explicitly writing GuidedBackprop ReLU
  self.cached_out = Output[0]


def view_grad(self, grad_in, grad_out):
    #scope only accessible through autograds return to python
    global image,  TITLE, contours, args 

    #feature map wise global average pooling
    alpha = (grad_in[0].reshape(1,512,-1).sum(dim=2) / 196).squeeze() 
    accum = torch.zeros((14,14), dtype = torch.float32)

    for i in range(512):
      accum += alpha[i] * self.cached_out[i]  



    if args["vismode"] == 0:   accum = F.relu(accum)
    elif args["vismode"] == 1: accum = accum
    elif args["vismode"] == 2: accum = accum.clip(max=0) #negative importance
    
    #view all three side by side ( should normalize these so comparison is more apt)
    else:
      fig = plt.figure(figsize = (15,15))
      fig.canvas.set_window_title(TITLE)

      ax1 = fig.add_subplot(131)
      ax1.imshow(r(image[0]).numpy().transpose(1,2,0))
      pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(accum.reshape(1,1,14,14)).numpy()).squeeze()
      imgc = ax1.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
      ax1.set_title(TITLE + " no applied clipping")
      
      ax2 = fig.add_subplot(132)
      ax2.imshow( r(image[0]).numpy().transpose(1,2,0)) retrieve original image
      pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(F.relu(accum.reshape(1,1,14,14))).numpy()).squeeze()
      imgc = ax2.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
      ax2.set_title(TITLE + " relu")

      ax3 = fig.add_subplot(133)
      imgc = ax3.imshow(r(image[0]).numpy().transpose(1,2,0))
      pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(accum.clip(max = 0).reshape(1,1,14,14)).numpy()).squeeze()
      imgc = ax3.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
      ax3.set_title(TITLE + " negatively clipping relu")

      fig.colorbar(imgc)
      plt.show()
      return 
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(TITLE)
    ax.set_title(TITLE)
    ax.imshow(r(image[0]).numpy().transpose(1,2,0))
    pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(accum.reshape(1,1,14,14)).numpy()).squeeze()
    imgc = ax.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
    fig.colorbar(imgc)
    if contours:
      for contour in contours: # contours describe the locations of foci
        ax.contour(pre_imaged[::-1], levels = [contour,1.0], colors = "k")
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--contours", action = "store_true", 
    help= "displays contours through user input ranges")
  parser.add_argument("--vismode", type = int , 
    help= "by default is ReLU, 0 is ReLU. 1 is no ReLU applied to the rescaled image. 2 is negative ReLU applied to the image", default =0)
  parser.add_argument("--top_preds", type = int, 
    help = "the number of salience renders per input image, from most confident descending.", default = 5)
  parser.add_argument("--imagedir", type = str, help = 
    "location from which to read images. All nonhidden files in the specified directory will attempt to be read from. Default is None, so one may reorganize project structure to grab images.",
    default = None)

  args = vars(parser.parse_args())
  contours = None


  if args["contours"]:
    contours = []
    s = "-1"
    while True:
      s = input("enter lower bound\t")
      if s != "" and s != "q":
        contours.append(float(s)) #upper bound is one.
      else: break

  vgg16model = tv.models.vgg16(pretrained = True, progress = True)
  images = loadall(args)

  names = {k: v for k, v in vgg16model.named_modules()}
  
  last_layer = names["features.28"] 
  last_layer.register_backward_hook(print_grad);
  last_layer.register_forward_hook(register_for_backprop_hook)

  r = tv.transforms.Resize((224,224)) #rescale for expected input dimensions
  for i, image in enumerate(images):
    output, preds = gettop(r(image), vgg16model)
    for j in range(args["top_preds"]):
      TITLE = str(preds[0][j][1:])
      output.flatten()[
          output.flatten().argsort(descending = True)
        ][j].reshape(1,-1).backward(retain_graph =True) # calc grad from x predictor

