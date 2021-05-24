import argparse
import os

from PIL import Image
from keras.applications import imagenet_utils as imgnet

import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd as A
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from IPython import embed


import gi 
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk



class ImshowWindow:
  def __init__(self, fig):
    self.fig = fig
    label_int_cursor = Gtk.Label()
    manager = self.fig.canvas.manager
    toolbar = manager.toolbar
    vbox = manager.vbox
    label_int_cursor.set_markup("values goes here")
    label_int_cursor.show()

    vbox.pack_start(label_int_cursor, False, False, 0)
    vbox.reorder_child(toolbar,-1)


    lbl_row = Gtk.Label()
    lbl_row.set_markup("value row minus ideally")
    lbl_row.show()
    vbox.pack_start(lbl_row, False, False, 0)
    vbox.reorder_child(toolbar,-1)
    b2grid = Gtk.Button(label = "return to gridview")
    b2grid.show()

    toolitem = Gtk.ToolItem()
    toolitem.set_tooltip_text("when a clicked image is loaded, clicking this will return to grid view")
    toolitem.show()
    toolitem.add(b2grid)
    toolbar.insert(toolitem, 8)
    self.xvalue = self.yvalue = None
    self.fig.axtemp = None
    




    def mainview(self, other):
      try:
        other.fig.axtemp.remove()
      except Exception as e:
        print(e)
      other.fig.canvas.draw()


    b2grid.connect("clicked", mainview, self)


    def update(event):
      if event.xdata is None:
        label_int_cursor.set_markup("values go here right here")
      else:
        label_int_cursor.set_markup(f'<span color="#ef0000">x,y = ({int(event.xdata)}, {int(event.ydata)})</span>')
    self.fig.canvas.mpl_connect("motion_notify_event", update)
    
    def load_img(event):
      if event.xdata is None or event.ydata is None:
        lbl_row.set_markup("lbl_row")
      else:
        self.xvalue = (int(event.xdata) - int(event.xdata//28)*2) // 28
        self.yvalue = (int(event.ydata) - int(event.ydata//28)*2) // 28
        lbl_row.set_markup(f"xvalue is {self.xvalue}, yvalue is {self.yvalue}")
    self.fig.canvas.mpl_connect("motion_notify_event", load_img)
   


    def pick(event):
      if isinstance(event.artist, matplotlib.image.AxesImage):


        try:
          if hasattr(event.artist.axes, "flag"):
            y, x = int(event.mouseevent.ydata), int(event.mouseevent.xdata)
            clicked = self.fig.axtemp.images[-1].get_array()[y,x]
            self.fig.axtemp.contour(self.fig.axtemp.copied, levels = [clicked])
        except AttributeError :
          pass 
        self.fig.axtemp  = self.fig.add_axes([0,0,1,1])
        image_list =  event.artist.axes.images
        self.fig.axtemp.imshow( np.array(image_list[0].get_array()))
        copied = self.fig.axtemp.copied = np.array(image_list[1].get_array()) 
        ax_for_cmap = self.fig.axtemp.imshow( copied.reshape(224,224),alpha = 0.5, cmap="jet", picker = True)
        self.fig.axtemp.flag = True
        #self.fig.axtemp.colormap(self.fig.axtemp, copied)


        self.fig.canvas.draw()
      print(type(event))
    self.fig.canvas.mpl_connect("pick_event", pick)

    """def zoom_pic(event):
      click = 0
      self.fig.axtemp  = self.fig.add_axes([0,0,1,1])
      for image in self.fig.axes[click].images: 
        self.fig.axtemp.imshow(image.get_array().copy())
      self.fig.canvas.draw()
    self.fig.canvas.mpl_connect("button_press_event", zoom_pic)"""


  def show(self):
    plt.show()



## Helper utilities for image processing through keras
def toTensor(image, preproc: bool = True):
  if preproc:
    prep= imgnet.preprocess_input(image)
    return torch.tensor((prep/255).transpose(2,0,1))

  return torch.tensor(image.astype(float)/255)

def toPred(tensor, top: int = 5):
  return imgnet.decode_predictions(tensor.detach().numpy(), top = top)












def getSaliency(img, grad, fm_card, fm_dims):
    alpha = (grad.reshape(1,fm_card,-1).sum(dim=2) / fm_dims**2).squeeze() 
    print(alpha.shape)
    accum = torch.zeros((fm_dims,fm_dims), dtype = torch.float32)
    try:
      for i in range(fm_card):
        accum += alpha[i] * img[i]
    except:
      embed()
      raise
    return accum
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


  if opts["imagedir"] and os.path.isdir(opts["imagedir"]):
    imgpath = os.path.abspath(opts["imagedir"])
    images = []
    for image in os.listdir(opts["imagedir"]): 
      images.append(toTensor(loadimg(os.path.join(imgpath, image))))
    
    if len(images) == 0:
      raise FileNotFoundError("No files loaded from specified image directory!")

    return images

  raise ValueError("No image directory specified! No images loaded to runtime.")






class GuidedBackpropReLU(A.Function):
  global CLIP
  """currently using the clip method, maybe clamp is better? anything forwards pass 0 is zero back, anything negative from grad is 0"""

  @staticmethod
  def forward(ctx, X):
    clipped = X.clip(min = 0)
    ctx.save_for_backward(X, clipped)
    print(X.shape, clipped.shape)
    return clipped
  
  @staticmethod
  def backward(ctx, gradient):
    X, clipped = ctx.saved_tensors
    guided_relu = gradient[1].clone()
    guided_relu[clipped == 0.0] = 0.0
    guided_relu[guided_relu < 0] = 0.0
    return guided_relu
def save_vis(self, grads_in, grads_out):
  if str(self.__class__.__name__) not in "Conv2d": return
  try: self.grad_cache = [grads_in[0].clone().detach(), grads_out[0].clone().detach()]
  except AttributeError as ae: 
    print(self)
    self.grad_cache = [grads_out[0].clone().detach()]

class WrapperForModel(nn.Module):

  def __init__(self, model, method):
    super(WrapperForModel, self).__init__()
    self.model = model
    self.method = method
    self.remap(self, method)
     
  def forward(self,X):
    return self.model(X)



  def remap(self, modules,method):
   try:
    for idx, module in modules._modules.items():
      self.remap(module, method)
      if module.__class__.__name__ == "ReLU":
        modules._module[idx] = method.apply
   
   except RuntimeError as e:
    print(e)
   
   except  torch.nn.modules.module.ModuleAttributeError as e:
    print(e)
    [self.remap(child, method) for child in modules.children()]

   except AttributeError as e:
    print(e)
    for module in modules.children():
      self.remap(module, method)
      if module.__class__.__name__ == "ReLU":
        modules._module[idx] = method.apply

class GuidedBackpropReLUObj(nn.Module):
  def __init__(self):
    super(GuidedBackpropReLUObj, self).__init__()
  def forward(self, X):
    return GuidedBackpropReLU.apply(X)

## GradCAM helpers and method
def gettop(img, model, n: int = 5):
  """returns the top n predictions from the model through keras imagenet utility."""
  try: out = model(img)
  except: out = model(img.reshape(1,*img.shape))
  return out, toPred(out, top = n)

def register_for_backprop_hook(self, Input, Output):
  #cache the feature maps for use in backward hook call to displa
  # NOTE this may be done more easily through explicitly writing GuidedBackprop ReLU
  self.cached_out = Output[0]
def view_grad_per_layer(self, grad_in, grad_out):
    global fig, NCOLS, NITERS, image
    salience = F.relu(getSaliency(self.cached_out, grad_out[0], grad_out[0].shape[1], grad_out[0].shape[2]))
    if NITERS > NCOLS: return
    ax_local = fig.add_subplot(3,4,NITERS )
    ax_local.imshow(r(image).numpy().transpose(1,2,0))
    img_out = matplotlib.colors.Normalize()(nn.Upsample(size = (224,224), mode = "bilinear")(salience.reshape(1,1,*grad_out[0].shape[2:])).numpy()).squeeze()
    ax_local.imshow(img_out,
      alpha = 0.5, cmap = "jet", picker = True)
    NITERS+=1
def view_grad(self, grad_in, grad_out):
    #scope only accessible through autograds return to python
    global image,  TITLE, contours, args 

    #feature map wise global average pooling
    alpha = (grad_out[0].reshape(1,512,-1).sum(dim=2) / 196).squeeze() 
    #embed()
    accum = torch.zeros((14,14), dtype = torch.float32)
    print(self.cached_out.shape)
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
      ax1.imshow(r(image).numpy().transpose(1,2,0))
      pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(accum.reshape(1,1,14,14)).numpy()).squeeze()
      imgc = ax1.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
      ax1.set_title(TITLE + " no applied clipping")
      
      ax2 = fig.add_subplot(132)
      ax2.imshow( r(image).numpy().transpose(1,2,0)) #retrieve original image and resize to expected dims for model
      pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(F.relu(accum.reshape(1,1,14,14))).numpy()).squeeze()
      imgc = ax2.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
      ax2.set_title(TITLE + " relu")

      ax3 = fig.add_subplot(133)
      imgc = ax3.imshow(r(image).numpy().transpose(1,2,0))
      pre_imaged = matplotlib.colors.Normalize()(nn.Upsample(scale_factor = 16.0, mode = "bilinear")(accum.clip(max = 0).reshape(1,1,14,14)).numpy()).squeeze()
      imgc = ax3.imshow(pre_imaged,alpha = 0.5, cmap = "jet")
      ax3.set_title(TITLE + " negatively clipping relu")

      fig.colorbar(imgc)
      plt.show()
      return 
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(TITLE)
    ax.set_title(TITLE)
    ax.imshow(r(image).numpy().transpose(1,2,0))
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
  l = []
  names = {k: v for k, v in vgg16model.named_modules()}
  guided = WrapperForModel(vgg16model, GuidedBackpropReLU)
  #guided.apply(lambda self: self.register_backward_hook(save_vis))
  last_layer = names["features.28"] 
  #last_layer.register_backward_hook(view_grad);
  last_layer.register_forward_hook(register_for_backprop_hook)
  all_L = True
  one_deep = names["features.2"]
  #one_deep.register_forward_hook(register_for_backprop_hook); one_deep.register_backward_hook(save_vis)
  #l = []
  for i, child in enumerate([module for module in vgg16model.modules()]):
    if "Conv2d" in child.__class__.__name__  and child.__dict__["in_channels"] != 3:
      child.register_forward_hook(register_for_backprop_hook); child.register_backward_hook(view_grad_per_layer)
    """   for backward guided :  elif "ReLU" in child.__class__.__name__:
      l.append(GuidedBackpropReLUObj())
    elif child.__class__.__name__ == "name" : l.append(child)
    elif not isinstance(child, nn.Sequential): l.append(child)
    elif i == 0: continue"""
  #vgg16model = nn.Sequential(*l)
  r = tv.transforms.Resize((224,224)) #rescale for expected input dimensions
  for i, image in enumerate(images):
    output, preds = gettop(r(image), vgg16model)
    for j in range(args["top_preds"]):
      NCOLS = 12
      NITERS = 1
      TITLE = str(preds[0][j][1:])
      if all_L:
        fig = plt.figure(figsize = (15,15))
        fig.canvas.set_window_title(TITLE)
        output.flatten()[
          output.flatten().argsort(descending = True)
        ][j].reshape(1,-1).backward(retain_graph =True) # calc grad from x predictor
        iw = ImshowWindow(fig); iw.show()
      else:
        output.flatten()[
          output.flatten().argsort(descending = True)
          ][j].reshape(1,-1).backward(retain_graph =True) # calc grad from x predictor
        plt.imshow(r(image).numpy().transpose(1,2,0))
        plt.imshow(getSaliency( 
          img = one_deep.cached_out,
          grad = one_deep.grad_cache[1],
          fm_card = 64,
          fm_dims = 224
          ).detach().numpy(), alpha = 0.5, cmap = "jet")
        plt.show()

