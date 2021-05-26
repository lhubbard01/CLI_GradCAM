__CLI GradCAM__
The primary purpose of this script is to aid in model interpretability. The implementation is based off the paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". The script requires keras for its applications package that provides useful output information from the model, given it is trained on ImageNet (the model used in this program was). 
This is still a work in progress but it is fun to tool around with, provide it with pictures of your own. 

It is launched through your typical python launch call "python3 impl.py". Throw in a " --help" to get a read out of expected arguments for the runtime.  There are many inefficiencies within this module, but I just threw it up here for the curious.

