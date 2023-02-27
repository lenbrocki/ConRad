import torchvision.transforms.functional as TF
import random
# randomly pick one of provided angles to rotate
class MyRotation():
    def __init__(self, angles):
        self.angles = angles
        
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)