from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import random


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),  # set-1
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor), # set-3
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),    #set-11
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor), # set-2 
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
            
            SubPolicy(0.4, "solarize",  5, 0.9, "autocontrast", 3, fillcolor), 
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast",  2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize",  8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
   
            SubPolicy(0.4, "solarize",  5, 0.9, "autocontrast", 1, fillcolor), 
            SubPolicy(0.8, "translateY",  9, 0.9, "translateY", 9, fillcolor),
            SubPolicy(0.8, "autocontrast",  0, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.2, "translateY",  7, 0.9, "color", 6, fillcolor),
            SubPolicy(0.7, "equalize",  6, 0.4, "color", 9, fillcolor),
             
            SubPolicy(0.3, "brightness",  7, 0.5, "autocontrast", 8, fillcolor), 
            SubPolicy(0.9, "autocontrast",  4, 0.5, "autocontrast", 6, fillcolor),
            SubPolicy(0.3, "solarize",  5, 0.6, "equalize", 5, fillcolor),
            SubPolicy(0.2, "translateY",  4, 0.3, "sharpness", 3, fillcolor),
            SubPolicy(0.0, "brightness",  8, 0.8, "color", 8, fillcolor),

            SubPolicy(0.2, "solarize",  6, 0.8, "color", 6, fillcolor), 
            SubPolicy(0.2, "solarize",  6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.4, "solarize",  1, 0.6, "equalize", 5, fillcolor),
            SubPolicy(0.0, "brightness",  0, 0.5, "solarize", 2, fillcolor),
            SubPolicy(0.9, "autocontrast",  5, 0.5, "brightness", 3, fillcolor),

            SubPolicy(0.7, "contrast",  5, 0.0, "brightness", 2, fillcolor), 
            SubPolicy(0.2, "solarize",  8, 0.1, "solarize", 5, fillcolor),
            SubPolicy(0.5, "contrast",  1, 0.2, "translateY", 9, fillcolor),
            SubPolicy(0.6, "autocontrast",  5, 0.0, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast",  4, 0.8, "equalize", 4, fillcolor),
            
            SubPolicy(0.0, "brightness",  7, 0.4, "equalize", 7, fillcolor), 
            SubPolicy(0.2, "solarize",  5, 0.7, "equalize", 5, fillcolor),
            SubPolicy(0.6, "equalize",  8, 0.6, "color", 2, fillcolor),
            SubPolicy(0.3, "color",  7, 0.2, "color", 4, fillcolor),
            SubPolicy(0.5, "autocontrast",  2, 0.7, "solarize", 2, fillcolor),
            
            SubPolicy(0.2, "autocontrast",  0, 0.1, "equalize", 0, fillcolor), 
            SubPolicy(0.6, "shearY",  5, 0.6, "equalize", 5, fillcolor),
            SubPolicy(0.9, "brightness",  3, 0.4, "autocontrast", 1, fillcolor),
            SubPolicy(0.8, "equalize",  8, 0.7, "equalize", 7, fillcolor),
            SubPolicy(0.7, "equalize",  7, 0.5, "solarize", 0, fillcolor),
            
            SubPolicy(0.8, "equalize",  4, 0.8, "translateY", 9, fillcolor), 
            SubPolicy(0.8, "translateY",  9, 0.6, "translateY", 9, fillcolor),
            SubPolicy(0.9, "translateY",  0, 0.5, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast",  3, 0.3, "solarize", 4, fillcolor),
            SubPolicy(0.5, "solarize",  3, 0.4, "equalize", 4, fillcolor),
            
            SubPolicy(0.1, "autocontrast",  5, 0.0, "brightness", 0, fillcolor), 
            SubPolicy(0.7, "equalize",  7, 0.6, "autocontrast", 4, fillcolor),
            SubPolicy(0.1, "color",  8, 0.2, "shearY", 3, fillcolor),
            SubPolicy(0.4, "shearY",  2, 0.7, "rotate", 0, fillcolor),
            
            SubPolicy(0.1, "shearY",  3, 0.9, "autocontrast", 5, fillcolor), 
            SubPolicy(0.5, "equalize",  0, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.3, "autocontrast",  5, 0.2, "rotate", 7, fillcolor),
            SubPolicy(0.8, "equalize",  2, 0.4, "invert", 0, fillcolor),
            
            SubPolicy(0.9, "equalize",  5, 0.7, "color", 0, fillcolor), 
            SubPolicy(0.1, "equalize",  1, 0.1, "shearY", 3, fillcolor),
            SubPolicy(0.7, "autocontrast",  3, 0.7, "equalize", 0, fillcolor),
            SubPolicy(0.5, "brightness",  1, 0.1, "contrast", 7, fillcolor),
            SubPolicy(0.1, "contrast",  4, 0.6, "solarize", 5, fillcolor),
            
            SubPolicy(0.2, "solarize",  3, 0.0, "shearX", 0, fillcolor), 
            SubPolicy(0.3, "translateX",  0, 0.6, "translateX", 0, fillcolor),
            SubPolicy(0.5, "equalize",  9, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.1, "shearX",  0, 0.5, "sharpness", 1, fillcolor),
            SubPolicy(0.8, "equalize",  6, 0.3, "invert", 6, fillcolor),
            
            SubPolicy(0.4, "shearX",  4, 0.9, "autocontrast", 2, fillcolor),
            SubPolicy(0.0, "shearX",  3, 0.0, "posterize", 3, fillcolor),
            SubPolicy(0.4, "solarize",  3, 0.2, "color", 4, fillcolor),
            SubPolicy(0.1, "equalize",  4, 0.7, "equalize", 6, fillcolor),
            
            SubPolicy(0.3, "equalize",  8, 0.4, "autocontrast", 3, fillcolor), 
            SubPolicy(0.6, "solarize",  4, 0.7, "autocontrast", 6, fillcolor),
            SubPolicy(0.2, "autocontrast",  9, 0.4, "brightness", 8, fillcolor),
            SubPolicy(0.1, "equalize",  0, 0.0, "equalize", 6, fillcolor),
            SubPolicy(0.8, "equalize",  4, 0.0, "equalize", 4, fillcolor),
            
            SubPolicy(0.5, "equalize",  5, 0.1, "autocontrast", 2, fillcolor), 
            SubPolicy(0.5, "solarize",  5, 0.9, "autocontrast", 5, fillcolor),
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int64),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "cutout": np.linspace(0.0, 0.2, 10),
        }
    
        def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
            #assert 0.0 <= v <= 0.2
            if v <= 0.:
                return img

            v = v * img.size[0]

            return CutoutAbs(img, v)

            # x0 = np.random.uniform(w - v)
            # y0 = np.random.uniform(h - v)
            # xy = (x0, y0, x0 + v, y0 + v)
            # color = (127, 127, 127)
            # img = img.copy()
            # PIL.ImageDraw.Draw(img).rectangle(xy, color)
            # return img


        def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
            # assert 0 <= v <= 20
            if v < 0:
                return img
            w, h = img.size
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)

            x0 = int(max(0, x0 - v / 2.))
            y0 = int(max(0, y0 - v / 2.))
            x1 = min(w, x0 + v)
            y1 = min(h, y0 + v)

            xy = (x0, y0, x1, y1)
            color = (125, 123, 114)
            # color = (0, 0, 0)
            img = img.copy()
            ImageDraw.Draw(img).rectangle(xy, color)
            return img

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "cutout": lambda img, magnitude: Cutout(img, magnitude),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img