from PIL.ImageOps import grayscale as pil_grayscale
from PIL.ImageOps import autocontrast as pil_autocontrast
from utils.functions import image_from_imagearr, imagearr_from_image
from utils.imiltyping import * 
import numpy as np

def grayscale(input_image: ImageLikeArray) -> NumberField2D:
    """Returns an inference of depth in the image

    Args:
        input_image (ImageLikeArray): Image array

    Returns:
        NumberScalarField2D: A scalar field o relative depth of the same resolution as input
    """
    img             = image_from_imagearr(input_image)
    gray            = pil_grayscale(img)
    enhanced_map    = pil_autocontrast(gray)
    ret             = np.asarray(enhanced_map)
    return ret


