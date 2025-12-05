from typing import Union, Any
from typing import Callable, Iterator
from nptyping import NDArray, Shape, Number, Bool, UnsignedInteger

ListPoint2D             = NDArray[Shape['N, 2'], Number]

BoolField2D             = NDArray[Shape['X, Y'], Bool] 
NumberField2D           = NDArray[Shape['X, Y'], Number]
ScalarField2D           = NDArray[Shape['X, Y'], Any]

VectorField3D           = NDArray[Shape['X, Y, 3'], Number]
NumerPoint3D            = NDArray[Shape['3'], Number]

GrayImageLikeArray      = NDArray[Shape['W, H, 1'], Number]
LAImageLikeArray        = NDArray[Shape['W, H, 2'], UnsignedInteger]
RGBImageLikeArray       = NDArray[Shape['W, H, 3'], UnsignedInteger]
RGBAImageLikeArray      = NDArray[Shape['W, H, 4'], UnsignedInteger]
ImageLikeArray          = Union[
                            GrayImageLikeArray,
                            LAImageLikeArray,
                            RGBImageLikeArray,
                            RGBAImageLikeArray,    
                        ]
