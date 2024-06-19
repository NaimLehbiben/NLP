import pickle
from typing import Dict
import numpy as np
import numpy.typing as npt


def encoded_image_loader() -> Dict[str, npt.NDArray[np.float64]]:
    return pickle.load(open("./data/encoded_images_PCA.p", "rb"))
