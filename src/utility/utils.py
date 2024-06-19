from typing import Union
import numpy as np
import numpy.typing as npt


def sampling(preds: npt.NDArray[np.float64], temperature: float = 1.0) -> np.intp:
    preds = np.asarray(preds).astype("float64")
    predsN = pow(preds, 1.0 / temperature)
    predsN /= np.sum(predsN)
    probas = np.random.multinomial(1, predsN, 1)
    return np.argmax(probas)
