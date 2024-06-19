from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class DataPreprocessor:
    def __init__(
        self,
        train_dataset_file: str = "./data/flickr_8k_train_dataset.txt",
        test_dataset_file: str = "./data/flickr_8k_test_dataset.txt",
        nbkeep: int = 100,
    ):
        self.__nbkeep = nbkeep
        self.__train_dataset_file = train_dataset_file
        self.__FLICKR_8K_TRAIN_DATASET_DF = pd.read_csv(
            self.__train_dataset_file, delimiter="\t"
        )
        self.__test_dataset_file = test_dataset_file
        self.__FLICKR_8K_TEST_DATASET_DF = pd.read_csv(
            self.__test_dataset_file, delimiter="\t"
        )

    def prepare_train_word_dictionary(self) -> Tuple[int, List[Tuple[int, str]]]:
        bow = {}
        nbwords = 0

        for _, row in tqdm(
            self.__FLICKR_8K_TRAIN_DATASET_DF.iterrows(),
            total=self.__FLICKR_8K_TRAIN_DATASET_DF.shape[0],
            leave=False,
            desc=f"iterating over {self.__train_dataset_file}...",
        ):
            # split caption into words and remove capital letters
            cap_wordsl = [w.lower() for w in row["captions"].split()]
            nbwords += len(cap_wordsl)
            for w in cap_wordsl:
                if w in bow:
                    bow[w] = bow[w] + 1
                else:
                    bow[w] = 1

        return nbwords, sorted(
            [(value, key) for (key, value) in bow.items()], reverse=True
        )

    @property
    def train_dataframe(self) -> pd.DataFrame:
        return self.__FLICKR_8K_TRAIN_DATASET_DF

    @property
    def train_image_id_as_list(self) -> List[str]:
        return self.__FLICKR_8K_TRAIN_DATASET_DF["image_id"].to_list()

    @property
    def train_captions_as_list(self) -> List[str]:
        return self.__FLICKR_8K_TRAIN_DATASET_DF["captions"].to_list()

    @property
    def test_dataframe(self) -> pd.DataFrame:
        return self.__FLICKR_8K_TEST_DATASET_DF

    @property
    def test_image_id_as_list(self) -> List[str]:
        return self.__FLICKR_8K_TEST_DATASET_DF["image_id"].to_list()

    @property
    def test_captions_as_list(self) -> List[str]:
        return self.__FLICKR_8K_TEST_DATASET_DF["captions"].to_list()

    @property
    def X_train(self) -> npt.NDArray[np.float64]:
        npzfile = np.load(f"./data/Training_data_{self.__nbkeep}.npz")
        return npzfile["X_train"]

    @property
    def X_test(self) -> npt.NDArray[np.float64]:
        # LOADING TEST DATA
        npzfile = np.load(f"./data/Test_data_{self.__nbkeep}.npz")
        return npzfile["X_test"]

    @property
    def Y_train(self) -> npt.NDArray[np.float64]:
        npzfile = np.load(f"./data/Training_data_{self.__nbkeep}.npz")
        return npzfile["Y_train"]

    @property
    def Y_test(self) -> npt.NDArray[np.float64]:
        # LOADING TEST DATA
        npzfile = np.load(f"./data/Test_data_{self.__nbkeep}.npz")
        return npzfile["Y_test"]

    def get_max_length_caption(self, listwords: List[str]) -> int:
        """Given a list of words this function outputs the length of the max length captions (used as input of the model for instance)

        Args:
            listwords (List[str]): The list of wanted words.

        Returns:
            int: The max length
        """
        maxLCap = 0

        for caption in self.train_captions_as_list:
            l = 0
            words_in_caption = caption.split()
            for j in range(len(words_in_caption) - 1):
                current_w = words_in_caption[j].lower()
                if current_w in listwords:
                    l += 1
                if l > maxLCap:
                    maxLCap = l
        return maxLCap

    def construct_x_train_y_train(
        self,
        encoded_images: Dict[str, npt.NDArray[np.float64]],
        word_embeddings: npt.NDArray[np.float64],
        listwords: List[str],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        X_train, Y_train = self.__construct_x_y(
            caps=self.train_captions_as_list,
            imgs=self.train_image_id_as_list,
            encoded_images=encoded_images,
            word_embeddings=word_embeddings,
            listwords=listwords,
        )
        np.savez(
            f"./data/Training_data_{self.__nbkeep}", X_train=X_train, Y_train=Y_train
        )  # Saving tensor
        print(
            f"tensor X_train, Y_train saved at : ./data/Training_data_{self.__nbkeep}"
        )
        return X_train, Y_train

    def construct_x_test_y_test(
        self,
        encoded_images: Dict[str, npt.NDArray[np.float64]],
        word_embeddings: npt.NDArray[np.float64],
        listwords: List[str],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        X_test, Y_test = self.__construct_x_y(
            caps=self.test_captions_as_list,
            imgs=self.test_image_id_as_list,
            encoded_images=encoded_images,
            word_embeddings=word_embeddings,
            listwords=listwords,
        )
        np.savez(
            f"./data/Test_data_{self.__nbkeep}", X_test=X_test, Y_test=Y_test
        )  # Saving tensor
        print(f"tensor X_test, Y_test saved at : ./data/Test_data_{self.__nbkeep}")
        return X_test, Y_test

    def __construct_x_y(
        self,
        caps: List[str],
        imgs: List[str],
        encoded_images: Dict[str, npt.NDArray[np.float64]],
        word_embeddings: npt.NDArray[np.float64],
        listwords: List[str],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Private function that builds a set of X and Y dataset given a caption list and a image_id list.

        Args:
        ----
            caps (List[str]): _description_
            imgs (List[str]): _description_
            encoded_images (Dict[str, npt.NDArray[np.float64]]): _description_
            word_embeddings (npt.NDArray[np.float64]): _description_
            listwords (List[str]): _description_

        Returns:
        ----
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: _description_
        """
        maxLCap = self.get_max_length_caption(listwords)
        nb = len(imgs)
        indexwords = {listwords[i]: i for i in range(len(listwords))}

        tinput = 202
        ll = 50
        # binmat = np.zeros((ll,39))
        nbtot = 0
        nbkept = 0
        # IGNORE START => we will never predict <start> .
        tVocabulary = len(listwords)

        X = np.zeros((nb, maxLCap, tinput))
        Y = np.zeros((nb, maxLCap, tVocabulary), bool)

        for i in range(nb):
            words_in_caption = caps[i].split()

            nbtot += len(words_in_caption) - 1
            indseq = 0
            for j in range(len(words_in_caption) - 1):
                current_w = words_in_caption[j].lower()

                if j == 0 and current_w != "<start>":
                    print("PROBLEM")
                if current_w in listwords:
                    X[i, indseq, 0:100] = encoded_images[imgs[i]]
                    X[i, indseq, 100:202] = word_embeddings[indexwords[current_w]]

                next_w = words_in_caption[j + 1].lower()

                # print("current_w="+str(current_w)+" next_w="+str(next_w)+" indseq="+str(indseq))
                index_pred = 0
                if next_w in listwords:
                    nbkept += 1
                    index_pred = indexwords[next_w]
                    Y[i, indseq, index_pred] = 1
                    indseq += 1
        return X, Y
