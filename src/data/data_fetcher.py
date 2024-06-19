import os
from requests import get  # to make GET request


def download(url: str, file_name: str) -> None:
    """Download a dataset

    Args:
    ----
        url (str): The url used to fetch the data
        file_name (str): The filename to used to store the result
    """
    if os.path.exists(file_name) is False:
        # open in binary mode
        with open(file_name, "wb") as file:
            # get request
            response = get(url)
            # write to file
            file.write(response.content)
