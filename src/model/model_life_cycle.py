from typing import Literal
import tensorflow as tf


def save_model(
    model: tf.keras.models.Sequential,
    save_name: str,
    model_format: Literal["h5", "keras"] = "keras",
) -> None:
    # serialize model to json
    # model_json = model.to_json()
    # with open(savename+".json", "w") as json_file:
    #     json_file.write(model_json)
    # print("json Model ",savename,".json saved to disk")

    # # serialize weights to HDF5

    model.save(f"{save_name}.{model_format}")

    print(f"Model {save_name}.{model_format} saved to disk")


def load_model(save_name: str, model_format: Literal["h5", "keras"] = "keras")->tf.keras.models.Sequential:
    # # print "Loading json model ",saveName
    # with open(savename + ".json", "r") as json_file:
    #     model = model_from_json(json_file.read())
    # print("json Model ", savename, ".json loaded ")
    # model.load_weights(savename + ".h5")
    # print("Weights ", savename, ".h5 loaded ")
    # return model
    return tf.keras.models.load_model(f"{save_name}.{model_format}")
