import numpy as np
import json


class PostProcessor:
    def __init__(self, json_config):
        self._json_config = json.loads(json_config)
        self._label_json_config = self._json_config["POST_PROCESS"][0]["LabelsPath"]
        self._reg_scale = self._json_config["POST_PROCESS"][0]["RegScale"]
        self._reg_offset = self._json_config["POST_PROCESS"][0]["RegOffset"]
        with open(self._label_json_config, "r") as json_file:
            self._labels = json.load(json_file)

    def forward(self, tensor_list, details_list):
        qp = details_list[0]["quantization_parameters"]
        scale = qp["scales"][0]
        offset = qp["zero_points"][0]

        value_dequant = (float(np.squeeze(tensor_list[0])) - offset) * scale
        value = value_dequant * self._reg_scale + self._reg_offset

        results = [{"label": self._labels["0"], "score": value}]
        return json.dumps(results)
