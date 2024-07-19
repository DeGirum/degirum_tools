import numpy as np
import json
from sys import float_info


class PostProcessor:
    def __init__(self, json_config):
        self._json_config = json.loads(json_config)
        self._label_json_config = self._json_config["POST_PROCESS"][0]["LabelsPath"]
        self._reg_scale = self._json_config["POST_PROCESS"][0].get("RegScale", 1.0)
        self._reg_offset = self._json_config["POST_PROCESS"][0].get("RegOffset", 0.0)
        with open(self._label_json_config, "r") as json_file:
            self._labels = json.load(json_file)

    def forward(self, tensor_list, details_list):
        qp = details_list[0]["quantization_parameters"]
        scale = qp["scales"][0]
        offset = qp["zero_points"][0]

        value_dequant = (float(np.squeeze(tensor_list[0])) - offset) * scale
        value = value_dequant * self._reg_scale + self._reg_offset

        if np.isnan(value):
            value = float_info.max

        if np.isinf(value):
            value = float(np.clip(value, float_info.min, float_info.max))

        results = [{"label": self._labels["0"], "score": value}]
        return json.dumps(results)
