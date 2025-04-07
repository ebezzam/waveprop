from enum import Enum
from turtle import pu
import numpy as np

""" SLMs """


class SLMOptions(Enum):
    ADAFRUIT = "adafruit"
    HOLOEYE_LC2012 = "holoeye_lc2012"

    @staticmethod
    def values():
        return [dev.value for dev in SLMOptions]


class SLMParam:
    NAME = "name"
    CELL_SIZE = "cell_size"
    SHAPE = "shape"
    SIZE = "size"
    DEADSPACE = "deadspace"
    FILL_FACTOR = "fill_factor"  # % of SLM pixel that is active
    PITCH = "pitch"
    PHASE = "phase"
    # (optional) color filter on top of SLM, each filter should be length-3 tuple for (R, G, B)
    COLOR_FILTER = "color_filter"


slm_dict = {
    # 1.8 inch RGB display by Adafruit: https://learn.adafruit.com/1-8-tft-display/overview
    # datasheet: https://cdn-shop.adafruit.com/datasheets/JD-T1800.pdf
    SLMOptions.ADAFRUIT.value: {
        SLMParam.NAME: SLMOptions.ADAFRUIT,
        SLMParam.CELL_SIZE: np.array([0.06e-3, 0.18e-3]),  # RGB sub-pixel
        SLMParam.SHAPE: np.array([128 * 3, 160]),
        SLMParam.SIZE: np.array([28.03e-3, 35.04e-3]),
        SLMParam.PHASE: False,
        # SLMParam.COLOR_ORDER: np.array([0, 1, 2]),
        # (3x1) color filter
        SLMParam.COLOR_FILTER: np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])[:, np.newaxis],
        # # E.G. for 2D color filter
        # SLMParam.COLOR_FILTER: np.array(
        #     [
        #         [(1, 0, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 1)]
        #     ]
        # ),
    },
    # https://holoeye.com/lc-2012-spatial-light-modulator/
    SLMOptions.HOLOEYE_LC2012.value: {
        SLMParam.NAME: SLMOptions.HOLOEYE_LC2012,
        SLMParam.PITCH: np.array([36e-6, 36e-6]),
        SLMParam.FILL_FACTOR: 0.58,
        SLMParam.SHAPE: np.array([768, 1024]),
        SLMParam.SIZE: np.array([27.6e-3, 36.9e-3]),
        SLMParam.PHASE: True,
    },
}

# # derive parameters from others
# for _key in slm_dict:
#     _config = slm_dict[_key]

#     # deadspace
#     if SLMParam.DEADSPACE not in _config.keys():
#         if SLMParam.CELL_SIZE in _config.keys():
#             _config[SLMParam.DEADSPACE] = (
#                 _config[SLMParam.SIZE] - _config[SLMParam.CELL_SIZE] * _config[SLMParam.SHAPE]
#             ) / (_config[SLMParam.SHAPE] - 1)
#         elif SLMParam.FILL_FACTOR in _config.keys() and SLMParam.PITCH in _config.keys():
#             _config[SLMParam.DEADSPACE] = _config[SLMParam.PITCH] * (
#                 1 - _config[SLMParam.FILL_FACTOR]
#             )
#         else:
#             raise ValueError("Cannot derive deadspace from given parameters.")

#     # cell size
#     if SLMParam.CELL_SIZE not in _config.keys():
#         _config[SLMParam.CELL_SIZE] = _config[SLMParam.PITCH] * _config[SLMParam.FILL_FACTOR]

#     # pitch
#     if SLMParam.PITCH not in _config.keys():
#         _config[SLMParam.PITCH] = _config[SLMParam.CELL_SIZE] + _config[SLMParam.DEADSPACE]

#     # fill factor
#     if SLMParam.FILL_FACTOR not in _config.keys():
#         _config[SLMParam.FILL_FACTOR] = 1 - _config[SLMParam.DEADSPACE] / _config[SLMParam.PITCH]

#     if SLMParam.COLOR_FILTER in _config.keys():
#         assert len(_config[SLMParam.COLOR_FILTER].shape) == 3


""" Camera sensors """


class SensorOptions(Enum):
    RPI_HQ = "rpi_hq"

    @staticmethod
    def values():
        return [dev.value for dev in SensorOptions]


class SensorParam:
    PIXEL_SIZE = "pixel_size"
    SHAPE = "shape"
    DIAGONAL = "diagonal"
    SIZE = "size"


sensor_dict = {
    # HQ Camera Sensor by Raspberry Pi
    # datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
    SensorOptions.RPI_HQ.value: {
        SensorParam.PIXEL_SIZE: np.array([1.55e-6, 1.55e-6]),
        SensorParam.SHAPE: np.array([3040, 4056]),
        SensorParam.DIAGONAL: 7.857e-3,
    }
}

# derived parameters
for _key in sensor_dict:
    _config = sensor_dict[_key]
    if SensorParam.DIAGONAL in _config.keys():
        # take into account possible deadspace
        _config[SensorParam.SIZE] = (
            _config[SensorParam.DIAGONAL]
            / np.linalg.norm(_config[SensorParam.SHAPE])
            * _config[SensorParam.SHAPE]
        )
    else:
        _config[SensorParam.SIZE] = _config[SensorParam.PIXEL_SIZE] * _config[SensorParam.SHAPE]
