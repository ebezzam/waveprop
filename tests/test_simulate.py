from waveprop.simulation import FarFieldSimulator
import torch
from waveprop.devices import sensor_dict, SensorParam


def test_far_field_simulator():

    sensor = "rpi_hq"
    sensor_shape = sensor_dict[sensor][SensorParam.SHAPE]


    sim = FarFieldSimulator(
        object_height=30e-2,
        scene2mask=30e-2,
        mask2sensor=4e-3,
        sensor=sensor,
        output_dim=sensor_shape,
    )

    obj = torch.rand(1, 1, 256, 256)
    image = sim.propagate(obj)
    assert image.shape == (1, 1, *sensor_shape)


if __name__ == "__main__":
    test_far_field_simulator()