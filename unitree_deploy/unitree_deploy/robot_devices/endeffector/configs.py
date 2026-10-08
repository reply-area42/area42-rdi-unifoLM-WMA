import abc
from dataclasses import dataclass

import draccus
import numpy as np


@dataclass
class EndEffectorConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@EndEffectorConfig.register_subclass("dex_1")
@dataclass
class Dex1_GripperConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    init_pose: list | None = None
    control_dt: float = 1 / 200
    mock: bool = False
    max_pos_speed: float = 180 * (np.pi / 180) * 2
    topic_gripper_command: str = "rt/unitree_actuator/cmd"
    topic_gripper_state: str = "rt/unitree_actuator/state"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")

@EndEffectorConfig.register_subclass("inspire_virtual")
@dataclass
class InspireVirtualGripperConfig(EndEffectorConfig):
    side: str
    port: str

    # Exactly one logical motor
    motors: dict[str, tuple[int, str]]

    # Six-dimensional calibrated hardware poses
    q_open: tuple[float, float, float, float, float, float]
    q_closed: tuple[float, float, float, float, float, float]

    # Raw values used in the training dataset, before normalization
    virtual_open: float
    virtual_closed: float

    control_dt: float = 1 / 100
    max_joint_step: float = 0.03
    init_opening: float | None = None
    mock_value: float | None = None


    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        if len(self.motors) != 1:
            raise ValueError(
                "An Inspire virtual gripper must expose one logical motor"
            )

        if len(self.q_open) != 6 or len(self.q_closed) != 6:
            raise ValueError("q_open and q_closed must contain six values")

@EndEffectorConfig.register_subclass("inspire1")
@dataclass
class Inspire1Config(EndEffectorConfig):
    side: str
    port: str

    # Exactly one logical motor
    motors: dict[str, tuple[int, str]]

    # Six-dimensional calibrated hardware poses
    q_open: tuple[float, float, float, float, float, float]
    q_closed: tuple[float, float, float, float, float, float]

    # Raw values used in the training dataset, before normalization
    virtual_open: float
    virtual_closed: float

    control_dt: float = 1 / 100
    max_joint_step: float = 0.03
    init_opening: float | None = None
    mock_value: float | None = None


    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        if len(self.motors) != 1:
            raise ValueError(
                "An Inspire virtual gripper must expose one logical motor"
            )

        if len(self.q_open) != 6 or len(self.q_closed) != 6:
            raise ValueError("q_open and q_closed must contain six values")
