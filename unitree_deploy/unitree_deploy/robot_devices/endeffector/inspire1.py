import os
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber

try:  # same driver package used by xr_teleoperate (`--ee inspire1` / `dftp`)
    from inspire_sdkpy import inspire_dds, inspire_hand_defaut
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`inspire_sdkpy` is not installed in this environment. Install the Inspire SDK used by "
        "xr_teleoperate (inspire_hand_ws/inspire_hand_sdk) into the unitree_deploy env."
    ) from e

from unitree_deploy.robot_devices.endeffector.configs import (
    Inspire1Config,
)

INSPIRE_NUM_MOTORS = 6  # pinky, ring, middle, index, thumb-bend, thumb-rotation
INSPIRE_RAW_MAX = 1000.0  # driver units: 0 = fully closed, 1000 = fully open

INSPIRE_THUMB_ROT_IDX = 5
THUMB_ROT_FIXED_VIRTUAL = 2.5  # valore fisso, stessa scala di virtual_open/virtual_closed

kTopicInspireCommand = {"right": "rt/inspire_hand/ctrl/r", "left": "rt/inspire_hand/ctrl/l"}
kTopicInspireState = {"right": "rt/inspire_hand/state/r", "left": "rt/inspire_hand/state/l"}


class _InspireDFXHand:
    """One Inspire hand through the Inspire DDS driver (same topics as xr_teleoperate).

    All values exposed here are normalized: 1.0 = open, 0.0 = closed (raw driver units / 1000).
    Left and right hands have separate topics, so each gripper owns its own publisher.
    """

    def __init__(self, side: str):
        self.side = side
        try:
            # No-op if the arm controller already initialized DDS in this process.
            ChannelFactoryInitialize(0, os.environ.get("UNITREE_NIC", "enp130s0"))
        except Exception:
            pass

        self._pub = ChannelPublisher(kTopicInspireCommand[side], inspire_dds.inspire_hand_ctrl)
        self._pub.Init()
        self._sub = ChannelSubscriber(kTopicInspireState[side], inspire_dds.inspire_hand_state)
        self._sub.Init()

        self._lock = threading.Lock()
        self._state = np.ones(INSPIRE_NUM_MOTORS, dtype=np.float32)
        self._has_state = False

        self._cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self._cmd.mode = 0b0001  # angle control

        threading.Thread(target=self._subscribe_state, name=f"inspire.{side}._subscribe_state", daemon=True).start()

    def _subscribe_state(self):
        while True:
            msg = self._sub.Read()
            if msg is not None:
                angles = getattr(msg, "angle_act", None)
                if angles is None:
                    raise AttributeError(
                        f"`inspire_hand_state` has no `angle_act` field; available: {[a for a in dir(msg) if not a.startswith('_')]}"
                    )
                q = np.asarray(list(angles)[:INSPIRE_NUM_MOTORS], dtype=np.float32) / INSPIRE_RAW_MAX
                with self._lock:
                    self._state = q
                    self._has_state = True
            time.sleep(0.002)

    def wait_for_state(self, timeout: float = 5.0):
        t0 = time.monotonic()
        while True:
            with self._lock:
                if self._has_state:
                    return
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"No message on `{kTopicInspireState[self.side]}` after {timeout}s. "
                    "Is the Inspire hand driver running (inspire_sdk.py on the laptop, or inspire_g1 on PC2)?"
                )
            time.sleep(0.01)

    def read(self) -> np.ndarray:
        with self._lock:
            return self._state.copy()

    def write(self, q: np.ndarray):
        q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
        for i in range(INSPIRE_NUM_MOTORS):
            self._cmd.angle_set[i] = int(round(q[i] * INSPIRE_RAW_MAX))
        self._pub.Write(self._cmd)


class Inspire1:
    def __init__(self, config: Inspire1Config):
        self.config = config
        self.side = config.side
        self.port = config.port
        self.motors = config.motors

        self.q_open = np.asarray(config.q_open, dtype=np.float32)
        self.q_closed = np.asarray(config.q_closed, dtype=np.float32)

        self.virtual_open = float(config.virtual_open)
        self.virtual_closed = float(config.virtual_closed)

        self.control_dt = config.control_dt
        self.max_joint_step = config.max_joint_step

        self._lock = threading.Lock()
        self._is_connected = False
        self._last_q = None
        self._last_time = None

        self.hand = None

        self.mock_value = getattr(config, "mock_value", None)
        self.is_mock = self.mock_value is not None

    @property
    def motor_names(self) -> list[str]:
        # Must return a list of length one
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [index for index, _ in self.motors.values()]

    def connect(self):
        if self._is_connected:
            return

        if self.is_mock:
            self._is_connected = True
            return

        # Replace with the actual Inspire SDK constructor.
        self.hand = self._create_inspire_connection(
            port=self.port,
            side=self.side,
        )

        self._is_connected = True
        self._last_q = self._read_hardware_positions()
        self._last_time = time.monotonic()

    def disconnect(self):
        if self.hand is not None:
            # Replace if the Inspire SDK provides a close method.
            close = getattr(self.hand, "close", None)
            if callable(close):
                close()

        self.hand = None
        self._is_connected = False

    def _create_inspire_connection(self, port: str, side: str):
        """
        Inspire hands are driven over DDS through the Inspire driver, so `port` is unused.
        """
        hand = _InspireDFXHand(side)
        hand.wait_for_state()
        return hand

    def _read_hardware_positions(self) -> np.ndarray:
        """
        Return the six Inspire actuator positions in the same order as
        q_open and q_closed.
        """
        return self.hand.read().astype(np.float32)

    def _write_hardware_positions(self, q_target: np.ndarray):
        """
        Send six target positions to the Inspire Hand.
        """
        self.hand.write(q_target)

    def _hardware_to_virtual(self, q: np.ndarray) -> float:
        mask = np.ones(INSPIRE_NUM_MOTORS, dtype=bool)
        mask[INSPIRE_THUMB_ROT_IDX] = False  # escluso: fissato, non informativo sull'apertura

        direction = (self.q_closed - self.q_open)[mask]
        denominator = float(np.dot(direction, direction))
        if denominator < 1e-8:
            raise ValueError("q_open and q_closed cannot be identical")

        alpha = float(np.dot((q - self.q_open)[mask], direction) / denominator)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        return self.virtual_open + alpha * (self.virtual_closed - self.virtual_open)

    def _virtual_to_hardware(self, virtual_q: float) -> np.ndarray:
        denominator = self.virtual_closed - self.virtual_open
        if abs(denominator) < 1e-8:
            raise ValueError("virtual_open and virtual_closed cannot be identical")

        alpha = float(np.clip((virtual_q - self.virtual_open) / denominator, 0.0, 1.0))
        hw_q = self.q_open + alpha * (self.q_closed - self.q_open)

        # Thumb-rotation: posizione fissa "fake gripper", indipendente dallo scalare della rete
        thumb_alpha = float(np.clip(
            (THUMB_ROT_FIXED_VIRTUAL - self.virtual_open) / denominator, 0.0, 1.0
        ))
        hw_q[INSPIRE_THUMB_ROT_IDX] = (
            self.q_open[INSPIRE_THUMB_ROT_IDX]
            + thumb_alpha * (self.q_closed[INSPIRE_THUMB_ROT_IDX] - self.q_open[INSPIRE_THUMB_ROT_IDX])
        )
        return hw_q

    def read_current_endeffector_q(self) -> np.ndarray:
        if self.is_mock:
            return np.asarray([self.mock_value], dtype=np.float32)

        q = self._read_hardware_positions()
        virtual_q = self._hardware_to_virtual(q)

        self._last_q = q.copy()
        self._last_time = time.monotonic()

        # One logical DoF
        return np.asarray([virtual_q], dtype=np.float32)

    def read_current_endeffector_dq(self) -> np.ndarray:
        if self.is_mock:
            return np.zeros(1, dtype=np.float32)

        now = time.monotonic()
        q = self._read_hardware_positions()

        if self._last_q is None or self._last_time is None:
            virtual_dq = 0.0
        else:
            dt = max(now - self._last_time, 1e-6)
            old_virtual = self._hardware_to_virtual(self._last_q)
            new_virtual = self._hardware_to_virtual(q)
            virtual_dq = (new_virtual - old_virtual) / dt

        self._last_q = q.copy()
        self._last_time = now

        return np.asarray([virtual_dq], dtype=np.float32)

    def write_endeffector(
        self,
        q_target,
        tauff_target=None,
        time_target=None,
        cmd_target=None,
    ):
        if self.is_mock:
            return

        virtual_target = float(np.asarray(q_target).reshape(-1)[0])
        desired_q = self._virtual_to_hardware(virtual_target)

        with self._lock:
            current_q = self._read_hardware_positions()

            # Rate limitation for the first real-robot tests
            safe_q = np.clip(
                desired_q,
                current_q - self.max_joint_step,
                current_q + self.max_joint_step,
            )

            # Never exceed the calibrated trajectory bounds
            lower = np.minimum(self.q_open, self.q_closed)
            upper = np.maximum(self.q_open, self.q_closed)
            safe_q = np.clip(safe_q, lower, upper)

            safe_q[5] = 2.5

            self._write_hardware_positions(safe_q)

    def go_start(self):
        value = (
            self.virtual_open
            if self.config.init_opening is None
            else self.config.init_opening
        )
        self.write_endeffector(np.asarray([value], dtype=np.float32))

    def go_home(self):
        self.write_endeffector(
            np.asarray([self.virtual_open], dtype=np.float32)
        )

    def endeffector_ik(self, value):
        return self._virtual_to_hardware(float(value))

    def retarget_to_endeffector(self, value):
        return self._virtual_to_hardware(float(value))