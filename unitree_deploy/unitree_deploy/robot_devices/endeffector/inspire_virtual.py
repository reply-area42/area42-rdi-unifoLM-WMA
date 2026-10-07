import threading
import time

import numpy as np

from unitree_deploy.robot_devices.endeffector.configs import (
    InspireVirtualGripperConfig,
)

class InspireVirtualGripper:
    def __init__(self, config: InspireVirtualGripperConfig):
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
        Instantiate the Inspire SDK, CAN or serial interface here.
        This is the only hardware-specific constructor.
        """
        raise NotImplementedError(
            "Implement connection using the Inspire Hand SDK"
        )

    def _read_hardware_positions(self) -> np.ndarray:
        """
        Return the six Inspire actuator positions in the same order as
        q_open and q_closed.
        """
        # Example:
        # q = self.hand.get_positions()
        # return np.asarray(q, dtype=np.float32)

        raise NotImplementedError(
            "Implement position reading using the Inspire Hand SDK"
        )

    def _write_hardware_positions(self, q_target: np.ndarray):
        """
        Send six target positions to the Inspire Hand.
        """
        # Example:
        # self.hand.set_positions(q_target.tolist())

        raise NotImplementedError(
            "Implement position commands using the Inspire Hand SDK"
        )

    def _hardware_to_virtual(self, q: np.ndarray) -> float:
        """
        Project the measured six-joint pose onto the calibrated
        open-to-closed grasp trajectory.
        """
        direction = self.q_closed - self.q_open
        denominator = float(np.dot(direction, direction))

        if denominator < 1e-8:
            raise ValueError("q_open and q_closed cannot be identical")

        alpha = float(
            np.dot(q - self.q_open, direction) / denominator
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))

        return (
            self.virtual_open
            + alpha * (self.virtual_closed - self.virtual_open)
        )

    def _virtual_to_hardware(self, virtual_q: float) -> np.ndarray:
        """
        Convert the raw model gripper value into six Inspire targets.
        """
        denominator = self.virtual_closed - self.virtual_open

        if abs(denominator) < 1e-8:
            raise ValueError(
                "virtual_open and virtual_closed cannot be identical"
            )

        alpha = (virtual_q - self.virtual_open) / denominator
        alpha = float(np.clip(alpha, 0.0, 1.0))

        return self.q_open + alpha * (self.q_closed - self.q_open)

    def read_current_endeffector_q(self) -> np.ndarray:
        q = self._read_hardware_positions()
        virtual_q = self._hardware_to_virtual(q)

        self._last_q = q.copy()
        self._last_time = time.monotonic()

        # One logical DoF
        return np.asarray([virtual_q], dtype=np.float32)

    def read_current_endeffector_dq(self) -> np.ndarray:
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