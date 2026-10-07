import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ChannelFactoryInitialize(0, "enp130s0")

subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init()

print("Waiting for rt/lowstate...")

while True:
    msg = subscriber.Read()

    if msg is not None:
        print(
            "Received:",
            "mode_machine =", msg.mode_machine,
            "q0 =", msg.motor_state[0].q,
        )
        break

    time.sleep(0.01)