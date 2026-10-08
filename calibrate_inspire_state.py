"""Print the Inspire hand state (rt/inspire/state) to calibrate q_open / q_closed.

Usage: with the hand in the wanted pose (e.g. held there by xr_teleoperate --ee inspire1),
run this script, press Enter to print the 6 values of each hand, Ctrl+C to exit.
Order per hand: pinky, ring, middle, index, thumb-bend, thumb-rotation (1.0 open, 0.0 closed).
"""
import os
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

ChannelFactoryInitialize(0, os.environ.get("UNITREE_NIC", "enp130s0"))
sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
sub.Init()

np.set_printoptions(precision=3, suppress=True)
while True:
    input("Enter to sample... ")
    msg = None
    t0 = time.time()
    while msg is None and time.time() - t0 < 3:
        msg = sub.Read()
    if msg is None:
        print("no state received")
        continue
    q = np.array([msg.states[i].q for i in range(12)])
    print("right:", tuple(round(float(x), 3) for x in q[:6]))
    print("left: ", tuple(round(float(x), 3) for x in q[6:]))
