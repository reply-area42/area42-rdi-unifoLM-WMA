"""
G1 Emergency Zero Torque
========================
USO:
  Modalità armata (consigliata, lanciare PRIMA dei test):
    python3 g1_estop.py eth0
    -> premere INVIO per inviare subito Zero Torque

  Invio immediato:
    python3 g1_estop.py eth0 --now
"""

import sys
import time
import argparse
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ZERO_TORQUE = 0
DAMPING = 1

RPC_TIMEOUT = 1.0       # timeout breve per ogni chiamata RPC
CONFIRM_TIMEOUT = 1.0   # attesa massima per la conferma dello stato
MAX_ATTEMPTS = 5        # tentativi prima di arrendersi

def get_fsm_id(client: LocoClient) -> Optional[int]:
    try:
        code, val = client.GetFsmId()
        return int(val) if code == 0 and val is not None else None
    except Exception:
        return None

def send_fsm(client: LocoClient, fsm_id: int) -> int:
    try:
        return client.SetFsmId(fsm_id)
    except Exception as e:
        print(f"  ERRORE invio SetFsmId({fsm_id}): {e}")
        return -1

def wait_for(client: LocoClient, target: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_fsm_id(client) == target:
            return True
        time.sleep(0.05)
    return False

def emergency_zero_torque(client: LocoClient) -> bool:
    t0 = time.time()
    print("\n!!! EMERGENZA: invio Zero Torque !!!")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 1) Zero Torque diretto, senza attendere lo stato statico
        code = send_fsm(client, ZERO_TORQUE)
        print(f"  [{attempt}] SetFsmId(0) -> codice {code}")
        if wait_for(client, ZERO_TORQUE, CONFIRM_TIMEOUT):
            print(f"  OK: Zero Torque confermato in {time.time() - t0:.2f} s")
            return True

        # 2) Riserva: Damping -> Zero Torque
        print("  Zero Torque non confermato, provo Damping -> Zero Torque")
        send_fsm(client, DAMPING)
        wait_for(client, DAMPING, 0.3)
        send_fsm(client, ZERO_TORQUE)
        if wait_for(client, ZERO_TORQUE, CONFIRM_TIMEOUT):
            print(f"  OK: Zero Torque confermato in {time.time() - t0:.2f} s")
            return True

    print(f"  FALLITO: stato attuale fsm_id={get_fsm_id(client)}. "
          "USARE IL TELECOMANDO / E-STOP FISICO.")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("iface", help="Interfaccia di rete (es. eth0)")
    parser.add_argument("--now", action="store_true",
                        help="Invia subito Zero Torque, senza attendere INVIO")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    client = LocoClient()
    client.SetTimeout(RPC_TIMEOUT)
    client.Init()

    if args.now:
        sys.exit(0 if emergency_zero_torque(client) else 1)

    print(f"E-STOP ARMATO (fsm_id attuale: {get_fsm_id(client)})")
    print("Premi INVIO per Zero Torque immediato, Ctrl+C per uscire.")
    try:
        while True:
            input()
            emergency_zero_torque(client)
            print("\nE-STOP ancora armato. INVIO per reinviare, Ctrl+C per uscire.")
    except (KeyboardInterrupt, EOFError):
        print("\nE-STOP disarmato.")

if __name__ == "__main__":
    main()