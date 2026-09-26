"""Interactive record/replay for the SO-101 (leader drives follower).

Run:  python record_replay.py

Commands:
  record <name>   press ENTER to start recording, ENTER again to stop -> recordings/<name>.rec.npz
  replay <name>   replay recordings/<name>.rec.npz on the follower
  end             disconnect and exit
Ctrl-C during a record/replay aborts just that action.
"""
import select
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# --- EDIT THESE to your ports (from `lerobot-find-port`) and calibrated ids ---
FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT   = "/dev/ttyACM1"
FOLLOWER_ID   = "follower"
LEADER_ID     = "leader"
FPS           = 30
EASE_SECONDS  = 2.0   # how long the follower takes to glide to a start pose
REC_DIR       = Path(__file__).resolve().parent / "recordings"


def _pace(tick_start: float, fps: int = FPS) -> None:
    """Sleep so each loop iteration lasts ~1/fps seconds."""
    time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - tick_start)))


def _rec_path(name: str) -> Path:
    """recordings/<name>.rec.npz — accepts a bare name or one with the suffix."""
    name = Path(name).name.removesuffix(".npz").removesuffix(".rec")
    return REC_DIR / f"{name}.rec.npz"


def enter_pressed() -> bool:
    """Non-blocking check for a line on stdin (consumes it)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def ease_to(follower, target: dict, seconds: float = EASE_SECONDS) -> None:
    """Linearly move the follower from its current pose to `target` instead of snapping."""
    obs = follower.get_observation()
    start = {k: float(obs[k]) for k in target}
    steps = max(1, int(seconds * FPS))
    for i in range(1, steps + 1):
        tick = time.perf_counter()
        a = i / steps
        follower.send_action({k: start[k] + a * (float(target[k]) - start[k]) for k in target})
        _pace(tick)


def teleop_until_enter(leader, follower, frames=None, joints=None) -> None:
    """Mirror leader -> follower until ENTER; append rows to `frames` if given."""
    while True:
        tick = time.perf_counter()
        action = leader.get_action()
        follower.send_action(action)
        if frames is not None:
            frames.append([action[j] for j in joints])
        if enter_pressed():
            return
        _pace(tick)


def record(leader, follower, name: str) -> None:
    path = _rec_path(name)
    REC_DIR.mkdir(exist_ok=True)
    start_pose = leader.get_action()
    joints = list(start_pose.keys())  # fix a stable column order

    ease_to(follower, start_pose)
    print("Follower mirrors the leader. Press ENTER to start recording…")
    teleop_until_enter(leader, follower)

    frames = []
    print("● Recording… press ENTER to stop")
    teleop_until_enter(leader, follower, frames, joints)

    np.savez(path, joints=np.array(joints), frames=np.array(frames, np.float32), fps=FPS)
    print(f"✓ Saved {len(frames)} frames ({len(frames) / FPS:.1f}s) -> {path}")


def replay(follower, name: str) -> None:
    path = _rec_path(name)
    if not path.exists():
        print(f"✗ No such recording: {path}")
        return
    data = np.load(path, allow_pickle=True)
    joints = list(data["joints"])
    frames = data["frames"]
    fps = int(data["fps"])
    if len(frames) == 0:
        print(f"✗ {path} is empty")
        return

    ease_to(follower, {j: v for j, v in zip(joints, frames[0])})
    print(f"▶ Replaying {len(frames)} frames at {fps} FPS…")
    for row in frames:
        tick = time.perf_counter()
        follower.send_action({j: float(v) for j, v in zip(joints, row)})
        _pace(tick, fps)
    print("✓ Done.")


def main() -> None:
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    leader.connect()
    follower.connect()
    print(__doc__)
    try:
        while True:
            print("> ", end="", flush=True)
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                print()
                break
            if not line:  # EOF (Ctrl-D)
                break
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)

            if cmd == "end":
                break
            if cmd not in ("record", "replay") or not arg:
                print("Usage: record <name> | replay <name> | end")
                continue
            try:
                if cmd == "record":
                    record(leader, follower, arg)
                else:
                    replay(follower, arg)
            except KeyboardInterrupt:
                print(f"\n✗ {cmd} aborted")
    finally:
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
