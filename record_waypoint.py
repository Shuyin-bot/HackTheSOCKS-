# record_waypoint.py — run once per waypoint you need
# (8 pickup poses, 1 "inspect/lift" pose if using wrist cam, 4 drop poses, 1 home)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import json, os

cfg = SO101FollowerConfig(port="/dev/ttyACM0", id="follower")
robot = SO101Follower(cfg)
robot.connect(calibrate=False)

name = input("Name this waypoint (e.g. pickup_1, drop_pile_2, inspect, home): ")
obs = robot.get_observation()
positions = {k: v for k, v in obs.items() if k.endswith(".pos")}

path = "waypoints.json"
data = json.load(open(path)) if os.path.exists(path) else {}
data[name] = positions
json.dump(data, open(path, "w"), indent=2)
print(f"Saved {name}: {positions}")
robot.disconnect()
