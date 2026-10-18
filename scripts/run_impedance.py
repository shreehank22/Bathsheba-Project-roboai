import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from perception.detector import segment_red, segment_green, camera_intrinsics, pixel_to_world, pixel_to_plane
from perception.camera_stream import push_frames, start_stream, stop_stream
from planning.trajectory import JointTrajectory
from planning.ik import dls_ik
from planning.fk import fk
from envs.pick_scene import load_pick_scene