import os
import mujoco
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MENAGERIE_PATH = os.environ.get('MUJOCO_MENAGERIE', os.path.expanduser('~/mujoco_menagerie'))
PANDA_SCENE_PATH = os.path.join(MENAGERIE_PATH, 'franka_emika_panda', 'scene.xml')
TASK_SCENE_PATH = os.path.join(PROJECT_ROOT, 'assets', 'scene_pick.xml')

# shared start pose for the run scripts. don't start from qpos0 (all zeros): the arm is fully
# extended there, which is singular for Cartesian control, and joint 4 sits outside its limit
Q_HOME = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float64)

def load_pick_scene():
    # load the panda scene from its own directory so its relative mesh paths resolve,
    # then attach the repo's task objects (table, cube, place patch, fixed camera)
    spec = mujoco.MjSpec.from_file(PANDA_SCENE_PATH)
    task = mujoco.MjSpec.from_file(TASK_SCENE_PATH)
    spec.attach(task, frame=spec.worldbody.add_frame(), prefix='')

    # grasp stability: set here because attach() drops <option> from the task XML.
    # noslip removes contact creep; the elliptic cone is needed for the cube's torsional friction
    # to work (with the default pyramidal cone, condim 4 makes the cube slip faster)
    spec.option.noslip_iterations = 4
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    return spec.compile()
