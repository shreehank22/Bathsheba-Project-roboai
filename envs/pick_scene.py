import os
import mujoco

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MENAGERIE_PATH = os.environ.get('MUJOCO_MENAGERIE', os.path.expanduser('~/mujoco_menagerie'))
PANDA_SCENE_PATH = os.path.join(MENAGERIE_PATH, 'franka_emika_panda', 'scene.xml')
TASK_SCENE_PATH = os.path.join(PROJECT_ROOT, 'assets', 'scene_pick.xml')

def load_pick_scene():
    # load the panda scene from its own directory so its relative mesh paths resolve,
    # then attach the repo's task objects (table, cube, place patch, fixed camera)
    spec = mujoco.MjSpec.from_file(PANDA_SCENE_PATH)
    task = mujoco.MjSpec.from_file(TASK_SCENE_PATH)
    spec.attach(task, frame=spec.worldbody.add_frame(), prefix='')
    return spec.compile()
