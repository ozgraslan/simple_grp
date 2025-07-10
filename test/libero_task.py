import os
from PIL import Image
import numpy as np

from scipy.ndimage import rotate
# from torchvision.transforms.functional import rotate, to_tensor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# repo_id = "lerobot/libero_10_image"

# dataset_metadata = LeRobotDatasetMetadata(repo_id)
# task = dataset_metadata.tasks[5]
# ep_task_list = [dct["tasks"][0] for dct in dataset_metadata.episodes.values()]

# ep_index = ep_task_list.index(task)
# print(task, ep_index, ep_task_list[ep_index])
# dataset = LeRobotDataset(repo_id, episodes=[ep_index])
# data = dataset[0]
# Image.fromarray((255*data['observation.images.image'].permute(1,2,0).numpy()).astype(np.uint8)).save('observation.images.image.png')
# Image.fromarray((255*data['observation.images.wrist_image'].permute(1,2,0).numpy()).astype(np.uint8)).save('observation.images.wrist_image.png')

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in range(1):
    obs, reward, done, info = env.step(dummy_action)
    Image.fromarray(obs["agentview_image"][::-1,::-1]).save("idc_agentview_image.png")
    Image.fromarray(obs["robot0_eye_in_hand_image"][::-1,::-1]).save("idc_robot0_eye_in_hand_image.png")

env.close()