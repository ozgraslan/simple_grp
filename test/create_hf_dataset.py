from dataclasses import dataclass
from typing import Optional

import h5py
import tyro
from tqdm import tqdm
import numpy as np
        
import datasets
from PIL import Image
        
from mani_skill.utils.structs import Pose
from mani_skill.utils.io_utils import load_json
        
@dataclass
class Args:
    hf_dataset_name: str = ""
    split_name: str = ""
    demo_path: str = ""
    """the path of demo dataset (pkl or h5)"""
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None



def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


if __name__ == "__main__":
    args = tyro.cli(Args)

    dataset_file = args.demo_path

    data = h5py.File(dataset_file, "r")
    start = args.start_idx
    end = args.end_idx

    json_path = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = env_info["env_kwargs"]

    dataset_tmp = {"img": [], "seg": [], "action": [], "rel_tcp_pose" : [], "tcp_pose" : [], 
                    "goal_img": [], "reward": [], "terminated": [], "truncated": []}

    print("Episode start end:", start, end)
    for eps_id in tqdm(range(start, end)):
        eps = episodes[eps_id]
        trajectory = data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        for i in range(trajectory["actions"].shape[0]):
            current_pose = trajectory["obs"]["extra"]["tcp_pose"][i]
            next_pose = trajectory["obs"]["extra"]["tcp_pose"][i+1]
            print("current_pose", current_pose[:3])
            current_pose = Pose.create_from_pq(p=current_pose[:3], q=current_pose[3:])
            next_pose = Pose.create_from_pq(p=next_pose[:3], q=next_pose[3:])
            rel_tcp_pose = next_pose * current_pose.inv()
            
            img = 255 * ((trajectory["obs"]["sensor_data"]["3rd_view_camera"]["rgb"][i] + 1) / 2)
            seg = trajectory["obs"]["sensor_data"]["3rd_view_camera"]["segmentation"][i][:,:,0]
            action = trajectory["actions"][i]

            goal_img = 255 * ((trajectory["obs"]["sensor_data"]["3rd_view_camera"]["rgb"][-1] + 1) / 2)

            dataset_tmp["img"].append(Image.fromarray(img.astype('uint8')))
            dataset_tmp["seg"].append(seg)
            dataset_tmp["action"].append(action)
            dataset_tmp["rel_tcp_pose"].append(rel_tcp_pose.raw_pose.numpy()[0])
            dataset_tmp["tcp_pose"].append(current_pose.raw_pose.numpy()[0])
            dataset_tmp["goal_img"].append(Image.fromarray(goal_img.astype('uint8')))
            dataset_tmp["reward"].append(trajectory["rewards"][i])
            dataset_tmp["terminated"].append(trajectory["terminated"][i])
            dataset_tmp["truncated"].append(trajectory["truncated"][i])
        del trajectory

    dataset = {}
    dataset["img"] = dataset_tmp["img"]
    dataset["seg"] = np.asarray(dataset_tmp["seg"], dtype=np.int64)
    ## action is target joint pos with gripper openness 7 joints + 1 gripper
    dataset["action"] = np.asarray(dataset_tmp["action"], dtype=np.float32)
    ## rel_tcp_pose is change in end effector pose, pose is (pos xyz, quat wxyz) 
    dataset["rel_tcp_pose"] = np.asarray(dataset_tmp["rel_tcp_pose"], dtype=np.float32)
    ## tcp_pose is end effector pose, pose is (pos xyz, quat wxyz)
    dataset["tcp_pose"] = np.asarray(dataset_tmp["tcp_pose"], dtype=np.float32)

    dataset["goal_img"] = dataset_tmp["goal_img"]
    dataset["reward"] = np.asarray(dataset_tmp["reward"], dtype=np.float32)
    dataset["terminated"] = np.asarray(dataset_tmp["terminated"], dtype=bool)
    dataset["truncated"] = np.asarray(dataset_tmp["truncated"], dtype=bool)    

    print({key: it.shape if type(it) is np.ndarray else len(it) for key, it in dataset.items()})
    # ds = datasets.Dataset.from_dict(dataset)
    # ds.save_to_disk(dataset_path)
    # ds.push_to_hub(args.hf_dataset_name, split=args.split_name)

