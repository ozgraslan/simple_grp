# import tyro
from lerobot2td import Config, create_td_dataset
from ctrlo_inference import CTRLOFeatureExtractor
from task2obj import task2obj_10


ctrlo_config = Config(
    model_name = "ctrlo",
    selected_columns = {
        "observation.images.image": ("patch", "slot"),
        "task_index": "task_index",
        "observation.state": "state",
        "action": "action",
        "action_is_pad": "valid_mask"
    },
    selected_shapes = {
        "patch": (256, 384),
        "slot": (7, 256),
        "state": (8, ),
        "action": (7, ),
        "task_index": (),
    },
    dataset_repo_id = "lerobot/libero_10_image",
    save_path = "/network/scratch/o/ozgur.aslan/libero_td/libero_10_ctrlo",
    task2obj = task2obj_10,
    memory_limit_in_mb=120000,
)

print(ctrlo_config)
ctrlo = CTRLOFeatureExtractor(ctrlo_config).to(ctrlo_config.device)
ctrlo.eval()
create_td_dataset(ctrlo, ctrlo_config)