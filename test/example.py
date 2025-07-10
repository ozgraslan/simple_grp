from torch.utils.data import DataLoader
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

repo_id = "lerobot/libero_10_image"
ds_meta = LeRobotDatasetMetadata(repo_id)

print(f"Total number of episodes: {ds_meta.total_episodes}")
episodes = list(range(ds_meta.total_episodes))
test_episodes = episodes[-2:]
print(f"Test episode numbers: {test_episodes}")

delta_timestamps = {
    "action": [t / ds_meta.fps for t in range(1)],
}

dataset = LeRobotDataset(repo_id, episodes=test_episodes, delta_timestamps=delta_timestamps)

dataloader = DataLoader(
    dataset,
    num_workers=0,
    batch_size=2,
    shuffle=True,
)

for batch in dataloader:
    print(batch["action"].shape)
