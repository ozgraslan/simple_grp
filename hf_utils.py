from huggingface_hub import HfApi, upload_folder


def create_model_repo(repo_id):
    api = HfApi()
    api.create_repo(
        repo_id=repo_id, 
        repo_type="model", 
        private=True, 
        exist_ok=True
    )


def push_all_checkpoints(local_dir, repo_id):
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message="Add/update checkpoints",
    )