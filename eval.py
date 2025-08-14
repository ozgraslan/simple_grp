import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from utils import quat2axisangle

def evaluate_model(task_suite_name="libero_10", num_trials=10, img_shape=(256, 256, 3), num_env_steps=500, num_ol_actions=8):

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()  
    for task_id in range(task_suite.n_tasks):

        task = task_suite.get_task(task_id)
        # task_name = task.name
        task_description = task.language
        # if not ("moka" in task_description.lower()):
        #     continue

        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
            f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": img_shape[0],
            "camera_widths": img_shape[1]
        }
        print(env_args)
        # Get default LIBERO initial states
        env_init_states = task_suite.get_task_init_states(task_id)
        print(f"Task {task_id} has {len(env_init_states)} initial states")
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        task_text_embeds, task_text_masks = encode_txt(task_description, t5_tokenizer, t5_model)
        obj_text_list = task2obj[task_description]
        obj_text_embeds, obj_text_masks = ctrlo.embed_text(obj_text_list)
        # print(obj_text_embeds.shape, obj_text_masks.shape)
        obj_text_embeds, obj_text_masks = obj_text_embeds.to(dtype).to(device), obj_text_masks.to(dtype).to(device)

        task_text_masks = torch.cat([task_text_masks, torch.ones((task_text_masks.shape[0], num_slot_tokens + num_patch_tokens + num_state_tokens + num_action_tokens), dtype=task_text_masks.dtype)], dim=1)    
        task_attention_mask = get_att_mask(block_mask_arr.unsqueeze(0).repeat(task_text_masks.shape[0], 1), task_text_masks)
        task_attention_mask = torch.logical_not(task_attention_mask)
        task_attention_mask = task_attention_mask.unsqueeze(1).repeat(1, transformer_heads, 1, 1)
        # print(num_tokens, task_attention_mask.shape)

        task_text_embeds, task_attention_mask = task_text_embeds.to(dtype).to(device), task_attention_mask.to(dtype).to(device)
        num_success = 0
        batch_videos_np = np.zeros((num_trials, 500, img_shape[2], img_shape[0], img_shape[1]), dtype=np.uint8)
        for t in range(num_trials):
            task_init_state = env_init_states[t % len(env_init_states)]
            frames, cum_reward = run_env(env, task_init_state, num_env_steps, num_ol_actions,
                                         task_text_embeds, task_attention_mask, obj_text_embeds, obj_text_masks)
            num_success += cum_reward
            frames_np = np.transpose(frames, axes=(0,3,1,2))
            batch_videos_np[t, :frames_np.shape[0]] = frames_np
        num_acc = num_success / num_trials
        if use_wandb:
            wandb.log({
                      "video": wandb.Video(batch_videos_np, 
                                caption=task_description, fps=10, format="mp4"),
                        "task success": num_acc, "task id": task_id})
        

        env.close() 

@torch.no_grad()
def run_env(env, task_init_state, num_env_steps, num_ol_actions,
             task_text_embeds, task_attention_mask, obj_text_embeds, obj_text_masks):
    model.eval()

    env.reset()
    ## taken from https://github.com/Physical-Intelligence/openpi/blob/36dc3c037eb8a3921be9ecb94369d60cbf56f58f/examples/libero/main.py
    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
    # and we need to wait for them to fall
    obs = env.set_init_state(task_init_state)
    for _ in range(10):
        obs, reward, done, info = env.step([0.0] * 6 + [-1.0]) ## dummy action

    done, step, frame_list, cum_reward = False, 0, [], 0
    action_queue = []
    while not done and step < num_env_steps:
        image = obs["agentview_image"][::-1, ::-1].copy()
        # print(image.shape)
        frame_list.append(image)
        if not action_queue:
            state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]], axis=0)
            if img_only_ctrlo:
                patch_embed, slot_embed = encode_img(image)
            else:
                patch_embed, slot_embed = encode_img_text(image, obj_text_embeds, obj_text_masks)
            batch = {"patch" : patch_embed.to(dtype).to(device),
                    "slot": slot_embed.to(dtype).to(device),
                    "state" : torch.tensor(state).unsqueeze(0).to(device, dtype=dtype),
                    "text_tokens": task_text_embeds.clone(),
                    "att_mask": task_attention_mask.clone().reshape(-1, num_tokens, num_tokens)
                    }
            pred_action =  model.get_action(batch)
            action_queue.extend(list(pred_action[:num_ol_actions]))
        obs, reward, done, info = env.step(action_queue.pop(0)) 
        cum_reward += reward
        step += 1

    return np.stack(frame_list, axis=0), cum_reward



if __name__ == "__main__":
    num_trials = 10
    num_env_steps = 500
    num_ol_actions = 8