import imageio
from robomimic.envs import env_robosuite
from robomimic.envs.wrappers import FrameStackWrapper
from robomimic.scripts import run_trained_agent
import argparse
import robomimic.utils.file_utils as FileUtils


# Create environment
kwargs = {
    "robots" : "Panda",
    "single_object_mode": 0,
    # "object_type": "can",
}
env = env_robosuite.EnvRobosuite(
    env_name = "NutAssembly",
    render= True,
    **kwargs
)
env = FrameStackWrapper(env, num_frames=2)



ckpt_path = "flow_gat_trained_models/graph_structure_experiment/20250611122432/models/model_epoch_400_NutAssemblySquare_success_0.68.pth"
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device='cuda', verbose=True)
# env, _ = FileUtils.env_from_checkpoint(
#         ckpt_dict=ckpt_dict, 
#         render=True,
        
#         verbose=True,
#     )
# Run trained agent
num_episodes = 10

video_path = "/home/nicolas/Documents/development/python/robomimic/robomimic/graph_structure_experiment.mp4"
video_writer = imageio.get_writer(video_path, fps=20)


for i in range(num_episodes):
    print(f"Running episode {i+1}/{num_episodes}")
    run_trained_agent.rollout(
        policy=policy,
        env=env,
        horizon=200,
        render=True,
        camera_names=["agentview"],
        # video_writer=video_writer,
        video_skip=5
    )
