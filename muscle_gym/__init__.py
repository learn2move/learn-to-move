from gym.envs.registration import register


register(
    id='MuscleHumanLoco2d-v2',
    entry_point='muscle_gym.envs.mujoco.human_loco2d:HumanLoco2dEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)



# -----------------------------------------------------------
