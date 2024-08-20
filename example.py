from cwarehouse.warehouse_env import WarehouseMultiEnv
import numpy as np







env = WarehouseMultiEnv(max_steps=10,image_observation=True)
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]


n_episodes = 10

for e in range(n_episodes):
 
    env.reset()
    terminated = False
    episode_reward = 0

    while not terminated:
        obs = env.get_obs()
        state = env.get_state()
        # env.render()  # Uncomment for rendering

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)

        local_obs, _, rewards, dones, infos, available_actions = env.step(actions)
        terminated = dones[0]
        episode_reward += np.mean(rewards)
     

    print("Total reward in episode {} = {}".format(e, episode_reward))

env.close()