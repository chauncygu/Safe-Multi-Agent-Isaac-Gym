import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class ReplayBuffer:

    def __init__(self, config, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, share_actions_shape, device='cpu'):


        self.num_envs = config["n_rollout_threads"]
        self.num_transitions_per_env = config["replay_size"]
        self.device = device
        self.sampler = config["sampler"]

        # get dims of all agents' acitons
        share_act_dim = 0
        for id in range(len(share_actions_shape)):
            share_act_dim += share_actions_shape[id].shape[0]

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        # share observation = states
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        # next share observation = next_states
        self.next_states = torch.zeros(num_transitions_per_env, num_envs, *states_shape,
                                              device=self.device)

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        #share actions
        self.share_actions = torch.zeros(num_transitions_per_env, num_envs, share_act_dim, device=self.device)

        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, observations, states ,actions, share_actions, rewards, next_obs ,next_state ,dones):
        if self.step >= self.num_transitions_per_env:
            #TODO: 有点bug 清不掉0 后续改下
            self.step = (self.step + 1) % self.num_transitions_per_env
            # raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.share_actions[self.step].copy_(share_actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.next_observations[self.step].copy_(next_obs)
        self.next_states[self.step].copy_(next_state)
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1


    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch

