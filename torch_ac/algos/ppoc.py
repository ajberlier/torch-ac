import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import torch_ac
from torch_ac.algos.oc import OCModel
from torch_ac.algos.ppo import PPOAlgo
from torch_ac.algos.base import BaseAlgo


class PPOCAlgo(PPOAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, arch, num_options, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super(PPOCAlgo, self).__init__(envs, arch, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef, 
                                       value_loss_coef, max_grad_norm, recurrence, adam_eps, clip_eps, epochs, batch_size, 
                                       preprocess_obss, reshape_reward)
        
        shape = (self.num_frames_per_proc, self.num_procs)
        self.options = torch.zeros(*shape, device=self.device, dtype=torch.int)
        #TODO: fix tuple strucutre
        self.dones = torch.zeros(*shape, device=self.device)


    def select_option(self, option_probs):
        # sample option based on option probabilities
        option_dist = torch.distributions.Categorical(option_probs)
        option = option_dist.sample()


    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_option_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():

                # initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_option_loss = 0
                batch_loss = 0

                # initialize memory
                if self.arch.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):

                    # sub-batch of experience
                    sb = exps[inds + i]

                    # select options
                    # option_select = []
                    # for obs, option_dist in zip(sb.obs, sb.option_dist):
                        # option = self.select_option(obs, option_dist)
                    option_probs = sb.obs.option_dist
                    option_dist = torch.distributions.Categorical(option_probs)
                    option_select = option_dist.sample()
                    # TODO ?? option_select is unused

                    if self.arch.recurrent:
                        dist, value, option_dist, memory = self.arch(sb.obs, memory * sb.mask)
                    else:
                        dist, value, option_dist = self.arch(sb.obs)
                    
                    # actor - policy loss
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # critic - value loss
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # TODO: needs work vvv
                    # termination condition
                    # TODO sb does not have dones
                    option_termination_mask = torch.zeros_like(sb.dones)
                    for i, done in enumerate(sb.dones):
                        if done or options[i] == 1:  # terminate option on episode end or when a new option is selected
                            option_termination_mask[i] = 1

                    # options - value loss
                    option = F.softmax(sb.option_dist, dim=-1)
                    option_loss = -torch.log(option.gather(1, options.unsqueeze(1))).squeeze() * option_termination_mask
                    option_loss = option_loss.mean()
                    
                    # compute loss
                    entropy = dist.entropy().mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss + self.option_loss_coef * option_loss
                    # TODO: needs work ^^^

                    # update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_option_loss += option_loss.item()
                    batch_loss += loss

                    # update memories for next epoch
                    if self.arch.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_option_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.arch.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.arch.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_option_losses.append(batch_option_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "option_loss": numpy.mean(log_option_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs
    
    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # do one agent-environment interaction
            # TODO handle recurrent flag? Or remove it
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.arch.recurrent:
                    dist, value, option_dist, memory = self.arch(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    # TODO confirm return arg order
                    # dist, option_dist, value = self.arch(preprocessed_obs)
                    dist, value, option_dist = self.arch(preprocessed_obs)
            option = option_dist.sample()
            action = dist.sample()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))
            #TODO: need to account for deliberation cost

            # add option_dist to prior obs
            for n_proc in range(self.num_procs):
                self.obs[n_proc]['option_dist'] = option_dist.logits.detach().cpu().numpy()[n_proc, :]

            # update experiences values
            self.obss[i] = self.obs
            self.obs = obs
            if self.arch.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)
            self.options[i] = option
            self.dones[i] = torch.tensor(done, device=self.device, dtype=torch.bool)

            # update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        # Get the next_value for the last element in the self.obs list
        # This gets a single additional next_value
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            # handle 4 len tuple from OCModel, unlike the other 3 len tuples
            # from OCModel its (dist, value, option_dist, memory)
            if self.arch.recurrent:
                dist, next_value, option_dist, memory = self.arch(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                # TODO confirm arg order
                dist, next_value, option_dist = self.arch(preprocessed_obs)

            # if self.arch.recurrent:
            #     _, next_value, _ = self.arch(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            # else:
            #     _, next_value = self.arch(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            # TODO: verify advantage estimates are correct...
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            # self.option_advantages[i] = 

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = torch_ac.DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.arch.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
