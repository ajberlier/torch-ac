import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions.categorical import Categorical

import torch_ac
from torch_ac.algos.ppo import PPOAlgo


class PPOCAlgo(PPOAlgo):
    """Proximal Policy Option-Critic algorithm
    ([Klissarov et al., 2017](https://drive.google.com/file/d/1Arr3LcOzB_M80Ku_mVgY2Q88g0LBTk_X/view))."""

    def __init__(self, envs, arch, num_options, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super(PPOCAlgo, self).__init__(envs, arch, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef, 
                                       value_loss_coef, max_grad_norm, recurrence, adam_eps, clip_eps, epochs, batch_size, 
                                       preprocess_obss, reshape_reward)
        
        self.arch = arch

        shape = (self.num_frames_per_proc, self.num_procs)
        self.options = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.option = random.randint(0, num_options-1)
        self.options[0] = self.option
        #TODO: fix tuple structure
        self.dones = torch.zeros(*shape, device=self.device)
        self.terms = torch.zeros(*shape, device=self.device, dtype=torch.bool)
        self.term = True


    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_option_losses = []
            log_option_entropies = []
            log_policy_losses = []
            log_value_losses = []
            log_term_losses =[]
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():

                # initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_option_loss = 0
                batch_option_entropy = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_term_loss = 0
                batch_loss = 0

                # initialize memory
                if self.arch.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):

                    # sub-batch of experience
                    sb = exps[inds + i]

                    if self.arch.recurrent:
                        dist, value, option_dist, term_dist, memory = self.arch(sb.obs, sb.option, memory * sb.mask)
                    else:
                        dist, value, option_dist, term_dist = self.arch(sb.obs, sb.option)
                    
                    # actor - policy loss
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean() - (self.entropy_coef * entropy) 
                    
                    # critic - value loss
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # policy over options - option loss
                    # Klissrov does not clip the options, why? 
                    option_entropy = dist.entropy().mean()
                    ratio = torch.exp(option_dist.log_prob(sb.option) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    option_loss = -torch.min(surr1, surr2).mean() - (self.entropy_coef * option_entropy)

                    # termination loss
                    term_reg = 0 # 0.1 used by sutton options framework original paper; i dont think the addition makes sense here....
                    term_loss = -torch.min(term_dist.probs[torch.arange(term_dist.probs.size(0)), sb.option] * (value - sb.value + term_reg) * (1 - sb.done))
                    
                    # compute loss
                    # TODO: add self.option_loss_coef as a hyperparameters?
                    self.policy_loss_coef = 1
                    self.term_loss_coef = 1
                    self.value_loss_coef = 1
                    self.option_loss_coef = 1 # 0.1 was used in klissrov
                    loss = self.policy_loss_coef * policy_loss + \
                            self.term_loss_coef * term_loss + \
                            self.value_loss_coef * value_loss + \
                            self.option_loss_coef * option_loss
                    # TODO: needs work ^^^

                    # update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_term_loss += term_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_option_loss += option_loss.item()
                    batch_option_entropy += option_entropy.item()
                    batch_loss += loss

                    # update memories for next epoch
                    if self.arch.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_term_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_option_loss /= self.recurrence
                batch_option_entropy /= self.recurrence
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
                log_term_losses.append(batch_term_loss)
                log_value_losses.append(batch_value_loss)
                log_option_losses.append(batch_option_loss)
                log_option_entropies.append(batch_option_entropy)
                log_grad_norms.append(grad_norm)

        # Log some values
        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "term_loss": numpy.mean(log_term_losses),
            "value_loss": numpy.mean(log_value_losses),
            "option_loss": numpy.mean(log_option_losses),
            "option_entropy": numpy.mean(log_option_entropies),
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
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.arch.recurrent:
                    dist, value, option_dist, term_dist, memory = self.arch(preprocessed_obs, self.options[i], self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value, option_dist, term_dist = self.arch(preprocessed_obs, self.options[i])

            # if terminated, choose new option
            if self.term:
                self.option = option_dist.sample()

            # select action
            action = dist.sample()

            # step environment
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # update option termination
            self.term = bool(Bernoulli(term_dist.probs[torch.arange(self.option.size(0)), self.option]).sample().item())

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
            self.options[i] = self.option

            self.terms[i] = self.term
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
            if self.arch.recurrent:
                dist, next_value, option_dist, term_dist, memory = self.arch(preprocessed_obs, self.options[i], self.memory * self.mask.unsqueeze(1))
            else:
                dist, next_value, option_dist, term_dist = self.arch(preprocessed_obs, self.options[i])

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
        exps.done = self.dones.transpose(0, 1).reshape(-1)
        exps.option = self.options.transpose(0, 1).reshape(-1)


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
