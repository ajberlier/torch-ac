import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from model import ACModel

class OCModel(ACModel):
    """The Option-Critic Architecture
    ([Bacon et al., 2017](https://dl.acm.org/doi/10.5555/3298483.3298491))."""

    def __init__(self, env_obs_space, action_space, num_options, use_memory=False, use_text=False):
        super(OCModel, self).__init__(env_obs_space, action_space, use_memory, use_text)

        self.action_space = action_space
        self.num_options = num_options
        # TODO: add arg for hidden layer width; hard coded as 64 for now as a carry over from original torch_ac repo
        # TODO: make these input args
        self.dc = 0.1 # deliberation cost (eta)

        # override ACModel networks as needed

        # actor
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size + 1, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n)
        )

        # options 
        self.options = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_options)
        )

        # terminations 
        self.terms = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_options)
        )

    # override ACModel forward pass
    def forward(self, obs, option, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # options
        x = self.options(embedding)
        option_dist = Categorical(logits=F.log_softmax(x, dim=1))
        
        # actor
        option_embedding = torch.cat((embedding, option.unsqueeze(1)), dim=1)
        x = self.actor(option_embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        # critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        # check for termination condition
        x = self.terms(embedding)
        term_dist = Categorical(logits=F.log_softmax(x, dim=1))

        return dist, value, option_dist, term_dist, memory