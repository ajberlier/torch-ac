import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from model import ACModel

class OCModel(ACModel):
    # TODO: cite OC paper here

    def __init__(self, env_obs_space, action_space, num_options, use_memory=False, use_text=False):
        super(OCModel, self).__init__(env_obs_space, action_space,  use_memory=False, use_text=False)

        # super never records action space, so keep a copy here
        self.action_space = action_space

        # TODO: add arg for hidden layer width; hard coded as 64 for now as a carry over from original torch_ac repo

        # ACModel overrides
        # actor
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # options 
        self.options = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, num_options)
        )

    # override ACModel forward pass
    def forward(self, obs, memory):
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
            embed_text = self._get_embed_text(self.obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        x = self.options(embedding)
        option_dist = Categorical(logits=F.log_softmax(x, dim=1))

        return dist, value, option_dist, memory