import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from model import Net


img_stack=4

transition = np.dtype([('s', np.float32, (img_stack, 96, 96)), 
                       ('a', np.float32, (3,)), ('a_logp', np.float32),
                       ('r', np.float32), ('s_', np.float32, (img_stack, 96, 96))])

GAMMA=0.99
EPOCH= 8 # beter than 10
MAX_SIZE = 2000 ## Increase (e.g. 10000) if your GPU has sufficient VRAM
BATCH=128 
EPS=0.1
LEARNING_RATE = 0.001 # bettr than 0.005 or 0.002 
torch.set_default_dtype(torch.float32)

class Agent():
    """ Agent for training """
    
    def __init__(self, device):
        self.training_step = 0
        self.net = Net(img_stack).float().to(device)
        self.buffer = np.empty(MAX_SIZE, dtype=transition)
        self.counter = 0
        self.device = device
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)  ## lr=1e-3

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp


    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == MAX_SIZE:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.float32).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.float32).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.float32).to(self.device).view(-1, 1)
        next_s = torch.tensor(self.buffer['s_'], dtype=torch.float32).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float32).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + GAMMA * self.net(next_s)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(EPOCH):
            for index in BatchSampler(SubsetRandomSampler(range(MAX_SIZE)), BATCH, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                
                # clipped function
                surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()