import torch
from torch.nn import functional as F


class A2CLearner():
    """
    Purpose: Implements the A2C (Advantage Actor-Critic) learning algorithm. The class combines 
    the actor and critic networks to update their parameters through reinforcement learning. 
    It uses the advantage function for updating the actor and the value function for updating the critic.

    Arguments:
        actor (nn.Module): The actor network, responsible for selecting actions based on states.
        critic (nn.Module): The critic network, responsible for evaluating the value of states.
        gamma (float, optional): Discount factor for future rewards (default: 0.99).
        entropy_beta (float, optional): Weight of the entropy term in the actor's loss (default: 0.05).
        actor_lr (float, optional): Learning rate for the actor's optimizer (default: 4e-4).
        critic_lr (float, optional): Learning rate for the critic's optimizer (default: 4e-3).
        max_grad_norm (float, optional): Maximum allowable gradient norm to prevent exploding gradients (default: 0.5).

    Output:
        None: Initializes the A2C learner with the provided parameters.
    """

    def __init__(self, writer, utils, actor, critic, gamma=0.99, entropy_beta=0.05,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        self.utils = utils
        self.writer = writer

    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = self.utils.process_memory(
            memory, self.gamma, discount_rewards)
        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma*self.critic(next_states)*(1-dones)
        value = self.critic(states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        actor_loss = (-logs_probs*advantage.detach()).mean() - \
            entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()

        self.utils.clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.writer.add_histogram("gradients/actor",
                                  torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.writer.add_histogram("parameters/actor",
                                  torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.utils.clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.writer.add_histogram("gradients/critic",
                                  torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.writer.add_histogram("parameters/critic",
                                  torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.critic_optim.step()

        # reports
        self.writer.add_scalar("losses/log_probs", -
                               logs_probs.mean(), global_step=steps)
        self.writer.add_scalar("losses/entropy", entropy, global_step=steps)
        self.writer.add_scalar("losses/entropy_beta",
                               self.entropy_beta, global_step=steps)
        self.writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        self.writer.add_scalar("losses/advantage",
                               advantage.mean(), global_step=steps)
        self.writer.add_scalar("losses/critic", critic_loss, global_step=steps)

        return actor_loss.item(), critic_loss.item()
