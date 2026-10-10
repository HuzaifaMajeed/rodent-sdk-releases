"""rat_model.py: a PyTorch model that learns to control the RODENT rat.

Used by the notebook train_your_rat.ipynb. You can also replace it with your own file: any model works
with RODENT as long as it keeps the three methods the notebook calls:

    forward(observations) -> a score for each move
    act(observation, best=False) -> the move to make, 0 to 4  (used in training, and by RODENT in runs)
    learn(episodes) -> a dict of numbers to print             (called after every batch of episodes)

What goes in and out
--------------------
In:  31 numbers between -1 and 1 that describe what the rat senses on one step: where it is and which way
     it faces, how far the walls are in 8 directions, how strong the light, sound and odour are (overall,
     on its left and right side, and which direction they come from), and how far through the episode it is.
Out: one of 5 moves: 0 stay, 1 forward, 2 turn left, 3 turn right, 4 run.

How it learns: PPO (Proximal Policy Optimisation)
-------------------------------------------------
The model has two parts that share nothing but their input:

  the actor   scores the 5 moves. The rat picks a move at random in proportion to those scores, so it
              keeps trying new things while it learns.
  the critic  estimates how much reward the rat can still expect from where it is now.

After each batch of episodes, every move is judged by its *advantage*: did things go better than the
critic expected after making it? Moves with a positive advantage become more likely, the others less
likely. PPO limits how far the actor can change in one update ("clipping"), which keeps training stable,
and the critic is trained to predict the reward that really came. A small entropy bonus stops the actor
from settling on one move too early.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017 (arXiv:1707.06347).
"""

from __future__ import annotations

import torch
from torch import nn

N_OBSERVATIONS = 31   # RODENT's observation (see env.observation_names in the notebook)
N_MOVES = 5           # stay, forward, turn left, turn right, run


def _network(inputs: int, outputs: int, hidden: int, last_layer_scale: float) -> nn.Sequential:
    """Two hidden layers with tanh. Orthogonal start weights are the usual choice for PPO."""
    layers = nn.Sequential(
        nn.Linear(inputs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, outputs),
    )
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
            nn.init.zeros_(layer.bias)
    # A small last layer: the actor starts close to "every move equally likely",
    # the critic close to "expect nothing".
    nn.init.orthogonal_(layers[-1].weight, gain=last_layer_scale)
    return layers


class RatModel(nn.Module):
    """A PPO actor-critic for the RODENT rat. All settings have defaults that work for the examples."""

    def __init__(self,
                 n_observations: int = N_OBSERVATIONS,
                 n_moves: int = N_MOVES,
                 hidden: int = 64,             # units in each hidden layer
                 learning_rate: float = 1e-3,
                 discount: float = 0.99,       # how much reward later counts compared to reward now
                 smoothing: float = 0.95,      # GAE lambda: trades bias against noise in the advantage
                 clip: float = 0.2,            # PPO: how far one update may move the actor
                 epochs: int = 4,              # passes over each batch of episodes
                 minibatch: int = 256,         # steps per gradient step
                 curiosity: float = 0.01,      # entropy bonus: keeps the rat trying different moves
                 critic_weight: float = 0.5):  # how much the critic's error counts in the loss
        super().__init__()
        self.actor = _network(n_observations, n_moves, hidden, last_layer_scale=0.01)
        self.critic = _network(n_observations, 1, hidden, last_layer_scale=1.0)
        self.settings = dict(hidden=hidden, learning_rate=learning_rate, discount=discount, smoothing=smoothing,
                             clip=clip, epochs=epochs, minibatch=minibatch, curiosity=curiosity,
                             critic_weight=critic_weight)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    # ---- using the model -----------------------------------------------------------------

    def forward(self, observations) -> torch.Tensor:
        """A score for each move (the actor's output before it is turned into a choice)."""
        return self.actor(torch.as_tensor(observations, dtype=torch.float32))

    def act(self, observation, best: bool = False) -> int:
        """The move to make, picked at random in proportion to the scores. This is how the model
        learned, and how RODENT drives the rat with it in a run (seeded, so runs repeat).
        best=True always takes the highest-scoring move instead; that can leave the rat repeating the
        same move, so the model is usually better used as it was trained."""
        with torch.no_grad():
            scores = self(observation)
        if best:
            return int(scores.argmax())
        return int(torch.distributions.Categorical(logits=scores).sample())

    # ---- learning ------------------------------------------------------------------------

    def learn(self, episodes: list[dict]) -> dict:
        """Improve from a batch of finished episodes. Each episode is a dict with:

            observations       what the rat sensed before each move
            actions            the move it made
            rewards            the reward after each move
            final_observation  what it sensed after the last move
            terminated         True if the episode ended because the task was done (not because time ran out)
        """
        s = self.settings
        observations, actions, advantages, returns = [], [], [], []
        with torch.no_grad():
            for episode in episodes:
                seen = torch.as_tensor(episode["observations"] + [episode["final_observation"]], dtype=torch.float32)
                values = self.critic(seen).squeeze(-1)
                if episode["terminated"]:
                    values[-1] = 0.0       # nothing more to come after a real ending
                rewards = torch.as_tensor(episode["rewards"], dtype=torch.float32)
                # Generalised advantage estimation, from the last step back to the first.
                advantage, running = torch.zeros_like(rewards), 0.0
                for t in reversed(range(len(rewards))):
                    surprise = rewards[t] + s["discount"] * values[t + 1] - values[t]
                    running = surprise + s["discount"] * s["smoothing"] * running
                    advantage[t] = running
                observations.append(seen[:-1])
                actions.append(torch.as_tensor(episode["actions"]))
                advantages.append(advantage)
                returns.append(advantage + values[:-1])
            observations = torch.cat(observations)
            actions = torch.cat(actions)
            advantages = torch.cat(advantages)
            returns = torch.cat(returns)
            old_log_probs = torch.distributions.Categorical(logits=self.actor(observations)).log_prob(actions)

        totals = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(s["epochs"]):
            for batch in torch.randperm(len(actions)).split(s["minibatch"]):
                moves = torch.distributions.Categorical(logits=self.actor(observations[batch]))
                ratio = torch.exp(moves.log_prob(actions[batch]) - old_log_probs[batch])
                advantage = advantages[batch]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                actor_loss = -torch.min(ratio * advantage,
                                        ratio.clamp(1 - s["clip"], 1 + s["clip"]) * advantage).mean()
                critic_loss = (self.critic(observations[batch]).squeeze(-1) - returns[batch]).pow(2).mean()
                entropy = moves.entropy().mean()
                loss = actor_loss + s["critic_weight"] * critic_loss - s["curiosity"] * entropy
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimiser.step()
                totals["actor_loss"] += actor_loss.item()
                totals["critic_loss"] += critic_loss.item()
                totals["entropy"] += entropy.item()
                updates += 1
        return {name: value / updates for name, value in totals.items()}

    def describe(self) -> str:
        weights = sum(p.numel() for p in self.parameters())
        return (f"PPO actor-critic, two hidden layers of {self.settings['hidden']}, {weights:,} weights. "
                f"Settings: {self.settings}")
