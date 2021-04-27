import torch
from torch import nn
from network import build_mlp, reparameterize, evaluate_lop_pi

class StateFunction(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return self.net(states)

class StateIndependentPolicy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)

#=============COLLECT DEMO==================
class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))

class TwinnedStateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()
        # print("Network shape")
        # print(state_shape[0])
        # print(state_shape[1])
        # print(action_shape[0])
        self.net1 = build_mlp(
            #input_dim=state_shape[0] + state_shape[1] + action_shape[0],
            input_dim=state_shape[0]+ action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            #input_dim=state_shape[0] + state_shape[1] + action_shape[0],
            input_dim=state_shape[0]+ action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
    def forward(self, states, actions):
        # print(states.shape)
        # print(actions.shape)
        # actions = actions[:,None,:, :]
        # print(actions.shape)
        # for layer in self.net1.layers:
        #   print(layer.output_shape)
        # for layer in self.net2.layers:
        #   print(layer.output_shape)
        # actions= actions.expand(-1,96,-1)
        # print(actions.shape)
        actions = torch.reshape(actions,[256,1])
        xs = torch.cat([states, actions], dim=-1)
        # print("xs", xs.shape)
        return self.net1(xs), self.net2(xs)

    def q1(self, states, actions):
        return self.net1(torch.cat([states, actions], dim=-1))
