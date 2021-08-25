import torch
import matplotlib.pyplot as plt

available_actions = [[0, 0, 0],  # no action
                     [-1, 0, 0],  # left
                     [-1, 0, 1],  # left+break
                     [1, 0, 0],  # right
                     [1, 0, 1],  # right+break
                     [0, 1, 0],  # acceleration
                     [0, 0, 1], ]  # break


def prepare_data(states, actions, window_size, compare):
    count = 0
    index = []
    ep, t, state_size = states.shape
    _, _, action_size = actions.shape

    output_states = torch.zeros((ep * (t - window_size + 1), state_size * window_size), dtype=torch.float)
    output_actions = torch.zeros((ep * (t - window_size + 1), action_size), dtype=torch.float)

    for i in range(ep):
        for j in range(t - window_size + 1):
            if (states[i, j] == -compare * torch.ones(state_size)).all() or (
                    states[i, j + 1] == -compare * torch.ones(state_size)).all():
                index.append([i, j])
            else:
                output_states[count] = states[i, j:j + window_size].view(-1)
                output_actions[count] = actions[i, j]
                count += 1
    output_states = output_states[:count]
    output_actions = output_actions[:count]

    return output_states, output_actions
