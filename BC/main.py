import numpy as np
import gym
import torch
import torch.nn as nn
from IPython import display
from matplotlib import pyplot as plt
from utils import prepare_data

plt.style.use("ggplot")

# init environment
env_name = "Pendulum-v0"
env = gym.make(env_name)
action_space_size = env.action_space.shape[0]
state_space_size = env.observation_space.shape[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Load Expert data (states and actions for BC, States only for BCO)
expert_states = torch.tensor(np.load("states_expert_Pendulum.npy"), dtype=torch.float)
expert_actions = torch.tensor(np.load("actions_expert_Pendulum.npy"), dtype=torch.float)
print("expert_states", expert_states.shape)
print("expert_actions", expert_actions.shape)

# selecting number expert trajectories from expert data
number_expert_trajectories = 50
a = np.random.randint(expert_states.shape[0] - number_expert_trajectories)
print(a)
expert_state, expert_action = prepare_data(expert_states[a: a + number_expert_trajectories],
                                           expert_actions[a: a + number_expert_trajectories], window_size=2, compare=5)
print("expert_state", expert_state.shape)
print("expert_action", expert_action.shape)

# concatenate expert states and actions, divided into 70% training and 30% testing
new_data = np.concatenate((expert_state[:, : state_space_size], expert_action), axis=1)
np.random.shuffle(new_data)
new_data = torch.tensor(new_data, dtype=torch.float)
n_samples = int(new_data.shape[0] * 0.7)
training_set = new_data[:n_samples]
testing_set = new_data[n_samples:]
print("training_set size:", training_set.shape)
print("testing_set size:", testing_set.shape)

# Network arch for Behavioral Cloning
bc_walker = nn.Sequential(
    nn.Linear(state_space_size, 40),
    nn.ReLU(),
    nn.Linear(40, 80),
    nn.ReLU(),
    nn.Linear(80, 120),
    nn.ReLU(),
    nn.Linear(120, 100),
    nn.ReLU(),
    nn.Linear(100, 40),
    nn.ReLU(),
    nn.Linear(40, 20),
    nn.ReLU(),
    nn.Linear(20, action_space_size),
)

# loss function
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(bc_walker.parameters(), lr=learning_rate)

loss_list = []
test_loss = []
batch_size = 256
n_epoch = 100

for epoch in range(n_epoch):
    total_loss = 0
    b = 0
    for batch in range(0, training_set.shape[0], batch_size):
        data = training_set[batch: batch + batch_size, :state_space_size]
        y = training_set[batch: batch + batch_size, state_space_size:]
        y_pred = bc_walker(data)
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b += 1
    if epoch % 10 == 0:
        print("EPOCH: %i, MSE LOSS: %.6f" % (epoch + 1, total_loss / b))
    loss_list.append(total_loss / b)
    x = testing_set[:, :state_space_size]
    y = testing_set[:, state_space_size:]
    y_pred = bc_walker(x)
    test_loss.append(criterion(y_pred, y).item())

# # plot test loss
# # torch.save(bc_walker, "bc_walker_n=2") # uncomment to save the model
# plt.plot(test_loss, label="Testing Loss")
# plt.xlabel("iterations")
# plt.ylabel("loss")
# plt.legend()
# plt.show()


# parameters
n = 2  # window size
n_iterations = 5  # max number of interacting with environment
n_ep = 1000  # number of epochs
max_steps = 1000  # max timestep per epoch
gamma = 1.0  # discount factor
seeds = [0, 1, 2, 3, 4]  # random seeds for testing

seed_reward_mean = []
seed_reward = []
for itr in range(n_iterations):
    ################################## interact with env ##################################
    G = []
    G_mean = []
    env.seed(int(seeds[itr]))
    torch.manual_seed(int(seeds[itr]))
    torch.cuda.manual_seed_all(int(seeds[itr]))

    for ep in range(n_ep):
        state = env.reset()
        rewards = []
        R = 0
        for t in range(max_steps):
            action = bc_walker(torch.tensor(state, dtype=torch.float))
            action = np.clip(action.detach().numpy(), -1, 1)
            next_state, r, done, _ = env.step(action)
            rewards.append(r)
            state = next_state
            if done:
                env.reset()
                break
        R = sum([rewards[i] * gamma ** i for i in range(len(rewards))])
        G.append(R)
        G_mean.append(np.mean(G))
        # if ep % 10 == 0:
        #     print("ep = {} , Mean Reward = {:.6f}".format(ep, R))
        display.clear_output(wait=True)

    seed_reward.append(G)
    seed_reward_mean.append(G_mean)
    print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward_mean[-1])))

env.close()

seed_reward_mean_bc = np.array(seed_reward_mean)
mean_bc = np.mean(seed_reward_mean_bc, axis=0)
std_bc = np.std(seed_reward_mean_bc, axis=0)

expert = np.load("reward_mean_pendulum_expert.npy")
mean_expert = np.mean(expert, axis=0)
std_expert = np.std(expert, axis=0)

x = np.arange(1000)

plt.plot(x, mean_expert, "-", label="Expert")
plt.fill_between(x, mean_expert + std_expert, mean_expert - std_expert, alpha=0.2)

plt.plot(x, mean_bc, "-", label="BC")
plt.fill_between(x, mean_bc + std_bc, mean_bc - std_bc, alpha=0.2)

plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
plt.title("Expert VS BC in Pendulum")
plt.legend()
plt.show()
