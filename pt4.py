import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gym


class CNNPolicyNet(nn.Module):
    def __init__(self, numActions):
        super(CNNPolicyNet, self).__init__()

        # define 3 conv layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # Input: 4 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # one fully connected, inputs adjusted to conv output
        self.fc = nn.Linear(64 * 84 * 84, 128)

        # Output heads
        self.policy_head = nn.Linear(128, numActions)  # policy
        self.value_head = nn.Linear(128, 1)  # value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten x
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))

        # get policy and value
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    # Final prediction, softmax the forward
    def predict(self, x):
        with torch.no_grad():
            logits, _ = self.forward(x)
            return F.softmax(logits, dim=-1), None


# load data
observations = torch.load("data/pong_observations.pt", weights_only=True)
actions = torch.load("data/pong_actions.pt", weights_only=True)


# convert obs to shape (num_samples, 4, 84, 84)
if len(observations.shape) == 4:
    observations = observations.permute(0, 3, 1, 2)



class PongDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# Create dataset and data loader
dataset = PongDataset(observations, actions)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Step 4: Train the Model
def train_model(model, data_loader, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for obs, action in data_loader:
            obs = obs.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            policy_logits, _ = model(obs)
            loss = F.cross_entropy(policy_logits, action)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "model_beh_clone.cpt")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNPolicyNet(numActions=6).to(device)

# Train the model
#train_model(model, data_loader)


# Step 5: Render the Environment
def render_env(env, policy, max_steps=500):
    obs = env.reset()
    for i in range(max_steps):
        with torch.no_grad():
            # Convert obs to a tensor and move to the device
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # Change to shape [batch_size, channels, height, width]

            # Forward pass through the policy to get action probabilities
            action_probs, _ = policy.predict(obs_tensor)

        # get action from possible
        action = torch.multinomial(action_probs, 1).item()

        # Step the environment
        obs, reward, done, info = env.step([action])

        if done:
            break  # Game over
    env.close()


# Create the Pong environment
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=2, env_kwargs={'render_mode': "human"})
env = VecFrameStack(env, n_stack=4)

# Load the trained model and play the game
model.load_state_dict(torch.load("model_beh_clone.cpt", map_location=device))
render_env(env, model, 5000)
