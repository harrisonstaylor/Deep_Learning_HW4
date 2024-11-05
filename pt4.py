import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gym
for env in gym.envs.registry.keys():
    print(env)

# Step 1: Define the CNN architecture
class CNNPolicyNet(nn.Module):
    def __init__(self, numActions):
        super(CNNPolicyNet, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # Input: 4 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(64 * 84 * 84, 128)  # Adjust input dimensions based on output of conv layers

        # Output heads
        self.policy_head = nn.Linear(128, numActions)  # Policy output
        self.value_head = nn.Linear(128, 1)  # Value output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc(x))

        # Get policy and value outputs
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict(self, x):
        with torch.no_grad():
            logits, _ = self.forward(x)
            return F.softmax(logits, dim=-1), None


# Step 2: Load the expert data
observations = torch.load("data/pong_observations.pt", weights_only=True)  # Shape: (num_samples, 84, 84, 4)
actions = torch.load("data/pong_actions.pt", weights_only=True)  # Shape: (num_samples, )

# Check the shape of the observations
print(f"Original observations shape: {observations.shape}")

# Convert the observations to the right shape (num_samples, 4, 84, 84)
# Assuming observations are in shape (num_samples, 84, 84, 4)
if len(observations.shape) == 4:  # (num_samples, 84, 84, 4)
    observations = observations.permute(0, 3, 1, 2)  # Change to (num_samples, 4, 84, 84)
else:  # Handle the case where the shape is unexpected
    print("Unexpected shape for observations. Please check the data.")


# Step 3: Create a Dataset and DataLoader
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
    torch.save(model.state_dict(), "model_beh_clone.pth")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNPolicyNet(numActions=6).to(device)

# Train the model
train_model(model, data_loader)

PATH = './pong_model.pth'
torch.save(model.state_dict(), PATH)

# Step 5: Render the Environment
def render_env(env, policy, max_steps=500):
    obs = env.reset()
    for i in range(max_steps):
        with torch.no_grad():
            actionProbs, _ = policy.predict(obs)
        action = torch.multinomial(actionProbs, 1).item()  # randomly pick an action according to policy
        obs, reward, done, info = env.step([action])
        if done:
            break  # game over
    env.close()


# Create the Pong environment
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=2, env_kwargs={'render_mode': "human"})
env = VecFrameStack(env, n_stack=4)

# Load the trained model and play the game
model.load_state_dict(torch.load("model_beh_clone.pth", map_location=device))
render_env(env, model, 5000)
