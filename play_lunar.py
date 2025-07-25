import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# Red neuronal (debe ser igual a la usada durante el entrenamiento)
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Crear entorno con visualización
env = gym.make("LunarLander-v3", render_mode="human")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Cargar modelo
model = DQN(obs_size, n_actions)
model.load_state_dict(torch.load("models/LuLa_v1.pth"))  # Asegúrate de que este archivo exista
model.eval()  # Modo evaluación (desactiva dropout, etc.)

# Ejecutar un episodio
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        state_v = torch.FloatTensor(state).unsqueeze(0)
        q_vals = model(state_v)
        action = torch.argmax(q_vals).item()
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

env.close()

