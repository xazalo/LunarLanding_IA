# Import required libraries / Importar librerías necesarias
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
import numpy as np 

# Import teacher modules / Importar módulos de profesores
from teachers.main_motor import main_motor
from teachers.aux_motors import aux_motors
from teachers.training_performance import training_performance
from teachers.landing import landing

# Import master modules / Importar módulos maestros
from masters.adjust_epsilon import adjust_epsilon
from masters.adjust_reward import adjust_reward

# Import manager modules / Importar módulos de gestión
from gym_manager.save_model import save_model
from gym_manager.save_model import save_metadata
from gym_manager.load_model import initialize_state

# Constants / Constantes
GAMMA = 0.99  # Discount factor / Factor de descuento
LR = 1e-3  # Learning rate / Tasa de aprendizaje
MIN_EPSILON = 0.01  # Minimum exploration rate / Tasa mínima de exploración
BATCH_SIZE = 64  # Training batch size / Tamaño del lote de entrenamiento
BUFFER_SIZE = 100_000  # Replay buffer size / Tamaño del buffer de experiencia
MAX_STEPS = 1000  # Max steps per episode / Máximo de pasos por episodio
VALUE_FOR_HUMAN = 0.0001  # Epsilon threshold for human rendering / Umbral para renderizado humano
PATH = './models'  # Model save path / Ruta para guardar modelos
MODEL_NAME = 'LuLa_v1'  # Model name / Nombre del modelo
BEST_REWARD = float('-inf')  # Track best reward / Seguimiento de mejor recompensa
SAVE_EPSILON_THRESHOLD = 0.95  # Epsilon threshold for saving / Umbral para guardar modelo

# Initialize environment first to get observation and action sizes
# Inicializar entorno primero para obtener tamaño de observación y acciones
env = gym.make("LunarLander-v3", render_mode=None)
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Create model instance BEFORE calling initialize_state
# Crear instancia del modelo ANTES de llamar a initialize_state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),  # Input layer / Capa de entrada
            nn.ReLU(),  # Activation function / Función de activación
            nn.Linear(128, 128),  # Hidden layer / Capa oculta
            nn.ReLU(),  # Activation function / Función de activación
            nn.Linear(128, n_actions)  # Output layer / Capa de salida
        )

    def forward(self, x):
        return self.net(x)  # Forward pass / Paso hacia adelante

model = DQN(obs_size, n_actions).to(device)

# Now initialize state (loading weights and metadata if they exist)
# Ahora inicializamos el estado (cargando pesos y metadatos si existen)
EPSILON, landings, crashes, soft_crashes = initialize_state(
    model,
    model_path=os.path.join(PATH, f"{MODEL_NAME}.pth"),
    metadata_path=os.path.join(PATH, "metadata.pth")
)

optimizer = optim.Adam(model.parameters(), lr=LR)  # Optimizer / Optimizador
loss_fn = nn.MSELoss()  # Loss function / Función de pérdida

# Global variables / Variables globales
steps = 0  # Total steps / Pasos totales
points = 0  # Current points / Puntos actuales
last_points = 0  # Last episode points / Puntos del último episodio
episode = 0  # Episode counter / Contador de episodios
replay_buffer = deque(maxlen=BUFFER_SIZE)  # Experience replay buffer / Buffer de experiencia

def select_action(state, epsilon):
    """Select action using epsilon-greedy policy / Seleccionar acción con política epsilon-greedy"""
    if random.random() < epsilon:  # Exploration / Exploración
        return env.action_space.sample()
    else:  # Exploitation / Explotación
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_vals = model(state_v)
            return int(torch.argmax(q_vals))

def train_step():
    """Perform one training step / Realizar un paso de entrenamiento"""
    if len(replay_buffer) < BATCH_SIZE:
        return  # Not enough samples / No hay suficientes muestras

    # Sample random batch from replay buffer
    # Muestra lote aleatorio del buffer de experiencia
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, dones, next_states = zip(*batch)

    # Convert to tensors / Convertir a tensores
    states_v = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones_v = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(device)

    # Calculate current Q values / Calcular valores Q actuales
    q_values = model(states_v).gather(1, actions_v)
    
    # Calculate target Q values / Calcular valores Q objetivo
    next_q_values = model(next_states_v).max(1)[0].unsqueeze(1).detach()
    expected_q = rewards_v + (GAMMA * next_q_values * (~dones_v))

    # Compute loss and update model / Calcular pérdida y actualizar modelo
    loss = loss_fn(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main training loop / Bucle principal de entrenamiento
while True:
    state, _ = env.reset()
    total_reward = 0
    steps_this_episode = 0
    last_engine_activations = 0  # Main engine activations / Activaciones motor principal
    left_engine_activations = 0  # Left engine activations / Activaciones motor izquierdo
    right_engine_activations = 0  # Right engine activations / Activaciones motor derecho

    for _ in range(MAX_STEPS):
        action = select_action(state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Track engine activations / Registrar activaciones de motores
        if action == 1:
            left_engine_activations += 1
        elif action == 3:
            right_engine_activations += 1
        elif action == 2:
            last_engine_activations += 1

        # Store experience in replay buffer / Almacenar experiencia en el buffer
        replay_buffer.append((state, action, reward, done, next_state))
        total_reward += reward
        steps_this_episode += 1
        steps += 1
        state = next_state

        train_step()  # Perform training / Realizar entrenamiento

        if done:
            break

    points = total_reward

    # Calculate main motor reward / Calcular recompensa del motor principal
    tA = main_motor(
        steps=steps_this_episode,
        engine_activations=last_engine_activations,
    )

    # Calculate auxiliary motors reward / Calcular recompensa de motores auxiliares
    aux_engine_activations = {'right': right_engine_activations, 'left': left_engine_activations}
    tB = aux_motors(
        aux_engine_activations=aux_engine_activations,
        roll=state[4],
    )

    tC = landing(
        state, terminated, truncated, total_reward, EPSILON
    )

    tP = training_performance(
        points, 
        last_points, 
        EPSILON 
    )

    # just for development
    # print(tA['reward'], tB['reward'], tP['reward'])

    # Normalize rewards / Normalizar rewards
    norm_tA = 100 * np.tanh(tA['reward'] / 50.0)          
    norm_tB = 100 * np.tanh(tB['reward'] / 200.0)         
    norm_tP = 100 * np.tanh(tP['reward'] / 5.0)
    norm_tC = tC['reward']

    # AJuste de pesos / wheight adjust
    points_sum = (norm_tA * 0.4 * 10) + (norm_tB * 0.3 * 12) + (norm_tP * 5) + (norm_tC * 35)

    # Logs for adjust wheights just in dev / para actualizar pesos en desarollo
    # print(norm_tA, norm_tB, norm_tP)

    bonus = adjust_reward(points_sum, last_points)

    # Update rewards in replay buffer / Actualizar recompensas en el buffer
    for i in range(1, steps_this_episode + 1):
        idx = -i
        if abs(idx) <= len(replay_buffer):
            s, a, _, d, ns = replay_buffer[idx]
            replay_buffer[idx] = (s, a, bonus, d, ns)

    # Adjust exploration rate / Ajustar tasa de exploración
    EPSILON = adjust_epsilon(EPSILON, points, last_points)

    # Switch to human render mode if epsilon is low enough
    # Cambiar a modo de renderizado humano si epsilon es suficientemente bajo
    if EPSILON <= VALUE_FOR_HUMAN and env.render_mode != "human":
        env.close()
        env = gym.make("LunarLander-v3", render_mode="human")
        print("Human rendering activated / Renderizado humano activado")

    # Update tracking variables / Actualizar variables de seguimiento
    last_points = points
    episode += 1
    points = 0

    print(f"Ep {episode} | Epsilon: {EPSILON:.3f} | Reward: {total_reward:.2f} | Bonus: {bonus:.2f} | Landings: {landings} | Soft Crash: {soft_crashes}")
    print(f"Ep {episode} | Épsilon: {EPSILON:.3f} | Recompensa: {total_reward:.2f} | Bono: {bonus:.2f} | Aterrizajes: {landings} | Amerizajes: {soft_crashes}")



    # Save model if it's the best so far and exploration is low
    # Guardar modelo si es el mejor hasta ahora y la exploración es baja
    if total_reward > BEST_REWARD and EPSILON < SAVE_EPSILON_THRESHOLD:
        BEST_REWARD = total_reward
        print(f"✅ Saving model (reward: {total_reward:.2f}, epsilon: {EPSILON:.3f})")
        print(f"✅ Guardando modelo (recompensa: {total_reward:.2f}, épsilon: {EPSILON:.3f})")
        save_model(model, base_path=PATH, filename=f"{MODEL_NAME}.pth")
        save_metadata(EPSILON, 0, 0, 0, base_path=PATH, filename="metadata.pth")