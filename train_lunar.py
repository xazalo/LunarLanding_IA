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
from teachers.main_engine import main_engine
from teachers.aux_engines import aux_engines
from teachers.training_performance import training_performance
from teachers.landing import landing
from teachers.crash import crash
from teachers.step_eval import step_eval
from teachers.safe_crash import safe_crash

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
VALUE_FOR_HUMAN = 0  # Epsilon threshold for human rendering / Umbral para renderizado humano
PATH = './models'  # Model save path / Ruta para guardar modelos
MODEL_NAME = 'LuLa_v1'  # Model name / Nombre del modelo
BEST_REWARD = float('-inf')  # Track best reward / Seguimiento de mejor recompensa
SAVE_EPSILON_THRESHOLD = 0.95  # Epsilon threshold for saving / Umbral para guardar modelo
SAVE_EPSILON_STEP=0.1
last_saved_epsilon_threshold = 1.0 
bonus = 0

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
last_points = 0  # Last episode points / Puntos del último episodio
episode = 0  # Episode counter / Contador de episodios
replay_buffer = deque(maxlen=BUFFER_SIZE)  # Experience replay buffer / Buffer de experiencia
last_bonus = 0

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
        replay_buffer.append((state, action, (bonus * 100), done, next_state))
        total_reward += reward
        steps_this_episode += 1
        steps += 1
        state = next_state

        train_step()  # Perform training / Realizar entrenamiento

        if done:
            break

    # Calculate main motor reward / Calcular recompensa del motor principal
    rME = main_engine(
        steps=steps_this_episode,
        engine_activations=last_engine_activations,
        state=state,
        MAX_STEPS=MAX_STEPS
    )

    if rME.get('WrongDirection', False):
        print("❌ Dirección incorrecta detectada. Penalización severa aplicada.")

    # Calculate auxiliary motors reward / Calcular recompensa de motores auxiliares
    aux_engine_activations = {'right': right_engine_activations, 'left': left_engine_activations}
    rAE = aux_engines(
        aux_engine_activations=aux_engine_activations,
        roll=state[4],
    )

    rL = landing(
        state, terminated, truncated, bonus
    )

    landings += rL['landing']

    if rL['landing'] == 1:
        EPSILON -= 0.005
        
    rC = crash(
        state=state,
        terminated=terminated,
        truncated=truncated,
        total_reward=total_reward,
        epsilon=EPSILON,
        landings=landings,
        crashes=crashes 
    )

    crashes += rC['crash']
    if rC['crash'] == 1:
        EPSILON += 0.002

    rSE = step_eval(
        steps=steps_this_episode,
        MAX_STEPS=MAX_STEPS
    )

    rSC = safe_crash(
        state, EPSILON
    )

    soft_crashes += rSC['soft_crash']
    if rSC['soft_crash'] == 1:
        crashes -= rC['crash']
        EPSILON -= 0.003

    tP = training_performance(
        bonus, 
        last_bonus, 
    )

    last_bonus = bonus

    # Normalize rewards / Normalizar rewards
    norm_rME = rME['reward'] * 10   
    norm_rAE = 100 * np.tanh(rAE['reward'] / 200.0)         
    norm_tP = 100 * np.tanh(tP['reward'] / 5.0)
    norm_rL = rL['reward'] / 100
    norm_rC = rC['reward']
    norm_rSE = rSE['reward'] * 10
    norm_rSC = rSC['reward']

    # Logs for adjust wheights just in dev / para actualizar pesos en desarollo
    print(
        f"Recompensas normalizadas:\n"
        f"  Motor principal (rME): {norm_rME:.2f}\n"
        f"  Motores auxiliares (rAE): {norm_rAE:.2f}\n"
        f"  Performance entrenamiento (tP): {norm_tP:.2f}\n"
        f"  Aterrizaje (rL): {norm_rL:.2f}\n"
        f"  Crash (rC): {norm_rC:.2f}\n"
        f"  Evaluación de pasos (rSE): {norm_rSE:.2f}\n"
        f"  Crash suave (rSC): {norm_rSC:.2f}"
    )

    # AJuste de pesos / wheight adjust
    redward_sum = (norm_rME * 10) + (norm_rAE * 12) + (norm_tP * 5) + (norm_rL * 35) + (norm_rC * 25) + (norm_rSE * 15) + (norm_rSC * 3)

    bonus = adjust_reward(redward_sum, last_points)

    # Update rewards in replay buffer / Actualizar recompensas en el buffer
    for i in range(1, steps_this_episode + 1):
        idx = -i
        if abs(idx) <= len(replay_buffer):
            s, a, _, d, ns = replay_buffer[idx]
            replay_buffer[idx] = (s, a, bonus, d, ns)

    EPSILON, last_points = adjust_epsilon(EPSILON, redward_sum, last_points)

    # Switch to human render mode if epsilon is low enough
    # Cambiar a modo de renderizado humano si epsilon es suficientemente bajo
    if EPSILON <= VALUE_FOR_HUMAN and env.render_mode != "human":
        env.close()
        env = gym.make("LunarLander-v3", render_mode="human")
        print("Human rendering activated / Renderizado humano activado")

    #print(f"Ep {episode} | Epsilon: {EPSILON:.3f} | Bonus: {bonus:.2f} | Landings: {landings} | Soft Crash: {soft_crashes}, Crashes: {crashes}, Steps: {steps}.")
    print(f"Ep {episode} | Épsilon: {EPSILON:.3f} | Bono: {bonus:.2f} | Aterrizajes: {landings} | Amerizajes: {soft_crashes}, Accidentes: {crashes}, Pasos: {steps}")

    # Update tracking variables / Actualizar variables de seguimiento
    redward_sum = 0
    episode += 1

# Comprobar si hemos cruzado un nuevo umbral de epsilon
    if EPSILON < last_saved_epsilon_threshold - SAVE_EPSILON_STEP:
        last_saved_epsilon_threshold -= SAVE_EPSILON_STEP
        print(f"📉 Epsilon crossed threshold: {last_saved_epsilon_threshold:.2f}")
        print(f"💾 Saving model due to epsilon decrease")
        save_model(model, base_path=PATH, filename=f"{MODEL_NAME}.pth")
        save_metadata(EPSILON, landings, crashes, soft_crashes, base_path=PATH, filename=f"metadata_eps{last_saved_epsilon_threshold:.2f}.pth")
