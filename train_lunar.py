# Import required libraries / Importar librerías necesarias
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
import numpy as np 
import sys

# Import teacher modules / Importar módulos de profesores
from teachers.main_engine import main_engine
from teachers.aux_engines import aux_engines
from teachers.training_performance import training_performance
from teachers.landing import landing
from teachers.crash import crash
from teachers.safe_crash import safe_crash
from teachers.center_performance import center_performance

# Import master modules / Importar módulos maestros
from masters.adjust_epsilon import adjust_epsilon
from masters.adjust_reward import adjust_reward

# Import manager modules / Importar módulos de gestión
from gym_manager.save_model import save_model
from gym_manager.save_model import save_metadata
from gym_manager.load_model import initialize_state

# Function for test / Funcion de testeo
from test_lunar import evaluate_model

# Constants / Constantes
GAMMA = 0.99  # Discount factor / Factor de descuento
LR = 1e-3  # Learning rate / Tasa de aprendizaje
MIN_EPSILON = 0.001  # Minimum exploration rate / Tasa mínima de exploración
BATCH_SIZE = 64  # Training batch size / Tamaño del lote de entrenamiento
BUFFER_SIZE = 500_000  # Replay buffer size / Tamaño del buffer de experiencia
MAX_STEPS = 2000  # Max steps per episode / Máximo de pasos por episodio
VALUE_FOR_HUMAN = 0  # Epsilon threshold for human rendering / Umbral para renderizado humano
PATH = './models'  # Model save path / Ruta para guardar modelos
MODEL_NAME = 'LuLa_v1'  # Model name / Nombre del modelo
BEST_REWARD = float('-inf')  # Track best reward / Seguimiento de mejor recompensa
MIN_EPSILON_FOR_SAVE = 0.01 # Value for save IA Agent / Valor para guardar la IA
bonus = 0 # Bonus for training / Bonus para el entrenamiento
MAX_EPISODES=1500 # Number maxim of episodes
TEST_EPISODES=200 # Number of test
MAX_AI_SAVES=10 # Max number of ai saves
MIN_BONUS_FOR_SAVE=200 # Minim points for save one ia
MIN_AVERAGE_SCORE=200
MAX_CRASH_RATIO=0
CRASH_RATIO_HIGHT=0

# //WARN There are infinite trainings / Hay entrenamientos infinitos

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
    model_path=os.path.join(PATH, f"Null.pth"),
    metadata_path=os.path.join(PATH, "M.pth")
)

optimizer = optim.Adam(model.parameters(), lr=LR)  # Optimizer / Optimizador
loss_fn = nn.MSELoss()  # Loss function / Función de pérdida

# Global variables / Variables globales
steps = 0  # Total steps / Pasos totales
last_points = 0  # Last episode points / Puntos del último episodio
episode = 0  # Episode counter / Contador de episodios
replay_buffer = deque(maxlen=BUFFER_SIZE)  # Experience replay buffer / Buffer de experiencia
last_bonus = 0
metadata = []

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

# Main training loop with enhanced bilingual logging / Bucle principal de entrenamiento con registro bilingüe mejorado
while True:
    # Episode initialization / Inicialización del episodio
    print(f"\n{'='*50}")
    print(f"Starting Episode {episode} / Iniciando Episodio {episode}")
    print(f"Current Epsilon: {EPSILON:.3f} | Épsilon Actual: {EPSILON:.3f}")
    print(f"Last Bonus: {last_bonus:.2f} | Último Bono: {last_bonus:.2f}")
    print(f"Total Landings: {landings} | Aterrizajes Totales: {landings}")
    print(f"Safe Splashes: {soft_crashes} | Amerizajes Seguros: {soft_crashes}")
    print(f"Crashes: {crashes} | Accidentes: {crashes}")
    print(f"Total Steps: {steps} | Pasos Totales: {steps}")
    print('='*50)

    state, _ = env.reset()
    total_reward = 0
    steps_this_episode = 0
    last_engine_activations = 0  # Main engine activations / Activaciones motor principal
    left_engine_activations = 0  # Left engine activations / Activaciones motor izquierdo
    right_engine_activations = 0  # Right engine activations / Activaciones motor derecho

    # Episode execution / Ejecución del episodio
    for _ in range(MAX_STEPS):
        action = select_action(state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


        # Store experience with logging / Almacenar experiencia con registro
        replay_buffer.append((state, action, (bonus * 100), done, next_state))
        total_reward += reward
        steps_this_episode += 1
        steps += 1
        state = next_state

        train_step()

        if done:
            print(f"Episode terminated | Episodio terminado (Reason: {'Terminated' if terminated else 'Truncated'})")
            break

    # Reward calculations with detailed logging / Cálculos de recompensa con registro detallado
    print("\n" + "="*50)
    print("Calculating Rewards / Calculando Recompensas")
    print("-"*50)
    
    # Main engine reward / Recompensa del motor principal
    rME = main_engine(steps=steps_this_episode, state=state, epsilon=EPSILON, max_steps=MAX_STEPS)
    if rME.get('WrongDirection', False):
        print("❌ Wrong direction detected! Severe penalty applied. | ¡Dirección incorrecta detectada! Penalización severa aplicada.")
    print(f"Main Engine Reward: {rME['reward']:.2f} | Recompensa Motor Principal: {rME['reward']:.2f}")

    # Auxiliary engines reward / Recompensa de motores auxiliares
    aux_engine_activations = {'right': right_engine_activations, 'left': left_engine_activations}
    rAE = aux_engines(aux_engine_activations=aux_engine_activations, roll=state[4], state=state)
    print(f"Auxiliary Engines Reward: {rAE['reward']:.2f} | Recompensa Motores Auxiliares: {rAE['reward']:.2f}")

    # Landing reward / Recompensa de aterrizaje
    rL = landing(state, terminated, truncated, bonus)
    if rL['landing'] == 1:
        print("🚀 Successful landing detected! | ¡Aterrizaje exitoso detectado!")
        EPSILON -= 0.018
    if rL['landing'] == 1 and EPSILON < 0.3:
        EPSILON -= 0.04
    print(f"Landing Reward: {rL['reward']:.2f} | Recompensa Aterrizaje: {rL['reward']:.2f}")

    # Crash reward / Recompensa de choque
    rC = crash(state=state, terminated=terminated, truncated=truncated, total_reward=total_reward, 
              epsilon=EPSILON, landings=landings, crashes=crashes)
    if rC['crash'] == 1:
        print("💥 Crash detected! | ¡Choque detectado!")
        EPSILON += 0.004
    if rC['crash'] == 1 and EPSILON > 0.3:
        print("💥 Crash detected! | ¡Choque detectado!")
        EPSILON += 0.001
    print(f"Crash Reward: {rC['reward']:.2f} | Recompensa Choque: {rC['reward']:.2f}")

    # Safe crash reward / Recompensa de amerizaje seguro
    rSC = safe_crash(state, EPSILON)
    if rSC['soft_crash'] == 1:
        print("🌊 Safe splashdown detected! | ¡Amerizaje seguro detectado!")
    print(f"Safe Crash Reward: {rSC['reward']:.2f} | Recompensa Amerizaje Seguro: {rSC['reward']:.2f}")

    # Training performance / Rendimiento del entrenamiento
    tP = training_performance(bonus, last_bonus)
    print(f"Training Performance: {tP['reward']:.2f} | Rendimiento Entrenamiento: {tP['reward']:.2f}")

    # Center performance / Rendimiento de posición central
    tCP = center_performance(state)
    print(f"Center Performance: {tCP['reward']:.2f} | Rendimiento Posición Central: {tCP['reward']:.2f}")

    # Normalized rewards display / Visualización de recompensas normalizadas
    print("\nNormalized Rewards / Recompensas Normalizadas:")
    print("-"*50)
    norm_rME = rME['reward'] * 10
    norm_rAE = 100 * np.tanh(rAE['reward'] / 200.0)
    norm_tP = 100 * np.tanh(tP['reward'] / 5.0)
    norm_rL = rL['reward'] / 100
    norm_rC = rC['reward']
    norm_rSC = rSC['reward']
    norm_tCP = tCP['reward']
    
    print(f"  Main Engine (rME): {norm_rME:7.2f} | Motor Principal: {norm_rME:7.2f}")
    print(f"  Aux Engines (rAE): {norm_rAE:7.2f} | Motores Aux: {norm_rAE:7.2f}")
    print(f"  Training Perf (tP): {norm_tP:7.2f} | Rendimiento: {norm_tP:7.2f}")
    print(f"  Landing (rL): {norm_rL:7.2f} | Aterrizaje: {norm_rL:7.2f}")
    print(f"  Crash (rC): {norm_rC:7.2f} | Choque: {norm_rC:7.2f}")
    print(f"  Safe Crash (rSC): {norm_rSC:7.2f} | Amerizaje: {norm_rSC:7.2f}")
    print(f"  Center Perf (tCP): {norm_tCP:7.2f} | Posición: {norm_tCP:7.2f}")

    # Weight adjustment / Ajuste de pesos
    reward_sum = (norm_rME * 13) + (norm_rAE * 10) + (norm_tP * 10) + (norm_rL * 15) + (norm_rC * 15) + (norm_rSC * 13) + (norm_tCP * 20)
    bonus = adjust_reward(reward_sum, last_points, total_reward)
    print(f"\nTotal Weighted Reward: {reward_sum:7.2f} | Recompensa Ponderada Total: {reward_sum:7.2f}")
    print(f"New Bonus: {bonus:7.2f} | Nuevo Bono: {bonus:7.2f}")

    weights = {
        "rME": 13,
        "rAE": 10,
        "tP": 10,
        "rL": 15,
        "rC": 15,
        "rSC": 13,
        "tCP": 20
    }

    weighted_metrics = {
        "rME": norm_rME * weights["rME"],
        "rAE": norm_rAE * weights["rAE"],
        "tP":  norm_tP  * weights["tP"],
        "rL":  norm_rL  * weights["rL"],
        "rC":  norm_rC  * weights["rC"],
        "rSC": norm_rSC * weights["rSC"],
        "tCP": norm_tCP * weights["tCP"]
    }

    entry = {
        "epsilon": EPSILON,
        "bonus": round(bonus, 2),
        "landings": landings,
        "soft_crashes": soft_crashes,
        "crashes": crashes
    }

    for key, value in weighted_metrics.items():
        entry[f"weighted_{key}"] = round(value, 4)

    metadata.append(entry)

    # Update replay buffer rewards / Actualizar recompensas en el buffer
    print(f"Updating replay buffer rewards... | Actualizando recompensas en el buffer...")
    for i in range(1, steps_this_episode + 1):
        idx = -i
        if abs(idx) <= len(replay_buffer):
            s, a, _, d, ns = replay_buffer[idx]
            replay_buffer[idx] = (s, a, bonus, d, ns)

    # Epsilon adjustment / Ajuste de épsilon
    EPSILON, last_points = adjust_epsilon(EPSILON, reward_sum, last_points, min_epsilon=MIN_EPSILON)
    print(f"Adjusted Epsilon: {EPSILON:.3f} | Épsilon Ajustado: {EPSILON:.3f}")

    # Human rendering check / Verificación de renderizado humano
    if EPSILON <= VALUE_FOR_HUMAN and env.render_mode != "human":
        print("\nSwitching to human rendering mode | Cambiando a modo de renderizado humano")
        env.close()
        env = gym.make("LunarLander-v3", render_mode="human")

    # Episode summary / Resumen del episodio
    print("\n" + "="*50)
    print(f"Episode {episode} Summary | Resumen Episodio {episode}")
    print("-"*50)
    print(f"Total Reward: {total_reward:7.2f} | Recompensa Total: {total_reward:7.2f}")
    print(f"Steps: {steps_this_episode:4d} | Pasos: {steps_this_episode:4d}")
    print(f"Engine Activations: Main={last_engine_activations}, Left={left_engine_activations}, Right={right_engine_activations}")
    print(f"Activaciones Motores: Principal={last_engine_activations}, Izquierdo={left_engine_activations}, Derecho={right_engine_activations}")
    print(f"New Bonus: {bonus:7.2f} | Nuevo Bono: {bonus:7.2f}")
    print(f"New Epsilon: {EPSILON:.3f} | Nuevo Épsilon: {EPSILON:.3f}")
    print('='*50 + "\n")
    print("\n📦 Replay Buffer Summary | Resumen del Buffer de Experiencia")
    print("-" * 50)
    print(f"Buffer Size: {len(replay_buffer):,} / {BUFFER_SIZE:,}")

    # Update counters / Actualizar contadores
    landings += rL['landing']
    crashes += rC['crash']
    soft_crashes += rSC['soft_crash']
    if rSC['soft_crash'] == 1:
        crashes -= rC['crash']
    last_bonus = bonus
    episode += 1

    # Training reset condition / Condición de reinicio del entrenamiento
    if episode > MAX_EPISODES:
     print("\nMaximum episodes reached! Resetting training...")
     print("¡Máximo de episodios alcanzado! Reiniciando entrenamiento...")

     # Reset model
     model = DQN(obs_size, n_actions).to(device)

     # Reset optimizer
     optimizer = optim.Adam(model.parameters(), lr=LR)

     # Reset experience buffer
     replay_buffer.clear()

     # Reset counters
     steps = 0
     episode = 0
     last_points = 0
     last_bonus = 0
     BEST_REWARD = float('-inf')

     # Reload metadata (optional)
     EPSILON, landings, crashes, soft_crashes = initialize_state(
        model,
        model_path=os.path.join(PATH, f"Null.pth"),
        metadata_path=os.path.join(PATH, "M.pth")
     )

     print("🔄 Training has been reset. | Entrenamiento reiniciado.")

    # Model saving condition / Condición para guardar modelo
    if EPSILON < MIN_EPSILON_FOR_SAVE and (bonus > MIN_BONUS_FOR_SAVE):
      if MAX_AI_SAVES != 0:
        print("\n" + "="*50)
        print("Model Saving Triggered | Activado Guardado de Modelo")
        print("-"*50)
        print(f"Epsilon threshold crossed: {EPSILON:.3f} < {MIN_EPSILON_FOR_SAVE:.3f}")
        print(f"Umbral de épsilon cruzado: {EPSILON:.3f} < {MIN_EPSILON_FOR_SAVE:.3f}")
        print(f"Performance improvement detected: bonus = {bonus:.2f}")
        print(f"Mejora de rendimiento detectada: bono = {bonus:.2f}")
    
        BEST_REWARD = bonus
        bonus_str = f"{bonus:.2f}".replace('.', '_')
        filename = f"{MODEL_NAME}_bonus{bonus_str}_{MAX_AI_SAVES}.pth"
        metadata_filename = f"metadata_bonus{bonus_str}_{MAX_AI_SAVES}.pth"

        save_model(model, base_path=PATH, filename=filename)
        save_metadata(metadata, base_path=PATH, filename=metadata_filename)
        
        model_path_eval = os.path.join(PATH, filename)
        average_score, crash_ratio = evaluate_model(model_path_eval, TEST_EPISODES, MAX_STEPS)

        metadata.append({
            "average_score": average_score,
            "crash_ratio": crash_ratio,
        })
    
        print(f"Saving model to: {filename} | Guardando modelo en: {filename}")
        save_metadata(metadata, base_path=PATH, filename=metadata_filename)

        # Verificación de rendimiento
        if crash_ratio <= MAX_CRASH_RATIO and average_score >= MIN_AVERAGE_SCORE:
            print("\n✅ Model PASSED evaluation — retained.")
            print("✅ El modelo PASÓ la evaluación — se conserva.")
            MAX_AI_SAVES -=1
            metadata.clear()
        elif crash_ratio > CRASH_RATIO_HIGHT:            
            
            metadata.clear()
            
            # Reload metadata (optional)
            EPSILON, landings, crashes, soft_crashes = initialize_state(
                model,
                model_path=os.path.join(PATH, f"Null.pth"),
                metadata_path=os.path.join(PATH, "M.pth")
            )

            model_path = os.path.join(PATH, filename)
            metadata_path = os.path.join(PATH, metadata_filename)

            if os.path.exists(model_path):
                    os.remove(model_path)
            if os.path.exists(metadata_path):
                    os.remove(metadata_path)

        else:
            print("\n❌ Model FAILED evaluation — deleting...")
            print("❌ El modelo FALLÓ la evaluación — eliminando archivos...")

            model_path = os.path.join(PATH, filename)
            metadata_path = os.path.join(PATH, metadata_filename)
        
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            metadata.clear()

      else:
          print('Training completed')
          print('Entrenamiento completado')
          metadata.clear()
          break


