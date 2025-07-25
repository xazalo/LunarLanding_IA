# Import required libraries / Importar librerías necesarias
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# Neural Network definition / Definición de la Red Neuronal
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        # Network architecture / Arquitectura de la red
        # Input: observation_size -> Hidden: 128 -> Hidden: 128 -> Output: n_actions
        # Entrada: observation_size -> Oculta: 128 -> Oculta: 128 -> Salida: n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),  # First fully connected layer / Primera capa totalmente conectada
            nn.ReLU(),                # Activation function / Función de activación
            nn.Linear(128, 128),      # Second fully connected layer / Segunda capa totalmente conectada
            nn.ReLU(),                # Activation function / Función de activación
            nn.Linear(128, n_actions) # Output layer / Capa de salida
        )

    def forward(self, x):
        # Forward pass / Paso hacia adelante
        return self.net(x)

# Environment setup / Configuración del entorno
print("Initializing LunarLander environment...")
print("Inicializando entorno LunarLander...")
env = gym.make("LunarLander-v3")
obs_size = env.observation_space.shape[0]  # Observation space size / Tamaño del espacio de observación
n_actions = env.action_space.n            # Number of possible actions / Número de acciones posibles

# Model loading / Carga del modelo
print("Loading trained model from 'models/LuLa_v1.pth'...")
print("Cargando modelo entrenado desde 'models/LuLa_v1.pth'...")
model = DQN(obs_size, n_actions)
model.load_state_dict(torch.load("models/LuLa_v1.pth", map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode / Poner el modelo en modo evaluación

# Evaluation setup / Configuración de evaluación
n_episodes = 100      # Number of test episodes / Número de episodios de prueba
total_rewards = []    # Store rewards for each episode / Almacenar recompensas por episodio
crashes = 0           # Count of crash landings / Contador de aterrizajes fallidos
successes = 0         # Count of successful landings / Contador de aterrizajes exitosos
safe_landings = 0     # Count of safe but not perfect landings / Contador de aterrizajes seguros pero no perfectos

print(f"\nStarting evaluation with {n_episodes} episodes...")
print(f"Iniciando evaluación con {n_episodes} episodios...\n")

for episode in range(n_episodes):
    # Reset environment for new episode / Reiniciar entorno para nuevo episodio
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Convert state to tensor and get Q-values / Convertir estado a tensor y obtener valores Q
        with torch.no_grad():  # Disable gradient calculation / Desactivar cálculo de gradientes
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()  # Select best action / Seleccionar mejor acción
        
        # Execute action in environment / Ejecutar acción en el entorno
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward  # Accumulate reward / Acumular recompensa

    # Store and classify episode results / Almacenar y clasificar resultados del episodio
    total_rewards.append(episode_reward)
    
    if episode_reward >= 200:  # Successful landing / Aterrizaje exitoso
        successes += 1
    elif episode_reward >= 100:  # Safe landing / Aterrizaje seguro
        safe_landings += 1
    elif episode_reward < 0:  # Crash landing / Aterrizaje fallido
        crashes += 1

    # Progress update every 10 episodes / Actualización de progreso cada 10 episodios
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1:03d}/{n_episodes} | Reward: {episode_reward:7.2f} | "
              f"Successes: {successes:2d} | Safe: {safe_landings:2d} | Crashes: {crashes:2d}")
        print(f"Episodio {episode + 1:03d}/{n_episodes} | Recompensa: {episode_reward:7.2f} | "
              f"Éxitos: {successes:2d} | Seguros: {safe_landings:2d} | Fallos: {crashes:2d}")

# Close environment / Cerrar entorno
env.close()

# Calculate statistics / Calcular estadísticas
average_score = np.mean(total_rewards)
std_dev = np.std(total_rewards)
min_score = np.min(total_rewards)
max_score = np.max(total_rewards)
success_rate = (successes / n_episodes) * 100
crash_rate = (crashes / n_episodes) * 100

# Print final evaluation report / Imprimir reporte final de evaluación
print("\n" + "="*50)
print(" " * 15 + "FINAL EVALUATION REPORT")
print(" " * 16 + "INFORME FINAL DE EVALUACIÓN")
print("="*50)

print(f"\n{'Metric':<25} | {'Value':>10} | {'Valor':>10}")
print("-"*50)
print(f"{'Average score':<25} | {average_score:10.2f} | {average_score:10.2f}")
print(f"{'Score std deviation':<25} | {std_dev:10.2f} | {std_dev:10.2f}")
print(f"{'Minimum score':<25} | {min_score:10.2f} | {min_score:10.2f}")
print(f"{'Maximum score':<25} | {max_score:10.2f} | {max_score:10.2f}")
print(f"{'Success rate (%)':<25} | {success_rate:10.1f} | {success_rate:10.1f}")
print(f"{'Crash rate (%)':<25} | {crash_rate:10.1f} | {crash_rate:10.1f}")

# Performance assessment / Evaluación de rendimiento
print("\n" + "="*50)
print("PERFORMANCE ASSESSMENT / EVALUACIÓN DE RENDIMIENTO")
print("="*50)

if average_score >= 200:
    print("\nOutstanding performance! The model has mastered lunar landing.")
    print("¡Rendimiento excepcional! El modelo domina el aterrizaje lunar.")
elif average_score >= 150:
    print("\nExcellent performance! The model lands successfully most of the time.")
    print("¡Excelente rendimiento! El modelo aterriza exitosamente la mayoría de las veces.")
elif average_score >= 100:
    print("\nGood performance. The model lands safely but could improve precision.")
    print("Buen rendimiento. El modelo aterriza con seguridad pero podría mejorar precisión.")
elif average_score >= 0:
    print("\nBasic performance. The model needs significant improvement.")
    print("Rendimiento básico. El modelo necesita mejoras significativas.")
else:
    print("\nPoor performance. The model fails to land properly.")
    print("Rendimiento pobre. El modelo no logra aterrizar adecuadamente.")