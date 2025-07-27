# Constants / Constantes
MODEL_NAME = "PERFECT_Lula_v1"  # Model name to evaluate / Nombre del modelo a evaluar
max_steps_per_episode = 2000  # Maximum steps per episode / Máximo de pasos por episodio

# Import required libraries / Importar librerías necesarias
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# Neural Network definition / Definición de la Red Neuronal
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),  # Input layer / Capa de entrada
            nn.ReLU(),  # Activation function / Función de activación
            nn.Linear(128, 128),  # Hidden layer / Capa oculta
            nn.ReLU(),  # Activation function / Función de activación
            nn.Linear(128, n_actions)  # Output layer / Capa de salida
        )

    def forward(self, x):
        return self.net(x)  # Forward pass / Paso hacia adelante

# Environment setup / Configuración del entorno
print("Initializing LunarLander environment... / Inicializando entorno LunarLander...")
env = gym.make("LunarLander-v3", render_mode=None)  # Create environment / Crear entorno
obs_size = env.observation_space.shape[0]  # Observation space size / Tamaño del espacio de observación
n_actions = env.action_space.n  # Number of possible actions / Número de acciones posibles

# Model loading / Carga del modelo
model_path = f"models/{MODEL_NAME}.pth"
print(f"Loading trained model from '{model_path}'... / Cargando modelo entrenado desde '{model_path}'...")
model = DQN(obs_size, n_actions)  # Create model instance / Crear instancia del modelo
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load weights / Cargar pesos
model.eval()  # Set to evaluation mode / Poner en modo evaluación

# Evaluation setup / Configuración de evaluación
n_episodes = 100  # Number of evaluation episodes / Número de episodios de evaluación
total_rewards = []  # Store rewards for each episode / Almacenar recompensas por episodio
crashes = 0  # Crash counter / Contador de choques
successes = 0  # Successful landing counter / Contador de aterrizajes exitosos
safe_landings = 0  # Safe landing counter / Contador de aterrizajes seguros

print(f"\nStarting evaluation with {n_episodes} episodes... / Comenzando evaluación con {n_episodes} episodios...\n")

# Evaluation loop / Bucle de evaluación
for episode in range(n_episodes):
    state, _ = env.reset()  # Reset environment for new episode / Reiniciar entorno para nuevo episodio
    done = False  # Episode completion flag / Bandera de finalización de episodio
    episode_reward = 0  # Reward accumulator for episode / Acumulador de recompensa por episodio
    step_count = 0  # Step counter / Contador de pasos

    # Episode execution / Ejecución del episodio
    while not done and step_count < max_steps_per_episode:
        with torch.no_grad():  # Disable gradient calculation / Deshabilitar cálculo de gradientes
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor / Convertir a tensor
            q_values = model(state_tensor)  # Get Q-values / Obtener valores Q
            action = torch.argmax(q_values).item()  # Select best action / Seleccionar mejor acción

        # Execute action in environment / Ejecutar acción en el entorno
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward  # Accumulate reward / Acumular recompensa
        step_count += 1  # Increment step counter / Incrementar contador de pasos

    # Store and analyze results / Almacenar y analizar resultados
    total_rewards.append(episode_reward)

    # Classify episode outcome / Clasificar resultado del episodio
    if episode_reward >= 200:
        successes += 1
        safe_landings += 1
        outcome = "Perfect landing! / ¡Aterrizaje perfecto!"
    elif episode_reward >= 50:
        successes += 1
        outcome = "Good landing / Buen aterrizaje"
    elif episode_reward >= 0:
        outcome = "Rough landing / Aterrizaje brusco"
    else:
        crashes += 1
        outcome = "Crashed / Choque"

    # Print episode summary / Mostrar resumen del episodio
    print(f"[E{episode + 1:03d}] Reward: {episode_reward:7.2f} | Steps: {step_count:4d} | {outcome}")
    print(f"[E{episode + 1:03d}] Recompensa: {episode_reward:7.2f} | Pasos: {step_count:4d} | {outcome.split('/')[1].strip()}")

    # Periodic progress report / Informe periódico de progreso
    if (episode + 1) % 10 == 0:
        print("\n" + "="*60)
        print(f"INTERIM REPORT / INFORME PARCIAL (Episodes {episode + 1:03d}/{n_episodes})")
        print("-"*60)
        print(f"Current average score: {np.mean(total_rewards[-10:]):.2f} / Puntuación media actual: {np.mean(total_rewards[-10:]):.2f}")
        print(f"Success rate: {successes}/{episode + 1} ({successes/(episode + 1)*100:.1f}%)")
        print(f"Tasa de éxito: {successes}/{episode + 1} ({successes/(episode + 1)*100:.1f}%)")
        print("="*60 + "\n")

env.close()  # Close environment / Cerrar entorno

# Calculate final statistics / Calcular estadísticas finales
average_score = np.mean(total_rewards)
std_dev = np.std(total_rewards)
min_score = np.min(total_rewards)
max_score = np.max(total_rewards)
success_rate = (successes / n_episodes) * 100
crash_rate = (crashes / n_episodes) * 100
safe_landing_rate = (safe_landings / n_episodes) * 100

# Generate final report / Generar informe final
print("\n" + "="*70)
print(" " * 20 + "FINAL EVALUATION REPORT")
print(" " * 21 + "INFORME FINAL DE EVALUACIÓN")
print("="*70)

# Detailed metrics table / Tabla de métricas detalladas
print(f"\n{'Metric':<30} | {'Value':>15} | {'Valor':>15}")
print("-"*70)
print(f"{'Episodes evaluated':<30} | {n_episodes:15d} | {n_episodes:15d}")
print(f"{'Average score':<30} | {average_score:15.2f} | {average_score:15.2f}")
print(f"{'Score standard deviation':<30} | {std_dev:15.2f} | {std_dev:15.2f}")
print(f"{'Minimum score':<30} | {min_score:15.2f} | {min_score:15.2f}")
print(f"{'Maximum score':<30} | {max_score:15.2f} | {max_score:15.2f}")
print(f"{'Success rate (%)':<30} | {success_rate:15.1f} | {success_rate:15.1f}")
print(f"{'Perfect landings (%)':<30} | {safe_landing_rate:15.1f} | {safe_landing_rate:15.1f}")
print(f"{'Crash rate (%)':<30} | {crash_rate:15.1f} | {crash_rate:15.1f}")

# Performance assessment / Evaluación de rendimiento
print("\n" + "="*70)
print(" " * 20 + "PERFORMANCE ASSESSMENT")
print(" " * 20 + "EVALUACIÓN DE RENDIMIENTO")
print("="*70)

if average_score >= 200:
    print("\nOutstanding performance! The model has mastered lunar landing.")
    print("¡Rendimiento excepcional! El modelo domina el aterrizaje lunar.")
    print("- Consistently achieves perfect landings / Logra aterrizajes perfectos consistentemente")
    print("- Excellent fuel efficiency / Excelente eficiencia de combustible")
    print("- Precise trajectory control / Control preciso de trayectoria")
elif average_score >= 150:
    print("\nExcellent performance! The model lands successfully most of the time.")
    print("¡Excelente rendimiento! El modelo aterriza exitosamente la mayoría de las veces.")
    print("- High success rate / Alta tasa de éxito")
    print("- Occasionally rough landings / Aterrizajes bruscos ocasionales")
    print("- Good overall control / Buen control general")
elif average_score >= 100:
    print("\nGood performance. The model lands safely but could improve precision.")
    print("Buen rendimiento. El modelo aterriza con seguridad pero podría mejorar precisión.")
    print("- Moderate success rate / Tasa de éxito moderada")
    print("- Needs better velocity control / Necesita mejor control de velocidad")
    print("- Sometimes excessive fuel use / A veces uso excesivo de combustible")
elif average_score >= 0:
    print("\nBasic performance. The model needs significant improvement.")
    print("Rendimiento básico. El modelo necesita mejoras significativas.")
    print("- Many rough landings / Muchos aterrizajes bruscos")
    print("- Frequent stability issues / Problemas frecuentes de estabilidad")
    print("- Poor fuel management / Mala gestión de combustible")
else:
    print("\nPoor performance. The model fails to land properly.")
    print("Rendimiento pobre. El modelo no logra aterrizar adecuadamente.")
    print("- High crash rate / Alta tasa de choques")
    print("- Unstable approach / Aproximación inestable")
    print("- Needs complete retraining / Necesita reentrenamiento completo")

print("\n" + "="*70)
print("Evaluation complete. / Evaluación completada.")
print(f"Model: {MODEL_NAME}")
print(f"Final average score: {average_score:.2f} / Puntuación media final: {average_score:.2f}")
print("="*70)