import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os

# Neural Network definition
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

def evaluate_model(model_name, n_episodes, max_steps_per_episode):
    print(f"Initializing LunarLander environment... / Inicializando entorno LunarLander...")
    env = gym.make("LunarLander-v3", render_mode=None)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f"Loading trained model from '{model_name}'... / Cargando modelo entrenado desde '{model_name}'...")
    model = DQN(obs_size, n_actions)
    model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu"), weights_only=False))
    model.eval()

    total_rewards = []
    crashes = 0
    successes = 0
    safe_landings = 0

    print(f"\nStarting evaluation with {n_episodes} episodes... / Comenzando evaluación con {n_episodes} episodios...\n")

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            step_count += 1

        total_rewards.append(episode_reward)

        if episode_reward >= 200:
            successes += 1
            safe_landings += 1
            outcome = "Perfect landing! / ¡Aterrizaje perfecto!"
        elif episode_reward >= 50:
            successes += 1
            outcome = "Good landing / Buen aterrizaje"
        elif episode_reward >= 0:
            successes +=1
            outcome = "Rough landing / Aterrizaje Mediocre"
        else:
            crashes += 1
            outcome = "Crashed / Choque"

        print(f"[E{episode + 1:03d}] Reward: {episode_reward:7.2f} | Steps: {step_count:4d} | {outcome}")
        print(f"[E{episode + 1:03d}] Recompensa: {episode_reward:7.2f} | Pasos: {step_count:4d} | {outcome.split('/')[1].strip()}")

        if (episode + 1) % 10 == 0:
            print("\n" + "="*60)
            print(f"INTERIM REPORT / INFORME PARCIAL (Episodes {episode + 1:03d}/{n_episodes})")
            print("-"*60)
            print(f"Current average score: {np.mean(total_rewards[-10:]):.2f} / Puntuación media actual: {np.mean(total_rewards[-10:]):.2f}")
            print(f"Success rate: {successes}/{episode + 1} ({successes/(episode + 1)*100:.1f}%)")
            print(f"Tasa de éxito: {successes}/{episode + 1} ({successes/(episode + 1)*100:.1f}%)")
            print("="*60 + "\n")

    env.close()

    average_score = np.mean(total_rewards)
    std_dev = np.std(total_rewards)
    min_score = np.min(total_rewards)
    max_score = np.max(total_rewards)
    success_rate = (successes / n_episodes) * 100
    crash_rate = (crashes / n_episodes) * 100
    safe_landing_rate = (safe_landings / n_episodes) * 100

    print("\n" + "="*70)
    print(" " * 20 + "FINAL EVALUATION REPORT")
    print(" " * 21 + "INFORME FINAL DE EVALUACIÓN")
    print("="*70)

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
    print(f"Model: {model_name}")
    print(f"Final average score: {average_score:.2f} / Puntuación media final: {average_score:.2f}")
    print("="*70)

    return average_score, crash_rate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained LunarLander model.")
    parser.add_argument("--model", type=str, required=True, help="Model filename (without .pth)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")

    args = parser.parse_args()
    evaluate_model(args.model, args.episodes, args.max_steps)
