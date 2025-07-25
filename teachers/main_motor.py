import numpy as np

def main_motor(steps, engine_activations):
    """
    Adjusts the reward focusing on:
    - Efficiency in main engine usage (few activations)
    
    Doesn't directly handle landings.
    
    Ajusta la recompensa enfocándose en:
    - Eficiencia en el uso del motor principal (pocos encendidos)
    
    No gestiona landings directamente.
    """

    # 2. Engine efficiency: fewer activations = better
    # 2. Eficiencia del motor: pocos encendidos = mejor
    efficiency_score = (steps - engine_activations) / max(steps, 1)  # ∈ [0,1] - efficiency ratio
    efficiency_reward = efficiency_score * 1  # Adjust this value if it penalizes too little/much

    return {
        'reward': efficiency_reward
    }