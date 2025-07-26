import numpy as np

def main_motor(steps, engine_activations, epsilon, landings):
    """
    Adjusts the reward focusing on:
    - Efficiency in main engine usage (few activations)
    - Allows for some flexibility before the first landing
    
    Ajusta la recompensa enfocándose en:
    - Eficiencia del uso del motor principal (pocos encendidos)
    - Permite cierta flexibilidad antes del primer aterrizaje
    """

    # Efficiency score ∈ [0, 1] - better if fewer activations
    efficiency_score = (steps - engine_activations) / max(steps, 1)

    if landings == 0:
        # Antes de aterrizar, bonificamos levemente el uso del motor para explorar
        efficiency_reward = 0.5 * engine_activations * epsilon
    else:
        # Tras aterrizar, penalizamos ineficiencia
        efficiency_reward = efficiency_score * 0.1 * epsilon

    return {
        'reward': efficiency_reward
    }
