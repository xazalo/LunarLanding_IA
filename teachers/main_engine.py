def main_engine(steps, engine_activations, state=None, x=0.05, MAX_STEPS=1000):
    """
    Recompensa por:
    - Tiempo en el aire (para evitar caídas violentas).
    - Cercanía al suelo (para incentivar aterrizajes controlados).
    Penalización por:
    - Uso excesivo del motor principal.
    - Ascensos innecesarios o prolongados.
    
    Nota: La recompensa está limitada a un máximo de 100 y mínimo de -200.
    """
    t = steps / MAX_STEPS
    penalty_scale = max(0.3, 1.0 - t)

    # Recompensa base por duración en el aire
    time_reward = steps * x

    # Penalización por activaciones del motor principal
    engine_penalty = engine_activations * x * penalty_scale * 3

    # Bonificación por cercanía al suelo
    height_bonus = 0
    wrong_direction = False

    if state is not None:
        y = state[1]  # Altura

    if y > 1.7:  # Está subiendo innecesariamente alto
        wrong_direction = True
    else:
        height_bonus = (2.0 - y) * 0.2  # Bonificación si se mantiene bajo

    total_reward = time_reward - engine_penalty + height_bonus
    
    # Limitar el valor máximo y mínimo de recompensa
    total_reward = max(min(total_reward, 10), -200)

    if wrong_direction:
        return {
            'reward': -200,
            'WrongDirection': True
        }

    return {
        'reward': round(total_reward, 2),
        'WrongDirection': False
    }
