def main_engine(steps, state=None, epsilon=1.0, max_steps=2000, x=0.05):
    """
    Calculates reward for main engine performance during landing.
    Takes into account time efficiency, vertical velocity control, and altitude management.
    Returns a reward scaled by exploration rate (epsilon).

    Calcula recompensa por el rendimiento del motor principal durante el aterrizaje.
    Considera eficiencia temporal, control de velocidad vertical y gestión de altitud.
    Devuelve una recompensa escalada por la tasa de exploración (epsilon).
    """
    # Time-based reward (encourage efficiency) / Recompensa basada en tiempo (fomenta eficiencia)
    time_reward = min(50, (steps / max_steps) * 80)

    # Strong penalty if exceeding max steps / Penalización fuerte si excede pasos máximos
    if steps > max_steps:
        return {'reward': -100000}

    # Initialize penalty/bonus variables / Inicializar variables de penalización/bonificación
    height_penalty = 0
    vy_penalty_or_bonus = 0
    ascent_penalty = 0
    wrong_direction = False

    if state is not None:
        y = state[1]  # Current altitude / Altitud actual
        vy = state[3]  # Vertical velocity / Velocidad vertical

        # Vertical velocity control scoring / Puntuación de control de velocidad vertical
        ideal_vy = -0.298  # Ideal descent rate / Tasa de descenso ideal
        deviation = abs(vy - ideal_vy)

        # Velocity control rewards / Recompensas por control de velocidad
        if deviation < 0.05:
            vy_penalty_or_bonus = 50
        elif deviation < 0.1:
            vy_penalty_or_bonus = 30
        elif deviation < 0.2:
            vy_penalty_or_bonus = 15
        else:
            vy_penalty_or_bonus = -deviation * 250

        # Additional penalty for high speed at low altitude / Penalización adicional por alta velocidad a baja altitud
        if y < 0.5 and vy < -0.5:
            vy_penalty_or_bonus -= (abs(vy) - 0.5) * 150

        # Check if going wrong direction (too high) / Verificar si va en dirección incorrecta (demasiado alto)
        if y > 1.8:
            wrong_direction = True
        else:
            height_penalty = (y ** 2) * 10

        # Reduce time reward if excessive speed / Reducir recompensa temporal si velocidad excesiva
        if abs(vy) > 1.2:
            time_reward *= 0.5

    # Calculate raw reward before scaling / Calcular recompensa bruta antes de escalar
    raw_reward = time_reward + vy_penalty_or_bonus - height_penalty - ascent_penalty

    # Scale with epsilon (more exploration → higher scaling) / Escalar con epsilon (más exploración → mayor escala)
    scale = 0.5 + epsilon * 0.5
    adjusted_reward = raw_reward * scale

    # Clamp final reward to [-100, 100] range / Limitar recompensa final a rango [-100, 100]
    total_reward = max(min(adjusted_reward, 100), -100) / 10

    if wrong_direction:
        return {'reward': -100, 'WrongDirection': True}

    return {
        'reward': round(total_reward),
        'WrongDirection': False
    }