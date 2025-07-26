def landing(state, terminated, truncated, bonus):
    """
    Calcula la recompensa del aterrizaje si fue exitoso (sin usar epsilon).

    Args:
        state (list): Estado final del episodio.
        terminated (bool): Si terminó naturalmente.
        truncated (bool): Si terminó por tiempo.
        bonus (float): Recompensa total del episodio.

    Returns:
        dict: Recompensa del aterrizaje (0 si no aplica).
    """
    legs_contact = state[6] == 1 and state[7] == 1

    if terminated and not truncated and legs_contact and bonus > 0:
        # Multiplicar para amplificar valores pequeños
        scaled_reward = bonus * 1000
        
        return {
            'reward': scaled_reward,
            'landing': 1,
            'bonus': bonus
        }

    return {
        'reward': 0,
        'landing': 0,
        'bonus': 0
    }
