def landing(state, terminated, truncated, bonus):
    """
    Calculates landing reward if successful (without using epsilon).
    / Calcula la recompensa del aterrizaje si fue exitoso (sin usar epsilon).

    Args / Argumentos:
        state (list): Final episode state / Estado final del episodio
        terminated (bool): Whether it ended naturally / Si terminó naturalmente
        truncated (bool): Whether it ended due to timeout / Si terminó por tiempo
        bonus (float): Total episode reward / Recompensa total del episodio

    Returns / Retorna:
        dict: Dictionary containing:
              - 'reward': Landing reward (0 if not applicable)
              - 'landing': 1 if successful landing, 0 otherwise
              - 'bonus': The bonus value used
              / Diccionario conteniendo:
              - 'reward': Recompensa de aterrizaje (0 si no aplica)
              - 'landing': 1 si aterrizaje exitoso, 0 en otro caso
              - 'bonus': El valor de bonus utilizado
    """
    # Check if both legs are in contact with ground
    # Verificar si ambas patas están en contacto con el suelo
    legs_contact = state[6] == 1 and state[7] == 1

    # Successful landing conditions:
    # 1. Episode terminated naturally
    # 2. Not truncated by timeout
    # 3. Both legs in contact
    # 4. Positive bonus
    # Condiciones para aterrizaje exitoso:
    # 1. Episodio terminado naturalmente
    # 2. No interrumpido por tiempo
    # 3. Ambas patas en contacto
    # 4. Bono positivo
    if terminated and not truncated and legs_contact and bonus > 0:
        # Scale reward to amplify small values
        # Escalar recompensa para amplificar valores pequeños
        scaled_reward = bonus * 100
        
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