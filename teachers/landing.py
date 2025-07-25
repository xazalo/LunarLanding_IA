def landing(state, terminated, truncated, total_reward, epsilon):
    """
    Calculates the landing reward if successful, adjusted by epsilon.
    Calcula la recompensa del aterrizaje si fue exitoso, ajustada por épsilon.

    Args/Argumentos:
        state (list): Final episode state / Estado final del episodio
        terminated (bool): Whether it ended naturally / Si terminó naturalmente
        truncated (bool): Whether it ended by timeout / Si terminó por tiempo
        total_reward (float): Total episode reward / Recompensa total del episodio
        epsilon (float): Current exploration level / Nivel de exploración actual

    Returns/Retorna:
        int: Normalized and adjusted landing reward (or 0 if not applicable)
              Recompensa del aterrizaje normalizada y ajustada (o 0 si no aplica)
    """
    # Check if both legs are in contact with ground
    # Verificar si ambas patas están en contacto con el suelo
    legs_contact = state[6] == 1 and state[7] == 1
    
    # Conditions for successful landing:
    # 1. Episode terminated naturally (not timeout)
    # 2. Reward indicates good performance (>100)
    # 3. Both legs touching ground
    # Condiciones para aterrizaje exitoso:
    # 1. Episodio terminado naturalmente (no por tiempo)
    # 2. Recompensa indica buen desempeño (>100)
    # 3. Ambas patas tocando el suelo
    if terminated and not truncated and total_reward > 100 and legs_contact:
        # Scale reward and adjust by exploration factor
        # Escalar recompensa y ajustar por factor de exploración
        scaled_reward = int((total_reward / 10) * epsilon)
        
        print(f"[SUCCESS] Successful landing! Reward: {scaled_reward} | ¡Aterrizaje exitoso! Recompensa: {scaled_reward}")
        return {
             'reward': scaled_reward
            }
    
    # No successful landing conditions met
    # No se cumplieron las condiciones de aterrizaje exitoso
    return {
        'reward': 0
    }