def crash(state, terminated, truncated, total_reward, epsilon, landings, crashes):
    """
    Calculates the crash penalty, modulated by number of crashes and learning stage.
    Also mildly rewards soft landings (both legs contact) early in training.

    Calcula la penalización por accidente, modulada por el número de colisiones y etapa de aprendizaje.
    También bonifica ligeramente los amerizajes suaves (ambas patas apoyadas) al principio del entrenamiento.
    
    Args:
        state (list): Final state of the episode
        terminated (bool): If episode ended naturally
        truncated (bool): If episode ended by timeout
        total_reward (float): Total episode reward
        epsilon (float): Exploration factor
        landings (int): Number of successful landings
        crashes (int): Number of total crashes so far

    Returns:
        dict: {'reward': value, 'type': 'soft' or 'hard'}
    """
    # Check leg contact
    legs_contact = state[6] == 1 and state[7] == 1
    successful_landing = terminated and not truncated and total_reward > 100 and legs_contact

    if successful_landing:
        return {'reward': 0, 'type': 'none'}

    MAX_CRASHES_ALLOWED = 10

    if legs_contact and landings == 0 and crashes < MAX_CRASHES_ALLOWED:
        # Reward soft landings (crashed but both legs touched ground)
        reward = int(10 * epsilon)  # Small positive reward
        crash_type = 'soft-landing'
    elif landings == 0 and crashes < MAX_CRASHES_ALLOWED:
        # Slight penalty for normal crash in early phase
        decay = max(0.0, 1.0 - (crashes / MAX_CRASHES_ALLOWED))
        base_penalty = int(30 * (epsilon - 0.5)) * -1
        reward = int(base_penalty * decay)
        crash_type = 'soft'
    else:
        # Full penalty for hard crash after agent has learned
        reward = int(-100 * (1 - epsilon))
        crash_type = 'hard'

    print(f"[CRASH] Crash detected. Type: {crash_type} | Reward: {reward} | Crashes: {crashes}")
    return {'reward': reward, 'type': crash_type}
