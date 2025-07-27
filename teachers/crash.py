def crash(state, terminated, truncated, total_reward, epsilon, landings, crashes):
    """
    Detects if there was a crash and calculates appropriate reward/penalty.
    / Detecta si hubo un choque y calcula la recompensa/penalización apropiada.

    Args / Argumentos:
        state (list): Current environment state / Estado actual del entorno
        terminated (bool): Whether episode terminated naturally / Si el episodio terminó naturalmente
        truncated (bool): Whether episode was truncated by timeout / Si el episodio fue truncado por tiempo
        total_reward (float): Total reward for the episode / Recompensa total del episodio
        epsilon (float): Current exploration rate / Tasa de exploración actual
        landings (int): Total successful landings so far / Total de aterrizajes exitosos hasta ahora
        crashes (int): Total crashes so far / Total de choques hasta ahora

    Returns / Retorna:
        dict: Dictionary containing:
              - 'reward': Calculated reward/penalty
              - 'crash': 1 if crashed, 0 otherwise
              / Diccionario conteniendo:
              - 'reward': Recompensa/penalización calculada
              - 'crash': 1 si hubo choque, 0 en otro caso
    """
    # Log input parameters / Registrar parámetros de entrada
    print("\n" + "="*50)
    print("Crash Detection / Detección de Choque")
    print("-"*50)
    print(f"Terminated: {terminated} | Terminado: {terminated}")
    print(f"Truncated: {truncated} | Truncado: {truncated}")
    print(f"Total Reward: {total_reward:.2f} | Recompensa Total: {total_reward:.2f}")
    print(f"Epsilon: {epsilon:.3f} | Épsilon: {epsilon:.3f}")
    print(f"Previous Landings: {landings} | Aterrizajes Previos: {landings}")
    print(f"Previous Crashes: {crashes} | Choques Previos: {crashes}")

    # Check legs contact / Verificar contacto de patas
    leg1 = state[6]
    leg2 = state[7]
    legs_contact = (leg1 == 1 and leg2 == 1)

    # Determine successful landing / Determinar aterrizaje exitoso
    successful_landing = terminated and not truncated and total_reward > 100 and legs_contact

    # Determine crash condition / Determinar condición de choque
    has_crashed = (terminated or truncated) and not successful_landing

    if has_crashed:
        # Calculate penalty based on epsilon (more exploration → less penalty)
        # Calcular penalización basada en epsilon (más exploración → menos penalización)
        crash_penalty = round(-100 * (1 - epsilon), 2)
        
        return {
            'reward': crash_penalty,
            'crash': 1,
            'details': {
                'termination_type': 'truncated' if truncated else 'terminated',
                'legs_contact': legs_contact,
                'total_reward': total_reward,
                'penalty_factor': (1 - epsilon)
            }
        }
    else:
        return {
            'reward': 100,
            'crash': 0,
            'details': {
                'termination_type': 'none' if not terminated else ('success' if successful_landing else 'other'),
                'legs_contact': legs_contact,
                'total_reward': total_reward
            }
        }