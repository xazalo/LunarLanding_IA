def crash(state, terminated, truncated, total_reward, epsilon, landings, crashes):
    """
    Detecta si hubo crash:
    - Si terminó o truncado y no fue un aterrizaje exitoso (ambas patas en suelo y buen total_reward).
    - Si hubo crash, devuelve reward=1000 para testeo.
    """

    leg1 = state[6]
    leg2 = state[7]
    legs_contact = (leg1 == 1 and leg2 == 1)

    successful_landing = terminated and not truncated and total_reward > 100 and legs_contact

    # Si terminó y no fue aterrizaje exitoso => crash
    has_crashed = (terminated or truncated) and not successful_landing

    if has_crashed:
        return {'reward': -10, 'crash': 1}
    else:
        return {'reward': 100, 'crash': 0}
