def step_eval(steps, MAX_STEPS):
    """
    Devuelve una recompensa entre -100 y +100 basada en qué tan cerca está `steps` del valor óptimo (mitad del episodio).
    Penaliza duraciones muy cortas o muy largas (ineficiencia).
    """
    # Normalizamos pasos entre 0 y 1 (ajustamos normalización)
    norm_steps = steps / max(MAX_STEPS, 1)

    # Campana invertida centrada en 0.5: -4(x - 0.5)^2 + 1 ∈ [0,1]
    score = -4 * (norm_steps - 0.5) ** 2 + 1
    score = max(0.0, min(1.0, score))  # Clamp a [0, 1]

    # Escalar a [-100, 100]
    reward = (score * 2 - 1) * 10  # ∈ [-100, 100]

    return {
        'reward': round(reward, 2)
    }
