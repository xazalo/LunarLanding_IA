import numpy as np

def safe_crash(state, epsilon):
    """
    Refuerzo positivo por choque suave (soft crash).
    Puntúa de 0 a 100 dependiendo de la suavidad del impacto (más cerca de 0 es mejor)
    y del nivel de exploración (epsilon).

    Devuelve también un flag soft_landing: 1 si se considera un aterrizaje suave, 0 si no.
    """
    legs_contact = state[6] == 1 or state[7] == 1
    if not legs_contact:
        return {'reward': 0, 'soft_crash': 0}

    vx = float(state[2])
    vy = float(state[3])

    raw_impact = (np.abs(vx) + np.abs(vy)) / 2
    impact = min(1.0, raw_impact)

    safety_score = (1.0 - impact) ** 2
    reward = 100 * safety_score * epsilon

    # Consideramos aterrizaje suave si el impacto es menor a 0.3
    is_soft_crash = 1 if impact < 0.3 else 0

    return {'reward': round(reward), 'soft_crash': is_soft_crash}
