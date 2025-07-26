def main_engine(steps, engine_activations, epsilon, landings):
    """
    Penaliza el uso del motor principal.
    Penaliza más si se usa mucho durante muchos pasos.
    Penaliza menos si se usa al final del episodio (pocos pasos = cerca del suelo).
    """

    # Ratio de activaciones sobre pasos
    inefficiency_ratio = engine_activations / max(steps, 1)  # ∈ [0,1]

    # Factor de castigo según cuántos pasos hubo: más pasos → más penalización
    step_factor = min(1.0, steps / 100.0)  # Penaliza más si steps > 100

    # Penalización final: fuerte, pero con tolerancia al aterrizaje
    penalty = -1 * inefficiency_ratio * step_factor * (1 + (1 - epsilon))

    return {
        'reward': penalty  # Negativo = penalización
    }
