import math

def aux_engines(aux_engine_activations, roll, state):
    """
    Reward between -100 and +100 based on:
    + Movement toward horizontal center (x → 0)
    + Proximity to horizontal center
    - Being upside down
    - Using auxiliary engines while grounded

    Recompensa entre -100 y +100 basada en:
    + Movimiento hacia el centro horizontal (x → 0)
    + Proximidad al centro horizontal
    - Estar boca abajo
    - Uso de motores auxiliares en el suelo
    """
    # Extract position and velocity / Extraer posición y velocidad
    x = state[0]  # Horizontal position / Posición horizontal
    vx = state[2]  # Horizontal velocity / Velocidad horizontal
    leg1 = state[6]  # Left leg contact / Contacto pata izquierda
    leg2 = state[7]  # Right leg contact / Contacto pata derecha

    # --- 1. Bonus for moving toward center ---
    # --- 1. Bonificación por moverse hacia el centro ---
    movement_toward_center = -x * vx  # Positive when moving toward center / Positivo cuando se mueve hacia el centro
    direction_bonus = max(min(movement_toward_center * 150, 50), -70)  # Limited between -70 and 50 / Limitado entre -70 y 50

    # --- 2. Bonus for being near center ---
    # --- 2. Bonificación por estar cerca del centro ---
    proximity_bonus = 150 * math.exp(-abs(x) * 50)  # Maximum bonus = 150 / Bonificación máxima = 150

    # --- 3. Penalty for being upside down ---
    # --- 3. Penalización por estar boca abajo ---
    flipped = False
    roll_penalty = 0
    if abs(roll) > math.pi * 0.95:  # ~171 degrees / ~171 grados
        roll_penalty = -100
        flipped = True

    # --- 4. Penalty for using auxiliary engines while grounded ---
    # --- 4. Penalización por usar motores auxiliares en el suelo ---
    grounded_penalty = 0
    total_aux_activations = aux_engine_activations.get('left', 0) + aux_engine_activations.get('right', 0)
    if leg1 == 1.0 and leg2 == 1.0 and total_aux_activations > 0:
        grounded_penalty = -1000000  # Extreme penalty / Penalización extrema

    # --- Calculate total reward ---
    # --- Calcular recompensa total ---
    total_reward = direction_bonus + proximity_bonus + roll_penalty + grounded_penalty
    total_reward = max(min(total_reward, 100), -100)  # Clamp between -100 and +100 / Limitar entre -100 y +100

    return {
        'reward': round(total_reward, 2),
        'flipped': flipped
    }