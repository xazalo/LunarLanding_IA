# 🚀 AI for Autonomous Spaceship Landing

This project involves the development of an artificial intelligence capable of learning to land a spaceship efficiently, minimizing fuel consumption, and preventing accidents.

---

## 📁 Project Structure

.
├── test.py # Script for testing the AI

├── train.py # AI training script

├── play.py # Visualization of AI performance

├── gym_manager/ # Data import/export and checkpoints

├── models/ # Storage for trained models

├── masters/ # Dynamic adjustment of the epsilon parameter

│ └── epsilon_basic_adjust.py

└── teachers/ # Specialized learning modules

├── main_engine.py

├── aux_engines.py

├── crash.py

├── safe_crash.py

├── landing.py

└── step_eval.py

---

## 🧠 Learning Logic

The AI is trained through specialized modules called **"teachers"**, each responsible for a different facet of the landing process:

### Learning Modules (`teachers/`)

| File             | Description                                                                                   |
|------------------|-----------------------------------------------------------------------------------------------|
| `landing.py`     | Scores the quality of the landing based on epsilon. This is the primary objective.            |
| `crash.py`       | Encourages crashes in early stages when epsilon and score are low, to accelerate learning.    |
| `safe_crash.py`  | Evaluates water landings. Initially rewards them, but penalizes them as epsilon improves.     |
| `step_eval.py`   | Evaluates the number of actions taken. Tolerates many at first, then optimizes fuel consumption. |
| `aux_engines.py` | Controls auxiliary engines to prevent critical angular deviations.                            |
| `main_engine.py` | Teaches efficient use of the main engine to save fuel.                                       |

---

## 🔧 Epsilon Adjustment (`masters/`)

| File                      | Function                                                                                     |
|---------------------------|----------------------------------------------------------------------------------------------|
| `epsilon_basic_adjust.py` | Dynamically adjusts the epsilon value based on received arguments. Seeks to reduce training time when epsilon is high and optimize performance when it is low. |

---

## ⚖️ Learning Weightings

Each module contributes a different weight to the AI's total learning:

- **landing** – 30%  
  The main objective: to land correctly.

- **crash** – 25%  
  It's crucial to avoid collisions, especially in advanced phases.

- **step_eval** – 15%  
  Optimizes the use of actions to reduce fuel consumption.

- **aux_engines** – 12%  
  Ensures horizontal stability during landing.

- **main_engine** – 10%  
  Control of the main engine, less critical than angular stabilization.

- **training_performance** – 5%  
  Compare the last training whit the new and reward improvements.

- **safe_crash** – 3%  
  Water landings as an initial learning tool; less relevant in the long term.

---

## 🎮 Main Files

- `train.py` – Trains the AI model.
- `test.py` – Performs performance tests without learning.
- `play.py` – Visualizes how the AI attempts to land the spaceship.

---

## 💾 Support Directories

- `saves/` – Import/export files and training checkpoints.
- `models/` – Storage for trained models.

---

## 📌 Note

This system is designed for controlled landing scenarios. It does **not** account for external factors such as weather, mechanical failures, or human decision-making.