# Flappy Bird with Deep Reinforcement Learning (DQN)

Υλοποίηση agent για το παιχνίδι Flappy Bird με χρήση **Deep Q-Networks (DQN)** σε περιβάλλον custom OpenAI Gym–style.

---

## 🎯 Στόχος

Να εκπαιδευτεί ένας agent που μαθαίνει να ελέγχει το Flappy Bird (flap / no-op) μεγιστοποιώντας το συνολικό reward και το score (pipes passed).

<img width="447" height="590" alt="Στιγμιότυπο οθόνης 2026-05-31 131411" src="https://github.com/user-attachments/assets/eec1749d-3b09-407f-8374-eaac79b53f64" />


---

## 🧠 Μέθοδος

* **State (4D):**

  * `bird_y`
  * `bird_velocity`
  * `pipe_x`
  * `bird_y - pipe_gap_y`

* **Actions:**

  * `0`: Do nothing
  * `1`: Flap

* **Model:**

  * Fully Connected Neural Network (MLP)
  * 4 → 64 → 64 → 2

* **Algorithm:**

  * Deep Q-Network (DQN)
  * Replay Buffer
  * Target Network
  * ε-greedy exploration

---

## 🏗️ Project Structure

```text
proj_1/
│
├── flappy_bird.py      # Environment (game logic + reward)
├── dqn.py              # Neural network
├── agent.py            # DQN agent
├── replay_buffer.py    # Experience replay
├── train.py            # Training loop (2000 episodes)
├── grid_search.py      # Hyperparameter tuning
├── main.py             # Manual / demo play
└── assets/             # Images (bird, etc.)
```

---

## ▶️ Πώς να το τρέξεις

### 1. Εγκατάσταση dependencies

```bash
pip install torch numpy pygame matplotlib
```

---

### 2. Manual Play (δοκιμή περιβάλλοντος)

```bash
python main.py
```

Controls:

* `SPACE`: flap

---

### 3. Hyperparameter Tuning με Grid Search

Για να δοκιμαστούν διαφορετικοί συνδυασμοί υπερπαραμέτρων:

```bash
python grid_search.py
```

Το grid search δοκιμάζει ενδεικτικά:
```python
param_grid = {
    "lr": [1e-3, 5e-4],
    "epsilon_decay": [0.995, 0.998],
    "gamma": [0.99, 0.95],
    "distance_weight": [0.01, 0.02]
}
```

Στο τέλος εμφανίζονται τα καλύτερα configurations, ταξινομημένα βάσει average score.

Παράδειγμα αποτελέσματος:
```
({'lr': 0.0005, 'epsilon_decay': 0.998, 'gamma': 0.95, 'distance_weight': 0.01}, 1.3)
```
---
### 4. Training με Best Configuration
Αφού ολοκληρωθεί το grid search, μπορείς να πάρεις το καλύτερο configuration και να το περάσεις στο `train.py`.

Παράδειγμα:
```python
BEST_CONFIG = {
    "lr": 5e-4,
    "epsilon_decay": 0.998,
    "gamma": 0.95,
    "distance_weight": 0.01,
    "episodes": 2000
}
```

Έπειτα τρέχεις:
```bash
python train.py
```
Το training αποθηκεύει:
- `best_flappy_model.pth`
- `final_flappy_model.pth`
- `training_scores.png`
- `training_rewards.png`

---

## ⚙️ Training Configuration

Το τελικό training έγινε με το καλύτερο configuration που προέκυψε από grid search:

```python
lr = 5e-4
epsilon_decay = 0.998
gamma = 0.95
distance_weight = 0.01
episodes = 2000
```

Η γενική ροή είναι:

```manual test → grid search → select best config → long training run → evaluation plots```

---

## 🧪 Reward Design

* +1 survival
* +10 pipe
* -100 death
* penalty απόστασης από gap
* μικρό penalty για flap

---

## 📈 Αποτελέσματα

* Σταδιακή βελτίωση performance
* Αύξηση average score μετά από tuning
* Visualization μέσω learning curves

---

## ⚠️ Περιορισμοί

* Χαμηλό τελικό score
* Ευαισθησία σε reward shaping
* Περιορισμένος χρόνος εκπαίδευσης

---

## 🚀 Μελλοντική εργασία

* Double DQN
* Dueling DQN
* CNN (image-based input)
* Ενισχυμένη σχεδίαση ανταμοιβών

---

## 📚 Tech Stack

* Python
* PyTorch
* NumPy
* Pygame

---

## 👤 Δημιουργήθηκε από

Ανδρέας Δαρσακλής
1η εργασία στην Υπολογιστική Νοημοσύνη - Βαθιά Ενισχυτική Μάθηση
Τμήμα Πληροφορικής - ΑΠΘ
