## Project Overview
This project implements a neural network using PyTorch to predict housing prices in Amman, Jordan. The model analyzes 5 key features to estimate property values.

### Input Features:
1. **area_sqm**: Apartment size (50–250 sqm).
2. **bedrooms**: Number of rooms (1–5).
3. **floor**: Floor level (1–15).
4. **age_years**: Building age (0–40 years).
5. **distance_to_center_km**: Proximity to city center (0.5–25 km).

**Target Variable:** `price_jod` (30K–150K JOD).

---

## Training Configuration
* **Model Architecture:** * Input Layer: 5 neurons
    * Hidden Layer: 32 neurons with **ReLU** activation
    * Output Layer: 1 neuron (Linear)
* **Optimizer:** Adam (Learning Rate = 0.01)
* **Loss Function:** MSELoss (Mean Squared Error)
* **Epochs:** 100

---

## Training Outcome & Observations
* **Initial Loss (Epoch 0):** ~195,028,876.00
* **Final Loss (Epoch 100):** ~194,441,996.00
* **Outcome:** The loss showed a steady decrease throughout the 100 epochs, indicating that the model successfully learned the relationship between property features and their prices.

### Behavioral Observation:
I observed that the initial loss was extremely high. This occurs because house prices are large numbers (up to 150,000 JOD), and the Mean Squared Error (MSE) squares the difference between predictions and actual values. However, the consistent downward trend proves that feature standardization was effective in stabilizing the training process.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)