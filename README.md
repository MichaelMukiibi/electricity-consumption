# Electricity Consumption Forecasting with PyTorch

Forecast hourly electricity consumption using the Kaggle PJM Hourly Energy Consumption (PJME) dataset and native PyTorch recurrent neural architectures (RNN, LSTM, GRU).

---

## Features
* **Pure PyTorch Pipeline:** Handles all min-max normalization, tensor transformations, and rolling window slicing natively.
* **Modular Models:** Uses the `ElectricityForecaster` module to switch seamlessly between RNN, LSTM, and GRU backbones.
* **Production-Ready Training:** Executes optimized backpropagation loops with Mean Squared Error (MSE) tracking and Adam optimization.

---

## Getting Started

### 1. Environment Setup
* Ensure Python 3.10+ and PyTorch are installed in your environment.
* Install required dependencies using your package manager:
  ```bash
  uv pip install torch pandas
  ```

### 2. W&B Tracking

* Add W&B API key to .env

```bash
WAND_API_KEY=wandb_
```

* Add key to environment

```bash
source .env
```

### 3. Training 

* ### Use an instant VM
To run the script on an automatically managed T4 GPU instance, the CLI handles instant VM provisioning, pipes the code from your local machine, and automatically destroys the runtime on completion:

```bash
colab run --gpu T4 --keep -s electricity-training train.py --cell LSTM --epochs 10 --batch_size 64 --wandb_key $WANDB_API_KEY 
```

* ### Provision a session
To configure your environment fully before shipping execution code, provision a dedicated T4 GPU session;

```bash
# Run scripted on named remote session
 colab run --gpu T4 --keep -s electricity-training train.py --cell LSTM --epochs 10 --batch_size 64 --wandb_key $WANDB_API_KEY 

# Stop the remote session
colab stop -s electricity-training
```

