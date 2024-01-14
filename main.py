import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from env import AegisEnv
import json

import sys
import signal
signal.signal(signal.SIGTERM, lambda: sys.exit(0))

PORT = int(os.getenv("PORT", 80))
OBS_SHAPE = json.loads(os.getenv("OBS_SHAPE", "[]"))
ACTION_SHAPE = json.loads(os.getenv("ACTION_SHAPE", "[]"))
POLICY = os.getenv("POLICY", "MlpPolicy")
#default to training every 32 steps
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 1_000_000))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 1000))
RESET = os.getenv("RESET", "").lower() in (True, 'true') #default to false
MODEL_PATH = os.getenv("MODEL_PATH", "models/model")
VERBOSE = int(os.getenv("VERBOSE", 0))
TAU = float(os.getenv("TAU", 0.005))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0003))
GAMMA = float(os.getenv("GAMMA", 0.99))

#TODO: test
# Create environment
env = AegisEnv(OBS_SHAPE, ACTION_SHAPE, port=PORT)
#TODO: support LSTM/CNN policies

def make_sac(env):
  return SAC(
      POLICY,
      env=env,
      batch_size=BATCH_SIZE,
      buffer_size=BUFFER_SIZE,
      learning_rate=LEARNING_RATE,
      gamma=GAMMA,
      tau=TAU,
      verbose=VERBOSE
  )

model = None
#load model if not RESET
if RESET:
    model = make_sac(env)
    model.save(MODEL_PATH)
else:
    try:
        print("Loading", MODEL_PATH)
        model = SAC.load(
            MODEL_PATH,
            env=env,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            verbose=VERBOSE
        )
        print(MODEL_PATH, "loaded")
    except FileNotFoundError as e:
        print(e)
        print('"{}" not found. Creating new model.'.format(MODEL_PATH))
        model = make_sac(env)
        model.save(MODEL_PATH)
        print("Done")

class SaveCallback(BaseCallback):
    def __init__(self, save_steps=1000, verbose: int = 0):
        super().__init__(verbose)
        self.steps_since_last_save = 0
        self.save_steps = save_steps

    def _on_step(self):
        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.save_steps:
          print(f"Saving model to '{MODEL_PATH}'...")
          self.model.save(MODEL_PATH)
          self.steps_since_last_save = 0

        return True

save_callback = SaveCallback(SAVE_STEPS)

#train
while True:
  model.learn(total_timesteps=999999999, callback=save_callback)
