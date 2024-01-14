# Aegis SAC node

Powered by [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

## Environment variables
* `PORT` - port to listen for action requests on, defaults to 80
* `OBS_SHAPE` - observation shape as a json array
* `ACTION_SHAPE` - action shape as int/json array
* `POLICY` - Stable Baselines PPO policy to use, defaults to `MlpPolicy`
* `BATCH_SIZE` - batch size when training; defaults to 32
* `BUFFER_SIZE` - defaults to 1000000
* `SAVE_STEPS` - Save every <this many> steps; defaults to 1000
* `MODEL_PATH` - load/save path for the Stable Baselines PPO model, defaults to `"models/model"`
* `RESET` - if true, will create a new model instead of loading an existing one
* `VERBOSE` - Stable Baselines PPO2 verbosity level (int)
* `TAU` - defaults to 0.005
* `GAMMA` - reward discount rate, defaults to 0.99

## TODO
* more documentation (env vars, request/response)
* support LSTM/CNN policies
* test :)
* allow done to be set through a route eg `/done`
