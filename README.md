# TimeSeriesForecasting

A comparison of some Deep Learning models for time series forecasting applications.

Some standard  architectures, and a few novel architectures.

All implemented in PyTorch.


---

## Features

- Deep learning models for time series forecasting:
 [Recurrent models, Convolutional models, Attention based models, Transformers, Direct Multi-step Forecaster, hybrid models, Neural Turing Machine + DNC, novel models]

- Point estimates and quantile forecasts

- Univariate and multivariate time series

- Exogenous features and horizon (future) features

- Rigorous analysis of the predictions and various forecasting performance metrics

- Metalearning how to best preprocess/deseasonalize/normalize the data for time series inputs; and other training parameters

---

## Repo Contents

    .
    ├── code                    # Main run script, utils
    ├── models                  # Pytorch models
    ├── tasks                   # Tools and utilities
    ├── data                    # Created after running one of the tasks
    ├── output                  # Created after running the training script
    └── README.md

and several other scripts

---

## Setup

Make sure you have Pytorch set up properly. I'm using 1.4.0

Step 1: Create a data set. E.g. by using a default synthetic data set:

```shell
$ python tasks/tsfake_task.py 1
```

or on your own data and task:

```shell
$ python tasks/{TASKNAME}.py -args...
```

Your data set task should implement a Pytorch Dataset class to work with a Pytorch DataLoader.
Use any of the `{TASKNAME}.py` scripts in the `/tasks` folder as a template.
(Discussed in more detail later).


Step 2: Run the training script:

```shell
$ python train.py
```

---

## Example Output

---

## To Do

- **History/Horizon Analysis**

- **sliding window preprocessing (log + standard scale) + learning how to optimally preprocess**

- **include ~ConvS2S implementation**

- **Models not yet implemented:**
    - Transformer
    - Cinar et al. Position based attention encoder-decoder
    - ES-RNN
    - "Order Matters" style Read-Process-Write for very long series
    - NTM/DNC

---