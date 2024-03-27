# RT-Swap: Addressing GPU Memory Bottlenecks for Real-Time Multi-DNN Inference

This repository contains the source code for RT-Swap, a system designed to mitigate GPU memory bottlenecks in real-time multi-DNN inference tasks. RT-Swap was presented at the 30th IEEE Real-Time Embedded Technology and Applications Symposium (RTAS 2024) in Hong Kong, China, in May 2024.

## Authors
- Woosung Kang
- Jinkyu Lee
- Youngmoon Lee
- Sangeun Oh
- Kilho Lee
- Hoon Sung Chwa

## Darknet vs. PyTorch

- **PyTorch Implementation:** Available on the "main" branch.
- **Darknet Implementation:** Available on the "darknet" branch.

### PyTorch Implementation Details

Due to differences in IPC (Inter-Process Communication) mechanisms between Python and C, minor modifications were made to the IPC communication part of the code. The core components and functionalities remain consistent across both implementations.

### Prerequisites

RT-Swap is compatible with ML frameworks supporting **CUDA 10.2 or higher** due to the requirement for CUDA low-level GPU VMM (Virtual Memory Management) APIs.

- **Implemented PyTorch version:** 2.1.1

### Code Organization

RT-Swap is organized into three main parts: ML-Framework (ml_framework), RT-Swap Library (library), and RT-Swap Scheduler (scheduler).

#### ML-Framework

To integrate RT-Swap with PyTorch, replace the original `module.py` with our modified version located at `ml_framework/module.py`. This enables the required IPC communication.

- **Path to the original module.py:** `home/{username}/.local/lib/python{version}/site-packages/torch/nn/module/module.py`

#### RT-Swap Library

Debugging is enabled by default in the Makefile, leading to extensive logging. To disable debugging, set `DEBUG=0` in the Makefile.

To customize VMM allocation granularity, adjust the `min_chunk_sz` value in the `Init` function (line 238).

#### RT-Swap Scheduler

The scheduler requires a swap configuration and basic memory information to function properly.

- **Swap Configuration:** Specify the configuration file path with `-cfg_path`. The file should follow the format below, listing model type, period, max swap volume, number of swap-allocated objects, and their indexes:

  ```
  modeltype, period, max swap volume, num of swap allocated objects, indexes of swap allocated objects
  ...
  ```

- **Basic Memory Info:** In `scheduler_fn.cpp`, set the total GPU memory limit with `MEM_LIMIT` and the minimum allocation chunk size with `MIN_CHUNK`.

### How to Run

Start the scheduler before running DNN tasks:

1. Run the scheduler: `./scheduler -sync {number of tasks} -cfg_path {path to configuration file}`
2. Execute a DNN task: `LD_PRELOAD=./library/libcuhook.so python3 test.py`
