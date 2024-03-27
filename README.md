# Public-RT-Swap
RT-Swap: Addressing GPU Memory Bottlenecks for Real-Time Multi-DNN Inference
Woosung Kang, Jinkyu Lee, Youngmoon Lee, Sangeun Oh, Kilho Lee, Hoon Sung Chwa
In 30th IEEE Real-Time Embedded Technology and Applications Symposium (RTAS 2024), Hong Kong, China, May 2024

## Darknet vs. PyTorch
#### RT-Swap based on PyTorch can be found on "main" branch
#### RT-Sawp based on Darknet can be found on "darknet" branch

### PyTorch Implementation
Due to the difference in IPC communication between Python and C code, minor parts (mainly about the IPC communication) are different from the Darknet version.\
Principal components and functionalities are the same.

## Prerequisites
RT-Swap is compatible with ML frameworks that support **CUDA 10.2 or higher** due to the availability of CUDA low-level GPU VMM APIs.\
Implemented PyTorch version: **2.1.1**

## Code Organization
RT-Swap consists of 3 parts: ML-Framework (ml_framework), RT-Swap Library (library), RT-Swap Scheduler (scheduler)

### ML-Framework
To run RT-Swap with PyTorch, you need to replace the original module.py with our module.py (ml_framework/module.py) to enable the IPC communication.\
Path to origin module.py: home/{username}/.local/lib/python{version}/site-packages/torch/nn/module/module.py

### RT-Swap Library
Currently, DEBUGGING is enabled in Makefile, RT-Swap library will print out lots of informatiom.\
To disable the DEBUG, set DEBUG=0 in Makefile.

To set customized VMM allocation granularity,\
set _min_chunk_sz_ value inside of _Init_ function (line: 238) 


### RT-Swap Scheduler


## How to Run
