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
Set _min_chunk_sz_ value inside of _Init_ function (line: 238) 


### RT-Swap Scheduler
RT-Swap scheduler requires **swap configuration** and **Basic memory info**\

**Swap configuration**
RT-Swap scheduler requires the path to the configuration file with argument **-cfg_path**\
Configuration file should contain following information with following formats.

""" modeltype, period, max swap volume, num of swap allocated objects, indexes of swap allocated objects """ **x per task**

Each memory object allocated by DNN task will assign specific index starting from 0.\
You need to identify which memory objects are assigned to be swapped inside of configuration file.\

***Basic memory info**
Inside of scheduler_fn.cpp, you need to set a total GPU memory amount by **MEM_LIMIT**.\
Inside of scheduler_fn.cpp, you need to set a minimum allocation chunk size of VMM by **MIN_CHUNK**.

## How to Run
You need to run the scheduler first, before run DNN tasks.

./scheduler -sync "{number of tasks}" -cfg_path "{path to configuration file}" \
LD_PRELOAD=./library/libcuhook.so python3 test.py

