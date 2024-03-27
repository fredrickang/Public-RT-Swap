#define _GNU_SOURCE

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

#include <math.h>
#include <cuda.h>

#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <algorithm>
#include <cassert>
#include <climits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "curand.h"
#include "hooklib.hpp"

using namespace std;

int non_access_flag = 1; // should be 1 for initial setting // current unified memory version (flag = 0 is correct)
size_t swap_volume;
CUdeviceptr swapspace = 0ULL;
size_t swapspace_pointer = 0;
size_t physical_pointer = 0;
void * host_pinned_ptr = NULL;

size_t current_swapout = 0;

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

/* =====CUDA Hooking APIs===== */

cudaError_t cudaMalloc(void **devPtr, size_t size){
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
        return cudaSuccess;
    }

    int isSwap; 
    isSwap = SendRequest(*devPtr, _cudaMalloc_, size);
    DEBUG_PRINT(BLUE "cudaMalloc[%d] - %lubyte - type: %d\n" RESET, entry_index, size, isSwap);
    if(isSwap == 1){
        // from swapspace + swapspace_pointer to swapspace + swapspace_pointer + size
        // check physical memory allocation 
        // if physical memory is enough pass
        // else allocate 
        // physical memory is not enough
        if(physical_pointer < swapspace_pointer + size){
            size_t needed = swapspace_pointer + size - physical_pointer;
            size_t aligned_sz = ((needed + min_chunk_sz -1)/min_chunk_sz) * min_chunk_sz;
            int iter = int(aligned_sz/ min_chunk_sz);
            for(int i = 0; i< iter; i++){
                CHECK_DRV(cuMemCreate(&pHandle[pHandle_idx], min_chunk_sz, &prop, 0));
                CHECK_DRV(cuMemMap(swapspace + physical_pointer, min_chunk_sz, 0, pHandle[pHandle_idx], 0));
                CHECK_DRV(cuMemSetAccess(swapspace + physical_pointer, min_chunk_sz, &accessDesc, 1ULL));
                physical_pointer += min_chunk_sz;
                pHandle_idx++;
            }
        }

        *devPtr = (void *)swapspace + swapspace_pointer;
        swapspace_pointer += size;
        err = cudaSuccess;
        
    }
    if(isSwap == 0){
        err = lcudaMalloc(devPtr, size);
        
    }
    if(isSwap == -1){
        CHECK_CUDA(cudaHostAlloc(devPtr, size, cudaHostAllocMapped));
        add_entry(&non_access_entry_list, entry_index, *devPtr, size);
        err = cudaSuccess;
    }

    if(isSwap != -1) add_entry(&gpu_entry_list, entry_index, *devPtr, size);
    entry_index++;

    return err;
}


cudaError_t cudaFree(void* devPtr){ /* free */
    SendRequest(devPtr, _cudaFree_, 0);
    del_entry(&gpu_entry_list, devPtr);

    return lcudaFree(devPtr);
}

/* =====BMW core===== */

void* swapThread(void *vargsp){
    DEBUG_PRINT(RED "Swap thread generated\n" RESET);
    sigset_t sigsetmask;
    int signum, ack;
    size_t processed_size = 0;

    sigemptyset(&sigsetmask);
    sigaddset(&sigsetmask, SIGUSR1);
    sigaddset(&sigsetmask, SIGUSR2);
    sigaddset(&sigsetmask, SIGTERM);
        
    tid = gettid();
    
    /* sending tid */
    CHECK_COMM(write(request_fd, &tid, sizeof(int)));

    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));

    msg->entry_index = -1;
    
    while(1){
        if(sigwait(&sigsetmask, &signum) > 0){
            DEBUG_PRINT(RED "SIGWAIT Error\n" RESET);
            exit(-1);
        }
        if(signum == SIGUSR1){            
            processed_size = swapout(signum);
            
            msg->type = _SO_DONE_;
            msg->size = processed_size;

            CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
            DEBUG_PRINT(GREEN "Swap-out Complete\n" RESET);
            SWAP_OUT = true; // swapped flag on
        }
        if(signum == SIGUSR2){
            if(SWAP_OUT) processed_size = swapin(signum);
            
            msg->type = _SI_DONE_;
            msg->size = processed_size;

            CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
            DEBUG_PRINT(GREEN "Swap-in Complete\n" RESET);
            SWAP_OUT = false;   // swapped flag off
        }
        if(signum == SIGTERM){
            DEBUG_PRINT(GREEN "Swap Thread Terminating\n" RESET);
            break;
        }
    }
}

/* Swap-in handler */
size_t swapin(int signum){
    DEBUG_PRINT(GREEN "Swap-in (SIGUSR2) handler callback\n" RESET);
    DEBUG_PRINT(GREEN "Swap-in Size %lu\n", current_swapout);
    // step 1. create physical chunk
    // step 2. map
    // step 3. copy in
    size_t processed_size = 0;
    int chunk_num = int(current_swapout/min_chunk_sz);
    for(int i = 0 ; i < chunk_num; i++){
        size_t offset = i*min_chunk_sz;
        CHECK_DRV(cuMemCreate(&pHandle[i], min_chunk_sz, &prop, 0));
        CHECK_DRV(cuMemMap(swapspace + offset, min_chunk_sz, 0, pHandle[i], 0));
        CHECK_DRV(cuMemSetAccess(swapspace + offset, min_chunk_sz, &accessDesc, 1ULL));
        processed_size += min_chunk_sz;
    }
    CHECK_CUDA(cudaMemcpy((void *)swapspace, host_pinned_ptr, current_swapout, cudaMemcpyHostToDevice));
    current_swapout = 0;
    return processed_size;
}

/* Uncconsecutive Swap out handler */
size_t swapout(int signum){
    if(non_access_flag){
        for(auto iter = non_access_entry_list.begin(); iter != non_access_entry_list.end(); ++iter){
            CHECK_CUDA(cudaFreeHost(iter->second.address));
        }
        non_access_entry_list.empty();
        non_access_flag = 0;
    }
    DEBUG_PRINT(GREEN "Swap-out (SIGUSR1) handler callback\n" RESET);
    size_t processed_size = 0;
    size_t swapout_size;
    CHECK_COMM(read(decision_fd, &swapout_size, sizeof(size_t)));
    
    if(swapout_size == 0) return 0;

    // step 1. copy out the data
    CHECK_CUDA(cudaMemcpy(host_pinned_ptr+current_swapout, (void *)(swapspace+current_swapout), swapout_size, cudaMemcpyDeviceToHost));
    // step 2. unmap the physical chunk
    // step 3. free chunk
    int chunk_num = int(swapout_size/min_chunk_sz);
    int start_idx = int(current_swapout/min_chunk_sz);
    for(int i = start_idx; i < chunk_num+start_idx; i++){
        CHECK_DRV(cuMemUnmap(swapspace+i*min_chunk_sz, min_chunk_sz));
        CHECK_DRV(cuMemRelease(pHandle[i]));
        processed_size += min_chunk_sz;
    }
    current_swapout += swapout_size;
    return processed_size;
}





#ifdef DEBUG2
void print_va_info(void *devPtr){
    if(!exist_in_entry(&gpu_entry_list, devPtr)){
        int closest_idx = floorSearch(devPtr);
        fprintf(stderr, "%d, %p, %d\n", closest_idx, gpu_entry_list[closest_idx].address, gpu_entry_list[closest_idx].size);
    }else{
        fprintf(stderr, "%d, %p, %d\n", find_index_by_ptr(&gpu_entry_list, devPtr), devPtr, gpu_entry_list[find_index_by_ptr(&gpu_entry_list, devPtr)].size);
    }
}
#else
void print_va_info(void *devPtr){

}
#endif


/* =====BMW Interface===== */

void Init(){
    int ret, runtimeVersion;
    sigset_t sigsetmask_main;

    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_DRV(cuCtxSetCurrent(ctx));
    CHECK_DRV(cuCtxGetDevice(&dev));

    /* VMM var setting*/

    memset(&prop, 0, sizeof(prop));
    memset(&accessDesc, 0, sizeof(accessDesc));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)dev;
    prop.win32HandleMetaData = NULL;

    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    cuMemGetAllocationGranularity(&min_chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);


    // customized min chunk size
    min_chunk_sz = 2*(1024*1024);

    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stderr,"Current CUDA runtime Version: %d\n", runtimeVersion);

    pid = getpid();

    char request[30];
    char decision[30];
    
    snprintf(request, 30, "/tmp/mm_request_%d",getpid());
    snprintf(decision, 30, "/tmp/mm_decision_%d",getpid());

    while((request_fd = open(request,O_WRONLY)) < 0);
    while((decision_fd = open(decision,O_RDONLY)) < 0);
    DEBUG_PRINT(BLUE "Request/Decision channel opened\n" RESET);

    atexit(Cleanup);
    DEBUG_PRINT(BLUE "Termination function registered\n" RESET);
    
    // GPU swap space preperation
    CHECK_COMM(read(decision_fd, &swap_volume, sizeof(size_t)));
    if(swap_volume){
        swap_volume = check_granularity(swap_volume, min_chunk_sz);
        int num_chunk = int(swap_volume/min_chunk_sz);
        pHandle = (CUmemGenericAllocationHandle *)malloc(sizeof(CUmemGenericAllocationHandle)*num_chunk);

        // host swap space preperation
        CHECK_CUDA(cudaHostAlloc(&host_pinned_ptr, swap_volume, cudaHostAllocDefault));

        CHECK_DRV(cuMemAddressReserve(&swapspace, swap_volume, 0, 0, 0));
        DEBUG_PRINT(GREEN"VA space reservation for Swap Done!\n"RESET);
    }    
    sigemptyset(&sigsetmask_main);
    
    sigaddset(&sigsetmask_main, SIGUSR1);
    sigaddset(&sigsetmask_main, SIGUSR2);
    sigaddset(&sigsetmask_main, SIGTERM);
    
    ret = pthread_sigmask(SIG_BLOCK, &sigsetmask_main, NULL);
    if(ret != 0) DEBUG_PRINT(RED "sigmask err: %d\n" RESET, ret);    

    pthread_create(&swap_thread_id, NULL, swapThread, NULL);
    DEBUG_PRINT(BLUE "Generating Swap Threads\n" RESET);

    DEBUG_PRINT(GREEN "==Initialization Sequence Done==\n" RESET);
}

size_t check_granularity(size_t requested, size_t min_granularity){
    if(requested % min_granularity != 0 ){
        DEBUG_PRINT(RED"Requested(%lu) are not multiple of min granularity(%lu)\n"RESET, requested, min_granularity);
    }
    return ((requested + min_granularity -1)/min_granularity) * min_granularity;
}

void Cleanup(){
    DEBUG_PRINT(BLUE "Cleaning up...\n" RESET);
    pthread_kill(swap_thread_id, SIGTERM);
    pthread_join(swap_thread_id, NULL);

    DEBUG_PRINT(BLUE "Swap Thread terminated\n" RESET);
    DEBUG_PRINT(GREEN "==BMW Termination Sequence Done==\n" RESET);
}

int SendRequest(void* devPtr, cudaAPI type, size_t size){
    //DEBUG_PRINT_ENTRY();
    
    int ack;
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> size = size;
    
    if(type == _cudaMalloc_)  msg -> entry_index = entry_index;
    if(type == _cudaFree_)  msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);
    if(type == _SWAPIN_) msg -> entry_index = -1;
    
    CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
    CHECK_COMM(read(decision_fd, &ack, sizeof(int)));
    return ack;
}

int SendRequest(void* devPtr, cudaAPI type, size_t size, int index){
    //DEBUG_PRINT_ENTRY();
    
    int ack;
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> size = size;
    
    if(type == _cudaMalloc_)  msg -> entry_index = index;
    if(type == _cudaFree_)  msg -> entry_index = find_index_by_ptr(&gpu_entry_list, devPtr);
    
    CHECK_COMM(write(request_fd, msg, sizeof(req_msg)));
    CHECK_COMM(read(decision_fd, &ack, sizeof(int)));
}

void add_entry(map<int,entry> *entry_list, int index, void* devPtr, size_t size){
    //DEBUG_PRINT(BLUE "Add: {%d, [%p, %lu]}\n" RESET, index, devPtr, size);
    entry tmp;
    tmp.address = devPtr;
    tmp.size = size;
    (*entry_list).insert({index, tmp});

    b_entry btmp;
    btmp.index = index;
    btmp.size= size;
    gpu_bentry_list.insert({devPtr,btmp});
}

void del_entry(map<int,entry> *entry_list, void* devPtr){
    DEBUG_PRINT(BLUE "Del: %p\n" RESET, devPtr);
    (*entry_list).erase(find_index_by_ptr(entry_list, devPtr));

    gpu_bentry_list.erase(devPtr);
}


int find_index_by_ptr(map<int,entry> *entry_list, void* ptr){
    return gpu_bentry_list[ptr].index;
}

bool exist_in_entry(map<int,entry> *entry_list, void *ptr){
    if(gpu_bentry_list.find(ptr) != gpu_bentry_list.end()) return true;
    return false;
}

void add_swap_entry(map<int,gswap>* entry_list, int index, void* origPrt, void* gpuPtr, void* cpuPtr, size_t size){
    DEBUG_PRINT(BLUE "Add (Swap): {%d, [%p, %p, %p, %lu]}\n" RESET, index, origPrt, gpuPtr, cpuPtr, size);
    gswap tmp;
    tmp.origin_address = origPrt;
    tmp.gpu_address = gpuPtr;
    tmp.cpu_address = cpuPtr;
    tmp.size = size;
    (*entry_list).insert({index, tmp});
}


/*  ===== Utils ===== */

#ifdef DEBUG2
void DEBUG_PRINT_ENTRY(){
    DEBUG_PRINT(BLUE "Current GPU Entry: ");
    auto iter = gpu_entry_list.begin();
    while(iter != gpu_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %lu]} ",iter->first, iter->second.address, iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_ENTRY(){

}
#endif

#ifdef DEBUG2
void DEBUG_PRINT_SWAP(){
    DEBUG_PRINT(BLUE "Current SWAP Entry: ");
    auto iter = swap_entry_list.begin();
    while(iter != swap_entry_list.end()){
        fprintf(stderr, "{%d, [%p, %p, %d]} ",iter->first, iter->second.gpu_address, iter->second.cpu_address , iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_SWAP(){

}
#endif


#ifdef DEBUG
void DEBUG_PRINT_PAGETABLE(){
    DEBUG_PRINT(GREEN "Current Page Table Entry: ");
    auto iter = pagetable.begin();
    while(iter != pagetable.end()){
        fprintf(stderr,"{Old: %p, New: %p}",iter->first, iter->second);
        ++iter;
    }
    fprintf(stderr,"\n" RESET);
}
#else
void DEBUG_PRINT_PAGETABLE(){

}
#endif


float checksum(float * input, int size){
    int items = size/sizeof(float);
    float sum = 0;
    for(int i = 0; i < items; i++){
        sum += input[i];
    }
    return sum;
}

char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaFree_:
            return string(_cudaFree_);
    }
}

double what_time_is_it_now()
{
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) == -1) exit(-1);
    
    return (double)time.tv_sec + (double)time.tv_nsec * 0.000000001;
}
