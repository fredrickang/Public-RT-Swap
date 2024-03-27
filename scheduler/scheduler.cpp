#include <fcntl.h>
#include <sys/types.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <float.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>


#include "scheduler.hpp"
#include "scheduler_fn.hpp"

#define REGISTRATION strdup("/scheduler")

char *cfg_path = strdup("rtswap_cfg.txt");
int Sync = 0;
using namespace std;

int main(int argc, char **argv){
    Sync = find_int_arg(argc, argv, "-sync", 0);
    cfg_path = find_char_arg(argc, argv, "-cfg_path", "rtswap_cfg.txt");

    int init_sync = Sync;
    int warmup = Sync;

    task_list_t *task_list = create_task_list();
    resource_t *gpu, *init_que, *warmup_que, *swap_in;
    
    gpu = create_resource();
    init_que = create_resource();
    swap_in = create_resource();
    warmup_que = create_resource();

    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);

    int target_pid;
    int fd_head;
    fd_set readfds;

    timeval timeout;
    task_info_t *task;

    do{
        target_pid = -1;

        fd_head = make_fdset(&readfds, reg_fd, task_list);
     
        timeout.tv_sec = 0;
        timeout.tv_usec = 1;

        if(select(fd_head+1, &readfds, NULL, NULL, NULL)){
            if(FD_ISSET(reg_fd, &readfds)) {
                check_registration(task_list, reg_fd, gpu);
            }

            for(task = task_list ->head; task !=NULL; task = task -> next){
                if(FD_ISSET(task->sch_req_fd, &readfds)){
                    sch_request_handler(task_list, task, gpu, init_que, swap_in, warmup_que);
                }
                if(FD_ISSET(task->mm_req_fd, &readfds))
                    mm_request_handler(task_list, task, swap_in);
            }

            if(!(init_que->waiting->count < init_sync)){
                if(init_sync){
                    read_cfg(cfg_path, task_list);
                    init_sync = 0;
                }
                if(init_que -> state == IDLE) target_pid = dequeue_backward("init_que",init_que->waiting, init_que);
                if(target_pid != -1) decision_handler(target_pid, task_list);
            }

            if( !(warmup_que->waiting->count < warmup) && (init_que->waiting->count == 0)){
                if(warmup){
                    init_memory_setting(warmup_que->waiting, task_list, swap_in);
                    warmup = 0;
                }
                if(warmup_que->state == IDLE ) {
                    target_pid = dequeue_asyncswap("Warmup", warmup_que->waiting, task_list, warmup_que, swap_in);
                    while(target_pid == -2){ // update queues during the swap out operation
                        fd_head = make_fdset(&readfds, reg_fd, task_list);
                        if(select(fd_head+1, &readfds, NULL, NULL, &timeout)){
                            for(task_info_t *task = task_list ->head; task != NULL; task = task->next){
                                if(FD_ISSET(task->sch_req_fd, &readfds)) {
                                    DEBUG_PRINT(RED"FD SET(%d)\n"RESET, task->id);
                                    sch_request_handler(task_list, task, gpu, init_que, warmup_que, swap_in); 
                                }
                            }
                        }
                        target_pid = dequeue_asyncswap("Warmup", warmup_que->waiting, task_list, warmup_que, swap_in);
                    }
                }
                if(target_pid != -1) decision_handler(target_pid, task_list);
            }

            if( !(gpu->waiting->count < Sync) && (init_que->waiting->count == 0) && (warmup_que->waiting->count == 0)){
                if(Sync){
                    //printShortTasksetInfo(task_list);
                    swap_in = create_resource();
                    init_memory_setting(gpu->waiting, task_list, swap_in);
                    send_release_time(task_list, gpu->waiting, swap_in->waiting);
                    Sync = 0;
                }

                if(gpu -> state == IDLE){
                    target_pid = dequeue_asyncswap("GPU", gpu->waiting, task_list, gpu, swap_in);
                    while(target_pid == -2){
                        fd_head = make_fdset(&readfds, reg_fd, task_list);
                        if(select(fd_head+1, &readfds, NULL, NULL, &timeout)){
                            for(task_info_t *task = task_list ->head; task != NULL; task = task->next){
                                if(FD_ISSET(task->sch_req_fd, &readfds)) {
                                    sch_request_handler(task_list, task, gpu, init_que, swap_in, warmup_que); 
                                }
                            }
                        }
                        target_pid = dequeue_asyncswap("GPU", gpu->waiting, task_list, gpu, swap_in);
                    }
                }
                if(target_pid != -1) decision_handler(target_pid, task_list);
            }
        }
    }while(!(task_list -> count == 0)); 
}   
