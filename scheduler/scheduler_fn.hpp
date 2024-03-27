#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <float.h>

//#define DEBUG
#define BLUE "\x1b[34m" //info
#define GREEN "\x1b[32m" // highlight
#define RED "\x1b[31m" // error
#define RESET "\x1b[0m" 
// #define BLUE 
// #define GREEN 
// #define RED 
// #define RESET 

#define commErrchk(ans) {commAssert((ans), __FILE__, __LINE__);}
inline void commAssert(int code, const char *file, int line){
    if(code < 0){
        fprintf(stderr, RED"[scheduler][%s:%3d]: CommError: %d\n"RESET,file,line,code);
        exit(code);
    }
}

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[scheduler][%s:%4d:%30s()] - [%f]: " fmt, \
__FILE__, __LINE__, __func__, what_time_is_it_now(),##args)
#else
#define DEBUG_PRINT(fmt, args...) 
#endif


double get_time_point();

extern FILE **fps;
void set_priority(int priority);
void set_affinity(int core);

task_list_t *create_task_list();
void register_task(task_list_t *task_list, task_info_t *task);
void de_register_task(task_list_t *task_list, task_info_t *task); 
resource_t *create_resource();
node_t* new_node(int pid, double deadline);
queue_t *create_queue();
int enqueue(char * que_name,queue_t *q, int pid, double deadline);
void nodeDelete(queue_t *q, node_t *del);
int dequeue(char * que_name, queue_t *q, resource_t *res);
int dequeue_asyncswap(char * que_name, queue_t *q, task_list_t *task_list, resource_t *res, resource_t * swap_in);
void update_deadline(task_info_t *task, double current_time);
void send_release_time(task_list_t *task_list, queue_t* waiting_queue, queue_t* swapin_queue);
task_info_t *find_task_by_id(task_list_t *task_list, int id);
task_info_t *find_task_by_pid(task_list_t *task_list, int pid);
void print_list(char * name, task_list_t * task_list);
void print_queue(char * name, queue_t * q);
void check_registration(task_list_t *task_list, int reg_fd, resource_t *res);
void do_register(task_list_t *task_list, reg_msg *msg);
void deregister(task_list_t *task_list, reg_msg *msg, resource_t *res);
void decision_handler(int target_pid, task_list_t *task_list);
void init_decision_handler(int target_pid, task_list_t *task_list);
int enqueue_backward(queue_t *q, int pid, int priority);
int dequeue_backward(char * que_name, queue_t *q, resource_t *res);

void init_memory_setting(queue * waiting, task_list_t * task_list, resource_t * swap_in);

int open_channel(char *pipe_name,int mode);
void close_channel(char * pipe_name);
void close_channels(task_info_t * task);
int make_fdset(fd_set *readfds, int reg_fd, task_list_t *task_list);

char *find_char_arg(int argc, char **argv, char *arg, char *def);
int find_int_arg(int argc, char **argv, char *arg, int def);
void del_arg(int argc, char **argv, int index);

void sch_request_handler(task_list_t *task_list, task_info_t *task, resource_t *res, resource_t *init_que, resource_t *swap_in, resource_t *warmup_que);
cudaAPI mm_request_handler(task_list_t * proc_list, task_info_t * proc, resource_t *swap_in);
size_t getmemorysize(map<int,size_t> entry);
task_info_t* choose_victim_init(task_list_t* proc_list, task_info_t* proc);
task_info_t* choose_victim(task_list_t* proc_list, task_info_t* proc);
size_t swapout(task_list_t* proc_list, task_info_t* proc, size_t size, resource_t* swap_in);
size_t swapout_init(task_list_t* proc_list, task_info_t* proc, size_t size);
size_t swapout_async(task_list_t* proc_list, task_info_t* proc, size_t size);
size_t swapin(task_list_t * task_list, task_info_t *target, resource_t* swap_in);
size_t swapin_async(task_list_t * task_list, task_info_t *target);
char * getcudaAPIString(cudaAPI type);

void printShortTasksetInfo(task_list_t *task_list);
void read_cfg(char * path, task_list_t *task_list);
