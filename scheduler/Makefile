DEBUG=1
LOG=0
CC=g++
CFLAGS=-g -Wall -std=c++11
OBJS=scheduler.o scheduler_fn.o
TARGET=scheduler

OPTS=-Ofast
CFLAGS+=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC 

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
CFLAGS+=-DDEBUG
endif

ifeq ($(LOG), 1)
CFLAGS+=-DDLOG
endif

CFLAGS+=$(OPTS)

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) -o $@  $(OBJS)

scheduler.o: scheduler.hpp scheduler_fn.hpp scheduler.cpp 
scheduler_fn.o: scheduler_fn.hpp scheduler.hpp scheduler_fn.cpp

clean:
	rm -f *.o
	rm -f $(TARGET)
