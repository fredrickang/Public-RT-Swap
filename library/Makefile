DEBUG=1

CFLAGS=-ldl -L/usr/local/cuda/lib64 -lcudart -lcuda -pthread

ifeq ($(DEBUG), 1)
CFLAGS+= -DDEBUG
endif

all: libcuhook.so 

libcuhook.so: ./src/cuhooklib.cpp
	g++ -I/usr/local/cuda/include -fPIC -shared -o libcuhook.so ./src/cuhooklib.cpp $(CFLAGS)

clean: 
	rm -f libcuhook.so


