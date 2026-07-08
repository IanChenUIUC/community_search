CC = clang++
CFLAGS = -std=c++20 -O2

all: main

main: main.cpp graph.cpp kcore.cpp
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f main
	
