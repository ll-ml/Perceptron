CC = gcc
CFLAGS = -Wall -Wextra -g -O2
LDFLAGS = -lm

# Detect OS
UNAME_S := $(shell uname -s)

test_tensor: test_tensor.c tensor.c tensor.h
	$(CC) $(CFLAGS) test_tensor.c tensor.c -o test_tensor $(LDFLAGS)

run: test_tensor
	./test_tensor

clean:
	rm -f test_tensor

ifeq ($(UNAME_S),Darwin)
leaks: test_tensor
	leaks --atExit -- ./test_tensor

leaks-full: test_tensor
	MallocStackLogging=1 leaks --atExit --list -- ./test_tensor
else
valgrind: test_tensor
	valgrind --leak-check=full --show-leak-kinds=all ./test_tensor
endif

ifeq ($(UNAME_S),Darwin)
check: leaks
else
check: valgrind
endif