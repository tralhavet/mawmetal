BIN = .
CC = gcc
CFLAGS = -O2 -Wall

PROGS = $(BIN)/MinimalAbsentWords

all:
	$(MAKE) progs

progs: $(PROGS)

$(BIN)/MinimalAbsentWords: MinimalAbsentWords.c sarray.o lcp.o stack.o int.h
	$(CC) $(CFLAGS) -o $(BIN)/MinimalAbsentWords sarray.o lcp.o stack.o MinimalAbsentWords.c -lz

sarray.o: sarray.c sarray.h int.h
	$(CC) -c $(CFLAGS) sarray.c

lcp.o: lcp.c lcp.h int.h
	$(CC) -c $(CFLAGS) lcp.c

stack.o: stack.c stack.h int.h
	$(CC) -c $(CFLAGS) stack.c

clean:
	/bin/rm -f *.o $(PROGS)

