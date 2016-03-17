.DEFAULT_GOAL := all

CC = g++
CFLAGS =
OPTFLAGS = -O3

TARGETS = sb mm
all: $(TARGETS)

mm:
	$(CC) $(OPTFLAGS) -o mm mm.cc

sb:
	$(CC) $(OPTFLAGS) -o sb sb.cc

clean:
	rm -f $(TARGETS)

# eof
