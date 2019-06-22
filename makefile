CFLAGS = -O2 -Wall `pkg-config --cflags --libs opencv`
SRCS = main.cpp condens.cpp selector.cpp filter.cpp getopt.c
HEADERS =  condens.h selector.h filter.h state.h getopt.h

particle_tracker: $(SRCS) $(HEADERS)
	g++ -o particle_tracker $(SRCS) $(CFLAGS)

.PHONY clean:
	rm -f particle_tracker particle_tracker.exe
