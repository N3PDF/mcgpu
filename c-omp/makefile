# c-omp

EXECUTABLE = c-omp
CORE = c-omp.c Vegas.c lepage_test.c

CC = gcc

OBJDIR = ../obj
BINDIR = ../bin
EXECUTABLEPATH = $(BINDIR)/$(EXECUTABLE)
VPATH += integrands/

OBJFILES = $(addprefix $(OBJDIR)/, $(patsubst %.c,%.o, $(CORE)))

ifeq ($(CC), icc)
	LD = icc
	CFLAGS = -g -qopenmp -fpp -lm
else
	CC = gcc
	LD = gcc
	CFLAGS = -Wall -Og -ggdb -cpp -fopenmp -fbounds-check -lm
endif

.PHONY: all clean run dirs

$(OBJDIR)/%.o: %.c | dirs
	@echo "Building: $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(EXECUTABLEPATH): $(OBJFILES)
	@echo "Linking into: $(EXECUTABLEPATH)"
	@$(LD) -o $(EXECUTABLEPATH) $^ $(CFLAGS) $(LDFLAGS)

dirs: $(OBJDIR) $(BINDIR)

$(OBJDIR) $(BINDIR):
	@mkdir -p $@

run: $(EXECUTABLEPATH)
	$(EXECUTABLEPATH) 5

run_debug: $(EXECUTABLEPATH)
	gdb $(EXECUTABLEPATH)

clean:
	@echo "Cleaning obj and bin folders"
	@rm -f $(OBJDIR)/* $(BINDIR)/*
