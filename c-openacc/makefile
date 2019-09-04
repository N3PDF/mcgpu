# c-omp

EXECUTABLE = c-openacc
CORE = c-openacc.c Vegas.c lepage_test.c

CC = pgcc

OBJDIR = ../obj
BINDIR = ../bin
EXECUTABLEPATH = $(BINDIR)/$(EXECUTABLE)
VPATH += integrands/

OBJFILES = $(addprefix $(OBJDIR)/, $(patsubst %.c,%.o, $(CORE)))

ifeq ($(CC), gcc)
	CC = gcc
	LD = gcc
	CFLAGS = -Wall -Og -ggdb -cpp -fbounds-check -lm
else
	CC = pgcc
	LD = pgcc
	CFLAGS = -acc -Minfo=accel 
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
	@echo "Running " $(EXECUTABLEPATH)
	@$(EXECUTABLEPATH) 5

run_cpu: $(EXECUTABLEPATH)
	@CUDA_VISIBLE_DEVICES="" $(EXECUTABLEPATH) 5

run_debug: $(EXECUTABLEPATH)
	gdb $(EXECUTABLEPATH)

clean:
	@echo "Cleaning obj and bin folders"
	@rm -f $(OBJDIR)/* $(BINDIR)/*
