# fortran-omp

EXECUTABLE = fortran-omp
CORE = fortran-omp.f90 lepage_test.f90
MODULES = Vegas.f90
FC = gfortran

OBJDIR = ../obj
MODDIR = ../mod
BINDIR = ../bin

EXECUTABLEPATH = $(BINDIR)/$(EXECUTABLE)

VPATH += integrands/

OBJFILES = $(addprefix $(OBJDIR)/, $(patsubst %.f90,%.o, $(CORE)))
MODFILES = $(addprefix $(OBJDIR)/, $(patsubst %.f90,%.o, $(MODULES)))

ifeq ($(FC), ifort)
	LD = ifort
	FFLAGS = -g -qopenmp -fpp -D USE_IFORT -module $(MODDIR)
else
	FC = gfortran
	LD = gfortran
	FFLAGS = -Wall -Og -ggdb -cpp -fopenmp -fbounds-check -J $(MODDIR)
endif

# Fortran compilation
ifeq ($(FC), ifort)
	LD = ifort
	FFLAGS = -g -qopenmp -fpp -D USE_IFORT -module $(MODDIR)
else
	FC = gfortran
	LD = gfortran
	FFLAGS = -Wall -Og -ggdb -cpp -fopenmp -fbounds-check -J $(MODDIR)
endif


.PHONY: all clean dirs


$(OBJDIR)/%.o: %.f90 | dirs
	@echo "Building: $<"
	@$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.f90 | $(MODFILES) dirs
	@echo "Building: $<"
	@$(FC) $(FFLAGS) -c $< -o $@

all: $(MODFILES) $(OBJFILES)
	@echo "Linking into: $(EXECUTABLEPATH)"
	@$(LD) -o $(EXECUTABLEPATH) $^ $(FFLAGS) $(LDFLAGS)


dirs: $(OBJDIR) $(MODDIR) $(BINDIR)

$(OBJDIR) $(MODDIR) $(BINDIR):
	@mkdir -p $@

clean:
	@echo "Cleaning obj, mod and bin folders"
	@rm -f $(OBJDIR)/* $(MODDIR)/* $(BINDIR)/*
