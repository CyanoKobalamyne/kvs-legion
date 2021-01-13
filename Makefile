# Legion runtime directory must be specified for compilation.
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Set flags for Legion.
DEBUG           ?= 1				# Include debugging symbols
MAX_DIM         ?= 3				# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG		# Compile time logging level

# Default target
PROG		?= test

# Compute file names.
OUTFILE		?= db_$(PROG)
GEN_SRC		?= $(OUTFILE).cc

# Include Legion's Makefile
include $(LG_RT_DIR)/runtime.mk
