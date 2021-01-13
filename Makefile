# Legion runtime directory must be specified for compilation.
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Include debugging symbols by default.
DEBUG           ?= 1
# Set maximum number of dimensions.
MAX_DIM         ?= 3
# Set default logging level.
OUTPUT_LEVEL    ?= LEVEL_DEBUG

# Default target
PROG		?= test

# Compute file names.
OUTFILE		?= db_$(PROG)
GEN_SRC		?= $(OUTFILE).cc

# Include Legion's Makefile
include $(LG_RT_DIR)/runtime.mk
