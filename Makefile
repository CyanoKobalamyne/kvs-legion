# Set Legion runtime directory.
LG_RT_DIR=./legion/runtime

# Set flags for Legion.
DEBUG           ?= 1				# Include debugging symbols
MAX_DIM         ?= 3				# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG		# Compile time logging level

# Compute file names.
OUTFILE		?= db_$(PROG)
GEN_SRC		?=	$(OUTFILE).cc

# Include Legion's Makefile
include $(LG_RT_DIR)/runtime.mk
