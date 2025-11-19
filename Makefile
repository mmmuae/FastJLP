#---------------------------------------------------------------------
# Makefile for BSGS
#
# Author : Jean-Luc PONS

ifdef gpu

SRC = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      GPU/GPUEngine.o Kangaroo.cpp HashTable.cpp \
      Backup.cpp Thread.cpp Check.cpp Network.cpp Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      GPU/GPUEngine.o Kangaroo.o HashTable.o Thread.o \
      Backup.o Check.o Network.o Merge.o PartMerge.o)

else

SRC = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
      Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      Kangaroo.o HashTable.o Thread.o Check.o Backup.o \
      Network.o Merge.o PartMerge.o)

endif

CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc

ifdef gpu
ifndef ccap
ccap != ./detect_cuda.sh
AUTO_CCAP := 1
endif
endif

ifdef gpu
HIGH_SM := $(shell if [ $(ccap) -ge 90 ]; then echo 1; else echo 0; fi)
NVCC_GENCODE = -gencode=arch=compute_$(ccap),code=sm_$(ccap)
ifeq ($(HIGH_SM),1)
NVCC_GENCODE += -gencode=arch=compute_$(ccap),code=compute_$(ccap)
endif
NVCC_HIGH_SM_FLAGS :=
ifeq ($(HIGH_SM),1)
NVCC_HIGH_SM_FLAGS += -Xptxas --allow-expensive-optimizations=true
endif
NVCC_COMMON_FLAGS = -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -I$(CUDA)/include $(NVCC_GENCODE) $(NVCC_HIGH_SM_FLAGS)
ifdef debug
NVCC_BUILD_FLAGS = -G -g
else
NVCC_BUILD_FLAGS = -O2
endif
endif


all: driverquery bsgs

ifdef gpu
ifdef AUTO_CCAP
driverquery:
	@echo "Compiling against automatically detected CUDA compute capability sm_${ccap}"
else
driverquery:
	@echo "Compiling against manually selected CUDA compute capability sm_${ccap}"
endif
	@echo "Build with 'make gpu=1' (or 'make bsgs gpu=1'); detect_cuda.sh is invoked automatically."
else
driverquery:
	@true
endif


ifdef gpu

ifdef debug
CXXFLAGS   = -DWITHGPU -m64  -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart

else

ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   =  -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread

endif

#--------------------------------------------------------------------

ifdef gpu
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) $(NVCC_COMMON_FLAGS) $(NVCC_BUILD_FLAGS) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

bsgs: $(OBJET)
	@echo Making Kangaroo-256...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo-256

$(OBJET): | $(OBJDIR) $(OBJDIR)/SECPK1 $(OBJDIR)/GPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/SECPK1: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p SECPK1

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/SECPK1/*.o
	@rm -f deviceQuery/*.o
	@rm -f deviceQuery/deviceQuery
	@rm -f cuda_version.txt
	@rm -f deviceQuery/cuda_build_log.txt

