EXECUTABLE := mma

CU_FILES   := tf32-global.cu

CU_DEPS    :=

CC_FILES   := 

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lnvidia-ml -lcudart
NVCC=/usr/local/cuda-12.0/bin/nvcc
NVCCFLAGS= -Xptxas -dlcm=cv -Xptxas -dscm=wt -gencode=arch=compute_86,code=\"sm_86,compute_86\" -lnvidia-ml -std=c++11


OBJS= $(OBJDIR)/fp16_32-global.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
