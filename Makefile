CUDAPATH = /home/apps/fas/GPU/cuda_6.0.37

NVCC = $(CUDAPATH)/bin/nvcc
CC = icc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99

NVCCFLAGS = -I$(CUDAPATH)/include -O3

LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm  -lcurand -lcusparse -lcublas

# Compiler-specific flags (by default, we always use sm_20)
GENCODE_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE = $(GENCODE_SM20)

.SUFFIXES : .cu .ptx

all: prbp prbpd prbp_norand prbp_dr

prbp_dr: prbp_dr.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

prbp_norand: prbp_norand.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

prbpd: prbpd.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<
prbp: prbp.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:	
	rm -f *.o $(BINARIES)
