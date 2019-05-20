CXX       = g++

# флаги компиляции
CXXFLAGS  =
CXXFLAGS += -std=c++11 -O2
CXXFLAGS += -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -fopenmp
CXXFLAGS += $(DEFINES)
CXXFLAGS += $(shell pkg-config --cflags blitz)

# флаги сборки (библиотеки)
LDFLAGS   = 
#LDFLAGS  += -llapack -lblas -lgfortran
#LDFLAGS  += -llapack -lopenblaso -lgfortran -fopenmp -L ./ -ldcmt
LDFLAGS  += -L ./ -ldcmt
LDFLAGS  += $(shell pkg-config --libs blitz)

SOURCES   = main.cc
BINARY    = generate

$(BINARY): $(SOURCES) *.hh Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(BINARY)

clean:
	rm -f $(BINARY) $(VISUAL)
