CXX=g++
CXXFLAGS=--std=c++11 -Wall -O3
INC=-I/usr/local/include/eigen3

SRCS=infectee.cpp
OBJS=$(SRCS:.cpp=.o)
PROGRAM=run

all: run

$(PROGRAM): $(OBJS) outbreak.cpp
	$(CXX) $(CXXFLAGS) $(LIB) $(INC) $(OBJS) outbreak.cpp -o $@

$(OBJS): %.o : %.cpp %.hpp
	$(CXX) $(CXXFLAGS) $(LIB) $(INC) -c $< -o $@

clean:
	rm -rf $(OBJS) $(PROGRAM)
