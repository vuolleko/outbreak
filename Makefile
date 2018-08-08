CXX=g++
CXXFLAGS=--std=c++11 -Wall -O3
INC=-I/usr/include/eigen3 -I/usr/include/python3.6m -I/usr/include/boost

SRCS=infectee.cpp
OBJS=$(SRCS:.cpp=.o)
PROGRAM=run
SHARED=outbreak4elfi.so

main: $(PROGRAM) 

lib: CXXFLAGS2=-fPIC -lboost_python3 -lpython3.6m -lboost_numpy3
lib: $(SHARED) 

$(SHARED): $(OBJS) outbreak4elfi.cpp outbreak.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(OBJS) -shared outbreak4elfi.cpp -o $@ $(CXXFLAGS2)

$(PROGRAM): $(OBJS) outbreak.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(OBJS) outbreak.cpp -o $@

$(OBJS): %.o : %.cpp %.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@ $(CXXFLAGS2)

clean:
	rm -rf $(OBJS) $(PROGRAM) $(SHARED)
