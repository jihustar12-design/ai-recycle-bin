CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Werror

test: test_decision
	./test_decision

test_decision: test_decision.cpp ../smart_bin/decision.h
	$(CXX) $(CXXFLAGS) -o $@ test_decision.cpp

clean:
	rm -f test_decision

.PHONY: test clean
