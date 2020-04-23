CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic -Werror

OBJS = $(addprefix src/, main.o)
OBJS_TESTS = $(addprefix tests/, test_tensor.o)

TARGET = prophecy
TARGET_TESTS = test

all: $(TARGET)

debug: CXXFLAGS+= -g -fsanitize=address
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OPTI) $^ -o $@

check: $(TARGET_TESTS)
	@echo; ./$(TARGET_TESTS)

$(TARGET_TESTS): $(OBJS_TESTS)
	$(CXX) $(CXXFLAGS) -lcriterion $^ -o $@

clean:
	$(RM) $(TARGET) $(OBJS) $(TARGET_TESTS) $(OBJS_TESTS)

.PHONY: all check clean
