CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -Werror

OBJS = $(addprefix src/, main.o                 \
                         matrix.o               \
                         model.o                \
                         layer.o                \
                         hidden_layer.o         \
                         dense_layer.o          \
                         activation_function.o)
OBJS_TESTS = $(addprefix tests/, test_matrix.o)

TARGET = prophecy
TARGET_TESTS = test

all: $(TARGET)

debug: CXXFLAGS+= -g #-fsanitize=address
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

check: $(TARGET_TESTS)
	@echo; ./$(TARGET_TESTS)

$(TARGET_TESTS): $(OBJS) $(OBJS_TESTS)
	$(CXX) $(CXXFLAGS) -lcriterion $^ -o $@

clean:
	$(RM) $(TARGET) $(OBJS) $(TARGET_TESTS) $(OBJS_TESTS)

.PHONY: all check clean
