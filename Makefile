CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic -Werror -g -fsanitize=address

SRC= \
    src/matrix.cc   \

SRC_TESTS= \
    tests/test_matrix.cc

OBJS=$(SRC:.cc=.o)
OBJS_TESTS=$(SRC_TESTS:.cc=.o)

TARGET=prophecy
TARGET_TESTS=test

all: perf

debug: CXXFLAGS+= -g
debug: $(TARGET)

perf: $(TARGET)

$(TARGET): $(OBJS)
	$(LINK.cc) -o $@ $^

$(TARGET_TESTS): $(OBJS_TESTS)
	$(LINK.cc) -o $@ $^ -lcriterion

clean:
	$(RM) $(TARGET) $(OBJS) $(TARGET_TESTS) $(OBJS_TESTS)

.PHONY: all clean
