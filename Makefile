CXX=g++
CXXFLAGS=-std=c++17 -O3 -g -I.

# Object files
OBJ_DIR=build
SRC_DIR=lib
TEST_DIR=tests

# Source files
SRC=$(SRC_DIR)/NN.cpp $(SRC_DIR)/ValueStruct.cpp

# Test files
NN_TEST=$(TEST_DIR)/NN.test.cpp
VALUE_TEST=$(TEST_DIR)/ValueStruct.test.cpp

# Test executables
NN_TEST_EXEC=$(OBJ_DIR)/nn-test
VALUE_TEST_EXEC=$(OBJ_DIR)/value-test

TARGET = build/example1
EXAMPLE = examples/example1.cpp

all: tests

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# NN tests
$(NN_TEST_EXEC): $(SRC) $(NN_TEST)
	$(CXX) $(CXXFLAGS) $(SRC) $(NN_TEST) -o $@

# ValueStructure tests
$(VALUE_TEST_EXEC): $(SRC) $(VALUE_TEST)
	$(CXX) $(CXXFLAGS) $(SRC) $(VALUE_TEST) -o $@



$(TARGET): $(OBJ_DIR) $(SRC) $(EXAMPLE)
	$(CXX) $(CXXFLAGS) $(SRC) $(EXAMPLE) -o $(TARGET)


# Run tests
tests: $(NN_TEST_EXEC) $(VALUE_TEST_EXEC)
	$(NN_TEST_EXEC)
	$(VALUE_TEST_EXEC)

examples: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(OBJ_DIR)/*.o $(OBJ_DIR)/*-test
