NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 --std=c++17 -I./include
CXX_FLAGS = -O3 --std=c++17 -I./include -pthread

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

TARGET = $(BIN_DIR)/connect4_gpu

OBJS = $(OBJ_DIR)/competitor1_mcts.o \
       $(OBJ_DIR)/competitor2_minimax.o \
       $(OBJ_DIR)/game_engine.o \
       $(OBJ_DIR)/main.o

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) output

$(OBJ_DIR)/competitor1_mcts.o: $(SRC_DIR)/competitor1_mcts.cu include/connect4_common.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/competitor2_minimax.o: $(SRC_DIR)/competitor2_minimax.cu include/connect4_common.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/game_engine.o: $(SRC_DIR)/game_engine.cpp include/connect4_common.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp include/connect4_common.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ -lpthread

run: $(TARGET)
	./$(TARGET) 100000 5

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) output/*.lock output/game_log.json
