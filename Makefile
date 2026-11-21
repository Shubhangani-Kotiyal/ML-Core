CXX = g++
CPPFLAGS = -Iinclude
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic
SOURCES = src/ml_algorithms_menu.cpp src/linear_regression.cpp src/kmeans.cpp src/svm.cpp src/hierarchical_clustering.cpp src/pca.cpp src/knn.cpp

.PHONY: all run clean test
all: ml_core

ml_core: $(SOURCES) include/MathUtils.h include/MLAlgorithm.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@

run: ml_core
	./ml_core

clean:
	rm -f ml_core

test: ml_core
	python3 tests/check_csv_limits.py
