CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic
SOURCES = ml_algorithms_menu.cpp LinearRegression.cpp kmean.cpp svm.cpp hierarchical_clustering.cpp pca.cpp knn.cpp

.PHONY: all run clean
all: ml_core

ml_core: $(SOURCES) MathUtils.h
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

run: ml_core
	./ml_core

clean:
	rm -f ml_core
