# KMeans_CPP
KMeans implemented in C++ with the thread library and AVX intrinsics

I have created three source codes that implement the KMeans using different methods:

1. task1.cpp calculates the KMeans sequentially.
2. task2.cpp calculates the KMeans using the thread library
3. task3.cpp calculates the KMeans using the thread library and the AVX intrinsics

To compile the first two just use the command "g++ task[1,2].cpp -o \your_executable" or "g++ -mavx2 task3.cpp -o \your_executable" for task3.cpp.