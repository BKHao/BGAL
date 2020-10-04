# BGAL - Basic Graphics Algorithms Library

A C++ library containing some simple graphics algorithms.

## Building BKHao

### Dependence

- CGAL 
- Eigen3
- Boost

### Makefile builds (Linux, other Unixes, and Mac)

```
git clone https://github.com/BKHao/BGAL
cd BGAL
mkdir build && cd build
cmake ..
make -j8
make install
```

After installed, using FindBGAL in your project please.

### MSVC on Windows

```
git clone https://github.com/BKHao/BGAL
```
Open cmake-gui

```
Where is the source code: BGAL

Where to build the binaries: BGAL/build
```

note: check the location of dependencies and install

Configure->Generate->Open Project

ALL_BUILD->INSTALL

## Test

The example and data is in 'test'. Include BGAL in your project when testing and using it.





