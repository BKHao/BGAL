# BKHao_CODE_2

Write something

## Building BKHao

### Dependence

- CGAL 
- Eigen3
- Boost

### Makefile builds (Linux, other Unixes, and Mac)

```
git clone https://github.com/BKHao/BKHao_CODE_2
cd BKHao_CODE_2
mkdir build && cd build
cmake ..
make -j8
make install
```

After installed, using FindBKHao in your project please.

### MSVC on Windows

```
git clone https://github.com/BKHao/BKHao_CODE_2
```
Open cmake-gui

Where is the source code: BKHao

Where to build the binaries: BKHao/build

Configure->Generate->Open Project

ALL_BUILD->INSTALL





