CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

## Path to a file.
SET(JPEG_INCLUDE_DIR "C:/Program Files (x86)/GnuWin32/include")

# Path to a library.
SET(JPEG_LIBRARY "C:/Program Files (x86)/GnuWin32/lib/jpeg.lib")

FIND_PACKAGE(JPEG REQUIRED)

ADD_LIBRARY(jpegrw STATIC
	JPEGReader.cpp JPEGWriter.cpp
    )	

TARGET_LINK_LIBRARIES(jpegrw ${JPEG_LIBRARY})