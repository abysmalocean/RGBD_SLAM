
# PCL library
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories( "/usr/include/eigen3" )
# opencv
find_package( OpenCV REQUIRED ) 

ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

ADD_LIBRARY( utility utility.cpp )
TARGET_LINK_LIBRARIES( utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( main main.cpp )
TARGET_LINK_LIBRARIES( main
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( incParameter incParameter.cpp )
TARGET_LINK_LIBRARIES( incParameter
    utility )



ADD_EXECUTABLE( pointClould pointClould.cpp )
TARGET_LINK_LIBRARIES( pointClould
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( scanData scanData.cpp )
TARGET_LINK_LIBRARIES( scanData
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( jointPCL joinPointCloud.cpp )
TARGET_LINK_LIBRARIES( jointPCL
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( 3DWorld 3DWorld.cpp )
TARGET_LINK_LIBRARIES( 3DWorld
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O )
# CSparse
FIND_PACKAGE( CSparse )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )
FIND_PACKAGE(SuiteSparse)
include_directories(${SUITESPARSE_INCLUDE_DIRS})



ADD_EXECUTABLE( rgbdSlam rgbdSlam.cpp )
TARGET_LINK_LIBRARIES( rgbdSlam
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} 
     ${CSPARSE_LIBRARY}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension)