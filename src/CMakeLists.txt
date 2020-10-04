cmake_minimum_required(VERSION 3.17)

set(BGAL_SRC
        # Algorithm
        #        Algorithm/BOC/BOC.cpp
        # BaseShape
        BaseShape/KDTree.cpp
        BaseShape/Line.cpp
        BaseShape/Point.cpp
        BaseShape/Polygon.cpp
        BaseShape/Triangle.cpp
        # Draw
        Draw/DrawPS.cpp
        # Geodesic
        Geodesic/Dijkstra/Dijkstra.cpp
        Geodesic/AbstractMethod.cpp
        # Intergral
        Integral/Integral.cpp
        Integral/Tetrahedron_arbq_rule.cpp
        # Model
        Model/ManifoldModel.cpp
        Model/Model.cpp
        Model/Model_Iterator.cpp
        # Optimization
        Optimization/ALGLIB/alglibinternal.cpp
        Optimization/ALGLIB/alglibmisc.cpp
        Optimization/ALGLIB/ap.cpp
        Optimization/ALGLIB/dataanalysis.cpp
        Optimization/ALGLIB/diffequations.cpp
        Optimization/ALGLIB/fasttransforms.cpp
        Optimization/ALGLIB/integration.cpp
        Optimization/ALGLIB/interpolation.cpp
        Optimization/ALGLIB/linalg.cpp
        Optimization/ALGLIB/optimization.cpp
        Optimization/ALGLIB/solvers.cpp
        Optimization/ALGLIB/specialfunctions.cpp
        Optimization/ALGLIB/statistics.cpp
        Optimization/LBFGS/LBFGS.cpp
        Optimization/GradientDescent/GradientDescent.cpp
        Optimization/LinearSystem/LinearSystem.cpp
        Optimization/LBFGS/LBFGS.cpp
        # PointCloudProcessing
        PointCloudProcessing/Registration/ICP/ICP.cpp
        # PQP
        PQP/Build.cpp
        PQP/BV.cpp
        PQP/PQP.cpp
        PQP/TriDist.cpp
        # Reconstruction
        Reconstruction/MarchingTerahedra/MarchingTetrahedra.cpp
        # Tessellation2D
        Tessellation2D/Side2D.cpp
        Tessellation2D/Tessellation2D.cpp
        # Tessellation3D
        Tessellation3D/Side3D.cpp
        Tessellation3D/Tessellation3D.cpp
        )

# Get shared lib
add_library(${PROJECT_NAME} SHARED ${BGAL_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VERSION})
# Get static lib
if (BUILD_STATIC_LIB)
    # Get static lib
    add_library(${PROJECT_NAME}_static STATIC ${BGAL_SRC})
    # Modify static lib name
    set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
    set_target_properties(${PROJECT_NAME}_static PROPERTIES VERSION ${VERSION})
    # save shared lib and static lib
    set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
endif ()
