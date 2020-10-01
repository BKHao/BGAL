cmake_minimum_required(VERSION 3.17)

set(BKHao_SRC
        # BK_Algorithm
        #        BK_Algorithm/BOC/BOC.cpp
        #        BK_Algorithm/GraphCuts/graph.cpp
        #        BK_Algorithm/GraphCuts/maxflow.cpp
        # BK_BaseShape
        BK_BaseShape/BKKDTree.cpp
        BK_BaseShape/BKLine.cpp
        BK_BaseShape/BKPoint.cpp
        BK_BaseShape/BKPolygon.cpp
        BK_BaseShape/BKTriangle.cpp
        # BK_Draw
        BK_Draw/BKDrawPS.cpp
        # BK_Geodesic
        BK_Geodesic/Dijkstra/BKDijkstra.cpp
        BK_Geodesic/BKAbstractMethod.cpp
        # BK_Intergral
        BK_Integral/BKIntegral.cpp
        BK_Integral/Tetrahedron_arbq_rule.cpp
        # BK_Model
        BK_Model/BKManifoldModel.cpp
        BK_Model/BKModel.cpp
        BK_Model/BKModel_Iterator.cpp
        # BK_Optimization
        BK_Optimization/ALGLIB/alglibinternal.cpp
        BK_Optimization/ALGLIB/alglibmisc.cpp
        BK_Optimization/ALGLIB/ap.cpp
        BK_Optimization/ALGLIB/dataanalysis.cpp
        BK_Optimization/ALGLIB/diffequations.cpp
        BK_Optimization/ALGLIB/fasttransforms.cpp
        BK_Optimization/ALGLIB/integration.cpp
        BK_Optimization/ALGLIB/interpolation.cpp
        BK_Optimization/ALGLIB/linalg.cpp
        BK_Optimization/ALGLIB/optimization.cpp
        BK_Optimization/ALGLIB/solvers.cpp
        BK_Optimization/ALGLIB/specialfunctions.cpp
        BK_Optimization/ALGLIB/statistics.cpp
        BK_Optimization/LBFGS/BKLBFGS.cpp
        BK_Optimization/GradientDescent/BKGradientDescent.cpp
        BK_Optimization/LinearSystem/BKLinearSystem.cpp
        BK_Optimization/LBFGS/BKLBFGS.cpp
        # BK_PointCloudProcessing
        BK_PointCloudProcessing/Registration/ICP/BKICP.cpp
        # BK_PQP
        BK_PQP/Build.cpp
        BK_PQP/BV.cpp
        BK_PQP/PQP.cpp
        BK_PQP/TriDist.cpp
        # BK_Reconstruction
        BK_Reconstruction/MarchingTerahedra/BKMarchingTetrahedra.cpp
        # BK_Tessellation2D
        BK_Tessellation2D/BKSide2D.cpp
        BK_Tessellation2D/BKTessellation2D.cpp
        # BK_Tessellation3D
        BK_Tessellation3D/BKSide3D.cpp
        BK_Tessellation3D/BKTessellation3D.cpp
        )

# Get shared lib
add_library(${PROJECT_NAME} SHARED ${BKHao_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VERSION})
# Get static lib
if (BUILD_STATIC_LIB)
    # Get static lib
    add_library(${PROJECT_NAME}_static STATIC ${BKHao_SRC})
    # Modify static lib name
    set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
    set_target_properties(${PROJECT_NAME}_static PROPERTIES VERSION ${VERSION})
    # save shared lib and static lib
    set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
endif ()

