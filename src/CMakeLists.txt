cmake_minimum_required(VERSION 3.17)

set(BKHao_HEADERS
        # BK_Algorithm
        BK_Algorithm/BOC/BOC.cpp
        BK_Algorithm/GraphCuts/graph.cpp
        BK_Algorithm/GraphCuts/maxflow.cpp
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
        BK_Optimization/ALGLIB/alglibinternal.h
        BK_Optimization/ALGLIB/alglibmisc.h
        BK_Optimization/ALGLIB/ap.h
        BK_Optimization/ALGLIB/dataanalysis.h
        BK_Optimization/ALGLIB/diffequations.h
        BK_Optimization/ALGLIB/fasttransforms.h
        BK_Optimization/ALGLIB/integration.h
        BK_Optimization/ALGLIB/interpolation.h
        BK_Optimization/ALGLIB/linalg.h
        BK_Optimization/ALGLIB/optimization.h
        BK_Optimization/ALGLIB/solvers.h
        BK_Optimization/ALGLIB/specialfunctions.h
        BK_Optimization/ALGLIB/statistics.h
        BK_Optimization/LBFGS/BKLBFGS.h
        BK_Optimization/GradientDescent/BKGradientDescent.h
        BK_Optimization/LinearSystem/BKLinearSystem.h
        BK_Optimization/LBFGS/BKLBFGS.cpp
        BK_Optimization/GradientDescent/BK
        # BK_PointCloudProcessing
        BK_PointCloudProcessing/Registration/ICP/BKICP.cpp
        # BK_PQP
        BK_PQP/Build.cpp
        BK_PQP/BV.cpp
        BK_PQP/BVTQ.cpp
        BK_PQP/MatVec.cpp
        BK_PQP/OBB_Disjoint.cpp
        BK_PQP/PQP.cpp
        BK_PQP/PQP_Compile.cpp
        BK_PQP/PQP_Internal.cpp
        BK_PQP/RectDist.cpp
        BK_PQP/Tri.cpp
        BK_PQP/TriDist.cpp
        # BK_Reconstruction
        BK_Reconstruction/MarchingTetrahedra/BKMarchingTetrahedra.cpp
        # BK_Tessellation2D
        BK_Tessellation2D/BKSide2D.cpp
        BK_Tessellation2D/BKTessellation2D.cpp
        # BK_Tessellation3D
        BK_Tessellation3D/BKSide3D.cpp
        BK_Tessellation3D/BKTessellation3D.cpp
        )

#add_library(BKHao_CODE ${cpp_files})
#target_link_libraries(BKHao_CODE ${Boost_LIBRARIES})
