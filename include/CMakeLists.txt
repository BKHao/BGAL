cmake_minimum_required(VERSION 3.17)

set(BKHao_HEADERS
        # BK_Algorithm
        BK_Algorithm/BOC/BOC.h
        # BK_BaseShape
        BK_BaseShape/BKKDTree.h
        BK_BaseShape/BKLine.h
        BK_BaseShape/BKPoint.h
        BK_BaseShape/BKPolygon.h
        BK_BaseShape/BKTriangle.h
        # BK_Draw
        BK_Draw/BKDrawPS.h
        # BK_Geodesic
        BK_Geodesic/Dijkstra/BKDijkstra.h
        BK_Geodesic/BKAbstractMethod.h
        # BK_Intergral
        BK_Integral/BKIntegral.h
        BK_Integral/Tetrahedron_arbq_rule.h
        # BK_Model
        BK_Model/BKManifoldModel.h
        BK_Model/BKModel.h
        BK_Model/BKModel_Iterator.h
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
        # BK_PointCloudProcessing
        BK_PointCloudProcessing/Registration/ICP/BKICP.h
        # BK_PQP
        BK_PQP/Build.h
        BK_PQP/BV.h
        BK_PQP/BVTQ.h
        BK_PQP/MatVec.h
        BK_PQP/OBB_Disjoint.h
        BK_PQP/PQP.h
        BK_PQP/PQP_Compile.h
        BK_PQP/PQP_Internal.h
        BK_PQP/RectDist.h
        BK_PQP/Tri.h
        BK_PQP/TriDist.h
        BK_PQP/GetTime.h
        # BK_Reconstruction
        BK_Reconstruction/MarchingTetrahedra/BKMarchingTetrahedra.h
        # BK_Tessellation2D
        BK_Tessellation2D/BKSide2D.h
        BK_Tessellation2D/BKTessellation2D.h
        # BK_Tessellation3D
        BK_Tessellation3D/BKSide3D.h
        BK_Tessellation3D/BKTessellation3D.h
        )

install(FILES ${BKHao_HEADERS} DESTINATION include/BKHao)