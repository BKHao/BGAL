#define _HAS_STD_BYTE 0
#include <iostream>
#include <fstream>
#include <functional>
#include <io.h>
#include <random>
#include <omp.h>
#include <BGAL/Optimization/LinearSystem/LinearSystem.h>
#include <BGAL/Optimization/ALGLIB/optimization.h>
#include <BGAL/Optimization/LBFGS/LBFGS.h>
#include <BGAL/BaseShape/Point.h>
#include <BGAL/BaseShape/Polygon.h>
#include <BGAL/Tessellation2D/Tessellation2D.h>
#include <BGAL/Draw/DrawPS.h>
#include <BGAL/Integral/Integral.h>
#include <BGAL/Model/ManifoldModel.h>
#include <BGAL/Model/Model_Iterator.h>
#include <BGAL/Optimization/GradientDescent/GradientDescent.h>
#include <BGAL/Tessellation3D/Tessellation3D.h>
#include <BGAL/BaseShape/KDTree.h>
#include <BGAL/PointCloudProcessing/Registration/ICP/ICP.h>
#include <BGAL/Reconstruction/MarchingTetrahedra/MarchingTetrahedra.h>
#include <BGAL/Geodesic/Dijkstra/Dijkstra.h>
#include <BGAL/CVTLike/CPD.h>
#include <BGAL/CVTLike/CVT.h>
#include "nanoflann.hpp"
#include "nanoflann/examples/utils.h"
#include "BGAL/Optimization/ALGLIB/dataanalysis.h"
#include "../build/MAIN/CVTBasedNewton.h"
#include "../build/MAIN/MyRPD.hpp"
#include "../build/MAIN/MyRPD_rnn.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/IO/read_points.h>
#include "../build/MAIN/laplaceIGL.hpp"

using namespace nanoflann;
using namespace alglib;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point11;
typedef Kernel::Vector_3 Vector;
typedef std::pair<Point11, Vector> Pwn;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron11;



int rnnnum = 80;


//Test BOC sign
void BOCSignTest()
{
	BGAL::_BOC::_Sign res = BGAL::_BOC::sign_(1e-6);
	if (res == BGAL::_BOC::_Sign::ZerO)
	{
		std::cout << "0" << std::endl;
	}
	else if (res == BGAL::_BOC::_Sign::PositivE)
	{
		std::cout << "1" << std::endl;
	}
	else if (res == BGAL::_BOC::_Sign::NegativE)
	{
		std::cout << "-1" << std::endl;
	}
	BGAL::_BOC::set_precision_(1e-5);
	res = BGAL::_BOC::sign_(1e-6);
	if (res == BGAL::_BOC::_Sign::ZerO)
	{
		std::cout << "0" << std::endl;
	}
	else if (res == BGAL::_BOC::_Sign::PositivE)
	{
		std::cout << "1" << std::endl;
	}
	else if (res == BGAL::_BOC::_Sign::NegativE)
	{
		std::cout << "-1" << std::endl;
	}
}

//Test LinearSystem
void LinearSystemTest()
{
	Eigen::MatrixXd m(3, 3);
	m.setZero();
	m(0, 0) = 1;
	m(1, 1) = 2;
	m(2, 2) = 3;
	Eigen::SparseMatrix<double> h = m.sparseView();
	Eigen::VectorXd r(3);
	r(0) = 2;
	r(1) = 4;
	r(2) = 6;
	Eigen::VectorXd res = BGAL::_LinearSystem::solve_ldlt(h, r);
	std::cout << res << std::endl;
}
//****************************************

//Test ALGLIB
void function1_grad(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)
{
	//
	// this callback calculates f(x0,x1) = 100*(x0+3)^4 + (x1-3)^4
	// and its derivatives df/d0 and df/dx1
	//
	func = 100 * pow(x[0] + 3, 4) + pow(x[1] - 3, 4);
	grad[0] = 400 * pow(x[0] + 3, 3);
	grad[1] = 4 * pow(x[1] - 3, 3);
}
void ALGLIBTest()
{
	//
	// This example demonstrates minimization of
	//
	//     f(x,y) = 100*(x+3)^4+(y-3)^4
	//
	// using LBFGS method, with:
	// * initial point x=[0,0]
	// * unit scale being set for all variables (see minlbfgssetscale for more info)
	// * stopping criteria set to "terminate after short enough step"
	// * OptGuard integrity check being used to check problem statement
	//   for some common errors like nonsmoothness or bad analytic gradient
	//
	// First, we create optimizer object and tune its properties
	//
	alglib::real_1d_array x = "[0,0]";
	alglib::real_1d_array s = "[1,1]";
	double epsg = 0;
	double epsf = 0;
	double epsx = 0.0000000001;
	alglib::ae_int_t maxits = 0;
	alglib::minlbfgsstate state;
	alglib::minlbfgscreate(1, x, state);
	alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
	alglib::minlbfgssetscale(state, s);

	//
	// Activate OptGuard integrity checking.
	//
	// OptGuard monitor helps to catch common coding and problem statement
	// issues, like:
	// * discontinuity of the target function (C0 continuity violation)
	// * nonsmoothness of the target function (C1 continuity violation)
	// * erroneous analytic gradient, i.e. one inconsistent with actual
	//   change in the target/constraints
	//
	// OptGuard is essential for early prototyping stages because such
	// problems often result in premature termination of the optimizer
	// which is really hard to distinguish from the correct termination.
	//
	// IMPORTANT: GRADIENT VERIFICATION IS PERFORMED BY MEANS OF NUMERICAL
	//            DIFFERENTIATION. DO NOT USE IT IN PRODUCTION CODE!!!!!!!
	//
	//            Other OptGuard checks add moderate overhead, but anyway
	//            it is better to turn them off when they are not needed.
	//
	alglib::minlbfgsoptguardsmoothness(state);
	alglib::minlbfgsoptguardgradient(state, 0.001);

	//
	// Optimize and examine results.
	//
	alglib::minlbfgsreport rep;
	alglib::minlbfgsoptimize(state, function1_grad);
	alglib::minlbfgsresults(state, x, rep);
	printf("%s\n", x.tostring(2).c_str()); // EXPECTED: [-3,3]

	//
	// Check that OptGuard did not report errors
	//
	// NOTE: want to test OptGuard? Try breaking the gradient - say, add
	//       1.0 to some of its components.
	//
	alglib::optguardreport ogrep;
	alglib::minlbfgsoptguardresults(state, ogrep);
	printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
	printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
	printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false
}
//***************************************

//LBFGSTest
void LBFGSTest()
{
	BGAL::_LBFGS::_Parameter param = BGAL::_LBFGS::_Parameter();
	param.epsilon = 1e-10;
	param.is_show = true;
	BGAL::_LBFGS lbfgs(param);
	class problem
	{
	public:
		double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& g)
		{
			double fval = (x(0) - 1) * (x(0) - 1) + (x(1) - 1) * (x(1) - 1);
			g.setZero();
			g(0) = 2 * (x(0) - 1);
			g(1) = 2 * (x(1) - 1);
			return fval;
		}
	};
	problem fun = problem();
	Eigen::VectorXd iterX(2);
	iterX(0) = 53;
	iterX(1) = -68;
	int n = lbfgs.minimize(fun, iterX);
	int a = 53;
	//int n = lbfgs.test(a);
	std::cout << iterX << std::endl;
	std::cout << "n: " << n << std::endl;
}
//***********************************

//BaseShapeTest
void BaseShapeTest()
{
	BGAL::_Point2 p0(0.3, 0.25);
	BGAL::_Point3 p1(1, 2, 3);
	BGAL::_Point3 p2(2, 3, 4);
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(0, 0);
	boundary.insert_(1, 0);
	boundary.insert_(1, 1);
	boundary.insert_(0, 1);
	boundary.end_();
	if (boundary.is_in_(p0))
	{
		std::cout << "yes!" << std::endl;
	}
	std::cout << p1.dot_(p2) << std::endl;
}
//***********************************

//TessellationTest2D
void Tessellation2DTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(BGAL::_Point2(0, 0));
	boundary.insert_(BGAL::_Point2(1, 0));
	boundary.insert_(BGAL::_Point2(1, 1));
	boundary.insert_(BGAL::_Point2(0, 1));
	boundary.end_();
	int num_sites = 2;
	std::vector<BGAL::_Point2> sites;
	//sites.push_back(BGAL::_Point2(0.5, 0.5));
	sites.push_back(BGAL::_Point2(0.5, 0));
	//sites.push_back(BGAL::_Point2(0.4, 0.3));
	//sites.push_back(BGAL::_Point2(0.3, 0.4));
	sites.push_back(BGAL::_Point2(0, 0.5));
	//for (int i = 0; i < num_sites; ++i)
	//{
	//    sites.push_back(BGAL::_Point2(BGAL::_BOC::rand_(), BGAL::_BOC::rand_()));
	//}
	BGAL::_Tessellation2D vor(boundary, sites);
	std::vector<BGAL::_Polygon> cells = vor.get_cell_polygons_();
	std::vector<std::vector<std::pair<int, int>>> edges = vor.get_cells_();
	std::ofstream out("..\\..\\data\\TessellationTest.ps");
	BGAL::_PS ps(out);
	ps.set_bbox_(boundary.bounding_box_());

	for (int i = 0; i < num_sites; ++i)
	{
		ps.draw_polygon_(cells[i], 0.001);
		ps.draw_point_(sites[i], 0.005, 1, 0, 0);
	}
	ps.end_();
	out.close();
}
//***********************************




//CVTLBFGSTest
void CapCVTLBFGSTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(BGAL::_Point2(0, 0));
	boundary.insert_(BGAL::_Point2(1, 0));
	boundary.insert_(BGAL::_Point2(1, 1));
	boundary.insert_(BGAL::_Point2(0, 1));
	boundary.end_();
	std::vector<BGAL::_Point2> sites;
	int num = 200;
	for (int i = 0; i < num; ++i)
	{
		sites.push_back(BGAL::_Point2(BGAL::_BOC::rand_(), BGAL::_BOC::rand_()));
	}
	BGAL::_Tessellation2D voronoi(boundary, sites);
	std::vector<BGAL::_Polygon> cells = voronoi.get_cell_polygons_();
	std::function<double(BGAL::_Point2 p)> rho
		= [](BGAL::_Point2 p)
	{
		return 1.0;
	};
	
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	
	double AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		//cout<<cell.area_()<<endl;
		AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "AreaDiff: " << AreaDiff << endl;

	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgCapVT
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		for (int i = 0; i < num; ++i)
		{
			BGAL::_Point2 p(X(i * 2), X(i * 2 + 1));
			if (!boundary.is_in_(p))
			{
				return std::numeric_limits<double>::max();
			}
			sites[i] = p;
		}
		voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double energy = 0;
		g.setZero();
		for (int i = 0; i < num; ++i)
		{
			energy += (cells[i].area_() - boundary.area_()/num) * (cells[i].area_() - boundary.area_() / num);
		}

		auto VCell = voronoi.get_cells_();
		//cout << VCell.size();
		vector<set<int>> Connections;
		for (int i = 0; i < num; ++i)
		{
			set<int> tmp;
			Connections.push_back(tmp);
		}
		for (int i = 0; i < VCell.size(); ++i)
		{
			for (auto j : VCell[i])
			{
				if (j.second != -1)
				{
					Connections[i].insert(j.second);
					Connections[j.second].insert(i);
				}
			}
		}
		map<pair<int, int>, pair<int, int>> PolyEdge;
		for (int i = 0; i < num; ++i)
		{
			for (auto j : Connections[i])
			{
				int cnt = 0;
				vector<int> tpoints;
				bool flag = 0;
				for (int pi = 0; pi < cells[i].num_(); pi++)
				{
					for (int pj = 0; pj < cells[j].num_(); pj++)
					{
						if (cells[i][pi] == cells[j][pj])
						{
							cnt++;
							int Nxtpi = pi + 1;
							int Nxtpj = pj - 1;
							if (Nxtpi >= cells[i].num_())
							{
								Nxtpi = 0;
							}
							if (Nxtpj < 0)
							{
								Nxtpj = cells[j].num_() - 1;
							}
							if (cells[i][Nxtpi] == cells[j][Nxtpj])
							{
								tpoints.push_back(pi);
								tpoints.push_back(Nxtpi);
							}
							else
							{
								Nxtpi = pi - 1;
								Nxtpj = pj + 1;
								if (Nxtpi < 0)
								{
									Nxtpi = cells[i].num_() - 1;
								}
								if (Nxtpj >= cells[j].num_())
								{
									Nxtpj = 0;
								}
								if (cells[i][Nxtpi] == cells[j][Nxtpj])
								{
									tpoints.push_back(Nxtpi);
									tpoints.push_back(pi);
								}
							}
							flag = 1;
							break;
							/*if (min(i, j) == i)
								tpoints.push_back(pi);
							else
								tpoints.push_back(pj);*/

						}

						
					}
					if (flag)
						break;

				}
				if (tpoints.size() != 2)
				{
					cout << "Error!!!!\n";
				}
				PolyEdge[make_pair(i, j)] = make_pair(tpoints[0], tpoints[1]);

			}
		}


		for (int i = 0; i < num; ++i)
		{
			double sumx = 0.0, sumy = 0.0;
			for (auto j : Connections[i])
			{
				double tmp = (cells[i].area_() - cells[j].area_()) / (sites[j] - sites[i]).length_();

				auto pts = PolyEdge[make_pair(i, j)];
				

				auto Epts1 = cells[i][pts.first];
				auto Epts2 = cells[i][pts.second];

				double l = (Epts1 - Epts2).length_();
				//double l = (Epts1 - Epts2).length_();
				auto vecx = l * ((Epts1.x() + Epts2.x()) / 2.0 - sites[i].x());
				auto vecy = l * ((Epts1.y() + Epts2.y()) / 2.0 - sites[i].y());
				//auto vec = ((Epts1 + Epts2) / 2.0 - sites[i]);
				sumx += tmp * vecx;
				sumy += tmp * vecy;
			}

			g(i * 2) = 2.0 * sumx;
			g(i * 2 + 1) = 2.0 * sumy;

			//Eigen::VectorXd inte = BGAL::_Integral::integral_polygon_fast(
			//	[&](BGAL::_Point2 p)
			//	{
			//		Eigen::VectorXd r(3);
			//		r(0) = rho(p) * ((sites[i] - p).sqlength_());
			//		r(1) = 2 * rho(p) * (sites[i].x() - p.x());
			//		r(2) = 2 * rho(p) * (sites[i].y() - p.y());
			//		return r;
			//	}, cells[i]
			//		);
			////energy += inte(0);
			/*g(i * 2) = inte(1);
			g(i * 2 + 1) = inte(2);*/
		}
		return energy;
	};


	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		//cout << cell.area_() << endl;
		AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "AreaDiff before: " << AreaDiff << endl;

	para.max_linearsearch = 1000;

	para.epsilon = 1e-8;
	BGAL::_LBFGS lbfgs2(para);
	Eigen::VectorXd iterX2(num * 2);
	for (int i = 0; i < num; ++i)
	{
		iterX2(i * 2) = sites[i].x();
		iterX2(i * 2 + 1) = sites[i].y();
	}
	lbfgs2.minimize(fgCapVT, iterX2);
	for (int i = 0; i < num; ++i)
	{
		sites[i] = BGAL::_Point2(iterX2(i * 2), iterX2(i * 2 + 1));
	}
	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		//cout<<cell.area_()<<endl;
		AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "AreaDiff after: " << AreaDiff << endl;

	ofstream out("..\\..\\data\\CapCVTLBFGSTest.ps");
	BGAL::_PS ps2(out);
	ps2.set_bbox_(boundary.bounding_box_());
	for (int i = 0; i < cells.size(); ++i)
	{
		ps2.draw_point_(sites[i], 0.005, 1, 0, 0);
		ps2.draw_polygon_(cells[i], 0.005);
	}
	ps2.end_();
	out.close();

}

void MyCapCVT3DTest()
{

	string modelName = "bunny";
	BGAL::_ManifoldModel model("..\\..\\data\\" + modelName + ".obj");

	std::function<double(BGAL::_Point3& p)> rho = [](BGAL::_Point3& p)
	{
		return 1;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 1e-4;
	para.max_linearsearch = 20;
	BGAL::_CVT3D cvt(model, rho, para, modelName);
	int num = 8000;
	double TotArea = 0;
	cvt.calculate_(num);
	for (int i = 0; i < model.number_faces_(); i++)
	{
		model.vertex_(model.face_(i)[0]);

		double side[3];//存储三条边的长度;

		auto a = model.vertex_(model.face_(i)[0]);
		auto b = model.vertex_(model.face_(i)[1]);
		auto c = model.vertex_(model.face_(i)[2]);

		side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
		side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
		side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
		double p = (side[0] + side[1] + side[2]) / 2;
		double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
		TotArea += area;
	}
	std::cout << "TotArea = " << TotArea << std::endl;


	/*std::vector<BGAL::_Point3> sts;

	std::ifstream inn("..\\..\\data\\" + modelName + "Sites_CVT_v=" + to_string(num) + ".txt");
	
	double x, y, z;
	while (inn >> x >> y >> z)
	{
		sts.push_back(BGAL::_Point3(x, y, z));
	}
	cout << sts.size();*/
	// error here.
	auto sts = cvt.get_sites();
	cvt.calculate_CapVT(sts);

	
	sts = cvt.get_sites();
	cvt.calculate_CapCVTByGD(sts);


	
	const std::vector<BGAL::_Point3>& sites_cap = cvt.get_sites();
	const BGAL::_Restricted_Tessellation3D& RVD_cap = cvt.get_RVD();
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells_cap = RVD_cap.get_cells_();

	


	
	ofstream out("..\\..\\data\\" + modelName + "CapCVT_v=" + to_string(num) + ".obj");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RVD_cap.number_vertices_(); ++i)
	{
		out << "v " << RVD_cap.vertex_(i) << std::endl;
	}
	double TotArea1 = 0, AreaDiff = 0;
	for (int i = 0; i < cells_cap.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		double cellarea = 0;
		for (int j = 0; j < cells_cap[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells_cap[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells_cap[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells_cap[i][j]) + 1 << "/" << i + 1 << std::endl;

			double side[3];//存储三条边的长度;

			auto a = RVD_cap.vertex_(std::get<0>(cells_cap[i][j]));
			auto b = RVD_cap.vertex_(std::get<1>(cells_cap[i][j]));
			auto c = RVD_cap.vertex_(std::get<2>(cells_cap[i][j]));

			side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
			side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
			side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
			double p = (side[0] + side[1] + side[2]) / 2;
			double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
			//cout << area << endl;
			cellarea += area;
		}
		//cout << cellarea << endl;
		AreaDiff += (cellarea - TotArea / num) * (cellarea - TotArea / num);
		TotArea1 += cellarea;
	}

	std::cout << "TotArea1 = " << TotArea1 << std::endl;
	std::cout << "AreaDiff = " << AreaDiff << std::endl;
	out.close();


	auto Vs = RVD_cap.get_sites_();
	auto Edges = RVD_cap.get_edges_();
	set<pair<int, int>> RDT_Edges;
	vector<set<int>> neibors;
	neibors.resize(Vs.size());
	for (int i = 0; i < Edges.size(); i++)
	{
		for (auto ee : Edges[i])
		{
			RDT_Edges.insert(make_pair(min(i, ee.first), max(i, ee.first)));
			neibors[i].insert(ee.first);
			neibors[ee.first].insert(i);
			//cout << ee.first << endl;

		}
	}
	std::ofstream outRDT("..\\..\\data\\" + modelName + "CapRDT_v=" + to_string(num) + ".obj");

	for (auto v : Vs)
	{
		outRDT << "v " << v << endl;
	}
	/*for (auto e : RDT_Edges)
	{
		outRDT << "l " << e.first+1 <<" "<< e.second+1  << endl;
	}*/

	//auto rdtFaces = cvt.rdtFaces;
	std::set<MyFaceCVT> rdtFaces = cvt.rdtFaces;
	for (auto e : RDT_Edges)
	{
		for (int pid : neibors[e.first])
		{
			if (RDT_Edges.find(make_pair(min(pid, e.first), max(pid, e.first))) != RDT_Edges.end())
			{
				if (RDT_Edges.find(make_pair(min(pid, e.second), max(pid, e.second))) != RDT_Edges.end())
				{
					int f1 = pid, f2 = e.first, f3 = e.second;

					int mid;
					if (f1 != max(f1, max(f2, f3)) && f1 != min(f1, min(f2, f3)))
					{
						mid = f1;
					}
					if (f2 != max(f1, max(f2, f3)) && f2 != min(f1, min(f2, f3)))
					{
						mid = f2;
					}
					if (f3 != max(f1, max(f2, f3)) && f3 != min(f1, min(f2, f3)))
					{
						mid = f3;
					}
					rdtFaces.insert(MyFaceCVT(max(f1, max(f2, f3)), mid, min(f1, min(f2, f3))));
				}
			}
		}
	}
	for (auto f : rdtFaces)
	{
		outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << endl;
	}

	outRDT.close();


	
}


void MyCapCVT2DTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(BGAL::_Point2(2.0 - 1.5, 2.0 - 2.0));
	boundary.insert_(BGAL::_Point2(2.0 - 0.5, 2.0 - 2.0));
	boundary.insert_(BGAL::_Point2(2.0 - 0.5, 2.0 - 0.5));
	boundary.insert_(BGAL::_Point2(2.0 + 0.5, 2.0 - 0.5));
	boundary.insert_(BGAL::_Point2(2.0 + 0.5, 2.0 - 2.0));
	boundary.insert_(BGAL::_Point2(2.0 + 1.5, 2.0 - 2.0));
	boundary.insert_(BGAL::_Point2(2.0 + 1.5, 2.0 + 2.0));
	boundary.insert_(BGAL::_Point2(2.0 + 0.5, 2.0 + 2.0));
	boundary.insert_(BGAL::_Point2(2.0 + 0.5, 2.0 + 0.5));
	boundary.insert_(BGAL::_Point2(2.0 - 0.5, 2.0 + 0.5));
	boundary.insert_(BGAL::_Point2(2.0 - 0.5, 2.0 + 2.0));
	boundary.insert_(BGAL::_Point2(2.0 - 1.5, 2.0 + 2.0));
	boundary.end_();
	std::vector<BGAL::_Point2> sites;
	int num = 200;
	for (int i = 0; i < num; ++i)
	{
		auto ppp = BGAL::_Point2(BGAL::_BOC::rand_() * 4.0, BGAL::_BOC::rand_() * 4.0);
		while (!boundary.is_in_(ppp))
		{
			ppp = BGAL::_Point2(BGAL::_BOC::rand_() * 4.0, BGAL::_BOC::rand_() * 4.0);
		}
		sites.push_back(ppp);
	}
	BGAL::_Tessellation2D voronoi(boundary, sites);
	std::vector<BGAL::_Polygon> cells = voronoi.get_cell_polygons_();
	std::function<double(BGAL::_Point2 p)> rho
		= [](BGAL::_Point2 p)
	{
		return 1.0;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 1e-4;
	para.max_linearsearch = 20;
	bool IFCap = 0;
	
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgCapVT
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		for (int i = 0; i < num; ++i)
		{
			BGAL::_Point2 p(X(i * 2), X(i * 2 + 1));
			if (!boundary.is_in_(p))
			{
				p = boundary.nearest_point_(p);
				//return std::numeric_limits<double>::max();
			}
			sites[i] = p;
		}
		voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double energy = 0;
		g.setZero();
		for (int i = 0; i < num; ++i)
		{
			energy += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
		}

		auto VCell = voronoi.get_cells_();
		//cout << VCell.size();
		vector<set<int>> Connections;
		for (int i = 0; i < num; ++i)
		{
			set<int> tmp;
			Connections.push_back(tmp);
		}
		for (int i = 0; i < VCell.size(); ++i)
		{
			for (auto j : VCell[i])
			{
				if (j.second != -1)
				{
					Connections[i].insert(j.second);
					Connections[j.second].insert(i);
				}
			}
		}
		map<pair<int, int>, pair<int, int>> PolyEdge;
		for (int i = 0; i < num; ++i)
		{
			for (auto j : Connections[i])
			{
				int cnt = 0;
				vector<int> tpoints;
				bool flag = 0;
				for (int pi = 0; pi < cells[i].num_(); pi++)
				{
					for (int pj = 0; pj < cells[j].num_(); pj++)
					{
						if (cells[i][pi] == cells[j][pj])
						{
							cnt++;
							int Nxtpi = pi + 1;
							int Nxtpj = pj - 1;
							if (Nxtpi >= cells[i].num_())
							{
								Nxtpi = 0;
							}
							if (Nxtpj < 0)
							{
								Nxtpj = cells[j].num_() - 1;
							}
							if (cells[i][Nxtpi] == cells[j][Nxtpj])
							{
								tpoints.push_back(pi);
								tpoints.push_back(Nxtpi);
							}
							else
							{
								Nxtpi = pi - 1;
								Nxtpj = pj + 1;
								if (Nxtpi < 0)
								{
									Nxtpi = cells[i].num_() - 1;
								}
								if (Nxtpj >= cells[j].num_())
								{
									Nxtpj = 0;
								}
								if (cells[i][Nxtpi] == cells[j][Nxtpj])
								{
									tpoints.push_back(Nxtpi);
									tpoints.push_back(pi);
								}
							}
							flag = 1;
							break;
							/*if (min(i, j) == i)
								tpoints.push_back(pi);
							else
								tpoints.push_back(pj);*/

						}


					}
					if (flag)
						break;

				}
				if (tpoints.size() != 2)
				{
					cout << "Error!!!!\n";
				}
				PolyEdge[make_pair(i, j)] = make_pair(tpoints[0], tpoints[1]);

			}
		}


		for (int i = 0; i < num; ++i)
		{
			double sumx = 0.0, sumy = 0.0;
			for (auto j : Connections[i])
			{
				double tmp = (cells[i].area_() - cells[j].area_()) / (sites[j] - sites[i]).length_();

				auto pts = PolyEdge[make_pair(i, j)];


				auto Epts1 = cells[i][pts.first];
				auto Epts2 = cells[i][pts.second];

				double l = (Epts1 - Epts2).length_();
				//double l = (Epts1 - Epts2).length_();
				auto vecx = l * ((Epts1.x() + Epts2.x()) / 2.0 - sites[i].x());
				auto vecy = l * ((Epts1.y() + Epts2.y()) / 2.0 - sites[i].y());
				//auto vec = ((Epts1 + Epts2) / 2.0 - sites[i]);
				sumx += tmp * vecx;
				sumy += tmp * vecy;
			}

			g(i * 2) = 2.0 * sumx;
			g(i * 2 + 1) = 2.0 * sumy;

			//Eigen::VectorXd inte = BGAL::_Integral::integral_polygon_fast(
			//	[&](BGAL::_Point2 p)
			//	{
			//		Eigen::VectorXd r(3);
			//		r(0) = rho(p) * ((sites[i] - p).sqlength_());
			//		r(1) = 2 * rho(p) * (sites[i].x() - p.x());
			//		r(2) = 2 * rho(p) * (sites[i].y() - p.y());
			//		return r;
			//	}, cells[i]
			//		);
			////energy += inte(0);
			/*g(i * 2) = inte(1);
			g(i * 2 + 1) = inte(2);*/
		}
		return energy;
	};
	std::function<void()> CapVT
		= [&]()
	{
		auto para2 = para;
		//para2.max_iteration = 10;
		para2.epsilon = 1e-6;
		para2.is_show = 0;
		para2.max_linearsearch = 20;
		BGAL::_LBFGS lbfgs2(para2);
		Eigen::VectorXd iterX2(num * 2);
		for (int i = 0; i < num; ++i)
		{
			iterX2(i * 2) = sites[i].x();
			iterX2(i * 2 + 1) = sites[i].y();
		}
		lbfgs2.minimize(fgCapVT, iterX2);
		for (int i = 0; i < num; ++i)
		{
			sites[i] = BGAL::_Point2(iterX2(i * 2), iterX2(i * 2 + 1));
		}
		voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double AreaDiff = 0;
		for (int i = 0; i < cells.size(); ++i)
		{
			auto cell = cells[i];
			//cout<<cell.area_()<<endl;
			AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
		}
		cout << "CapVT Energy after: " << AreaDiff << endl;

		//Eigen::VectorXd tmp_g(num * 2);
		//double tmp_energy = fg(iterX2, tmp_g);
	//	cout << "CVT Energy after: " << tmp_energy << endl;
		

	};
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fg
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		for (int i = 0; i < num; ++i)
		{
			BGAL::_Point2 p(X(i * 2), X(i * 2 + 1));
			if (!boundary.is_in_(p))
			{
				p = boundary.nearest_point_(p);
				//return std::numeric_limits<double>::max();
			}
			sites[i] = p;
		}
		/*if (IFCap)
			CapVT();*/

			voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double energy = 0;
		g.setZero();
		for (int i = 0; i < num; ++i)
		{
			Eigen::VectorXd inte = BGAL::_Integral::integral_polygon_fast(
				[&](BGAL::_Point2 p)
				{
					Eigen::VectorXd r(3);
					r(0) = rho(p) * ((sites[i] - p).sqlength_());
					r(1) = 2 * rho(p) * (sites[i].x() - p.x());
					r(2) = 2 * rho(p) * (sites[i].y() - p.y());
					return r;
				}, cells[i]
					);
			energy += inte(0);
			g(i * 2) = inte(1);
			g(i * 2 + 1) = inte(2);
		}
		return energy;
	};

	std::function<double(const Eigen::VectorXd& X)> CVTLoss
		= [&](const Eigen::VectorXd& X)
	{
		for (int i = 0; i < num; ++i)
		{
			BGAL::_Point2 p(X(i * 2), X(i * 2 + 1));
			if (!boundary.is_in_(p))
			{
				p = boundary.nearest_point_(p);
				//return std::numeric_limits<double>::max();
			}
			sites[i] = p;
		}
		/*if (IFCap)
			CapVT();*/

		voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double energy = 0;
		
		for (int i = 0; i < num; ++i)
		{
			Eigen::VectorXd inte = BGAL::_Integral::integral_polygon_fast(
				[&](BGAL::_Point2 p)
				{
					Eigen::VectorXd r(3);
					r(0) = rho(p) * ((sites[i] - p).sqlength_());
					r(1) = 2 * rho(p) * (sites[i].x() - p.x());
					r(2) = 2 * rho(p) * (sites[i].y() - p.y());
					return r;
				}, cells[i]
					);
			energy += inte(0);
		}
		return energy;
	};
	
	para.epsilon = 1e-8;

	//for (int ii = 1; ii <= 8; ii++)
	//{
	//	
	//	//para.max_iteration = 20;
	//	para.epsilon = 1e-5;
	//	BGAL::_LBFGS lbfgs(para);
	//	Eigen::VectorXd iterX(num * 2);
	//	for (int i = 0; i < num; ++i)
	//	{
	//		iterX(i * 2) = sites[i].x();
	//		iterX(i * 2 + 1) = sites[i].y();
	//	}
	//	IFCap = 0;
	//	lbfgs.minimize(fg, iterX);
	//	for (int i = 0; i < num; ++i)
	//	{
	//		sites[i] = BGAL::_Point2(iterX(i * 2), iterX(i * 2 + 1));
	//	}
	//	
	//	CapVT();

	//	
	//	
	//}
	
	//para.max_iteration = 20;
		para.epsilon = 1e-5;
		BGAL::_LBFGS lbfgs(para);
		Eigen::VectorXd iterX(num * 2);
		for (int i = 0; i < num; ++i)
		{
			iterX(i * 2) = sites[i].x();
			iterX(i * 2 + 1) = sites[i].y();
		}
		IFCap = 0;
		lbfgs.minimize(fg, iterX);
		for (int i = 0; i < num; ++i)
		{
			sites[i] = BGAL::_Point2(iterX(i * 2), iterX(i * 2 + 1));
		}
		cout << "CVT Energy: " << CVTLoss(iterX) << endl;
		CapVT();
	//CVTLoss(iterX);
		for (int i = 0; i < num; ++i)
		{
			iterX(i * 2) = sites[i].x();
			iterX(i * 2 + 1) = sites[i].y();
		}
	cout << "CVT Energy: " << CVTLoss(iterX) << endl;
	
	// GD
	double learning_rate = 10.0;


	

	int cnt = 10;
	while (1)
	{
		CapVT();
		for (int i = 0; i < num; ++i)
		{
			iterX(i * 2) = sites[i].x();
			iterX(i * 2 + 1) = sites[i].y();
		}
		cout << " Energy before: " << CVTLoss(iterX) << endl;

		Eigen::VectorXd g(num * 2);
		double energy = fg(iterX, g);
		Eigen::VectorXd dX = -learning_rate * g;
		for (int i = 0; i < num; ++i)
		{
			iterX(i * 2) += dX(i * 2);
			iterX(i * 2 + 1) += dX(i * 2 + 1);
		}
		
		cout << "Energy after: " << CVTLoss(iterX) << endl;
		cnt++;
		if (cnt > 30)
			break;

	}

	CapVT();
	for (int i = 0; i < num; ++i)
	{
		iterX(i * 2) = sites[i].x();
		iterX(i * 2 + 1) = sites[i].y();
	}
	cout << "Final CVT Energy: " << CVTLoss(iterX) << endl;
	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	double AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		cout<<cell.area_()<<endl;
		AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "Final AreaDiff: " << AreaDiff << endl;

	std::ofstream out("..\\..\\data\\MyCapCVT2DTest.ps");
	BGAL::_PS ps(out);
	ps.set_bbox_(boundary.bounding_box_());
	for (int i = 0; i < cells.size(); ++i)
	{
		ps.draw_point_(sites[i], 0.005, 1, 0, 0);
		ps.draw_polygon_(cells[i], 0.005);
	}
	ps.end_();
	out.close();
	
}

//CVTLBFGSTest
void CVTLBFGSTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	/*boundary.insert_(BGAL::_Point2(0, 0));
	boundary.insert_(BGAL::_Point2(1, 0));
	boundary.insert_(BGAL::_Point2(1, 1));
	boundary.insert_(BGAL::_Point2(0, 1));*/
	boundary.insert_(BGAL::_Point2(2.0-1.5, 2.0-2.0));
	boundary.insert_(BGAL::_Point2(2.0-0.5, 2.0-2.0));
	boundary.insert_(BGAL::_Point2(2.0-0.5, 2.0-0.5));
	boundary.insert_(BGAL::_Point2(2.0+0.5, 2.0-0.5));
	boundary.insert_(BGAL::_Point2(2.0+0.5, 2.0-2.0));
	boundary.insert_(BGAL::_Point2(2.0+1.5, 2.0-2.0));
	boundary.insert_(BGAL::_Point2(2.0+1.5, 2.0+2.0));
	boundary.insert_(BGAL::_Point2(2.0+0.5, 2.0+2.0));
	boundary.insert_(BGAL::_Point2(2.0+0.5, 2.0+0.5));
	boundary.insert_(BGAL::_Point2(2.0-0.5, 2.0+0.5));
	boundary.insert_(BGAL::_Point2(2.0-0.5, 2.0+2.0));
	boundary.insert_(BGAL::_Point2(2.0-1.5, 2.0+2.0));
	boundary.end_();
	std::vector<BGAL::_Point2> sites;
	int num = 200;
	for (int i = 0; i < num; ++i)
	{
		auto ppp = BGAL::_Point2(BGAL::_BOC::rand_()*4.0, BGAL::_BOC::rand_() * 4.0);
		while (!boundary.is_in_(ppp))
		{
			ppp = BGAL::_Point2(BGAL::_BOC::rand_() * 4.0, BGAL::_BOC::rand_() * 4.0);
		}
		sites.push_back(ppp);
	}
	cout << "start.\n";
	BGAL::_Tessellation2D voronoi(boundary, sites);
	std::vector<BGAL::_Polygon> cells = voronoi.get_cell_polygons_();
	std::function<double(BGAL::_Point2 p)> rho
		= [](BGAL::_Point2 p)
	{
		return 1.0;
	};
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fg
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		for (int i = 0; i < num; ++i)
		{
			BGAL::_Point2 p(X(i * 2), X(i * 2 + 1));
			if (!boundary.is_in_(p))
			{
				p = boundary.nearest_point_(p);
				//return std::numeric_limits<double>::max();
			}
			sites[i] = p;
		}
		voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double energy = 0;
		g.setZero();
		for (int i = 0; i < num; ++i)
		{
			Eigen::VectorXd inte = BGAL::_Integral::integral_polygon_fast(
				[&](BGAL::_Point2 p)
				{
					Eigen::VectorXd r(3);
					r(0) = rho(p) * ((sites[i] - p).sqlength_());
					r(1) = 2 * rho(p) * (sites[i].x() - p.x());
					r(2) = 2 * rho(p) * (sites[i].y() - p.y());
					return r;
				}, cells[i]
					);
			energy += inte(0);
			g(i * 2) = inte(1);
			g(i * 2 + 1) = inte(2);
		}
		return energy;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 1e-4;
	para.max_linearsearch = 100;
	BGAL::_LBFGS lbfgs(para);
	Eigen::VectorXd iterX(num * 2);
	for (int i = 0; i < num; ++i)
	{
		iterX(i * 2) = sites[i].x();
		iterX(i * 2 + 1) = sites[i].y();
	}
	lbfgs.minimize(fg, iterX);
	for (int i = 0; i < num; ++i)
	{
		BGAL::_Point2 p(iterX(i * 2), iterX(i * 2 + 1));
		if (!boundary.is_in_(p))
		{
			p = boundary.nearest_point_(p);
			//return std::numeric_limits<double>::max();
		}
		sites[i] = p;
	}
	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	double AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		//cout<<cell.area_()<<endl;
		AreaDiff+= (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "AreaDiff: " << AreaDiff << endl;
	
	std::ofstream out("..\\..\\data\\CVTLBFGSTest.ps");
	BGAL::_PS ps(out);
	ps.set_bbox_(boundary.bounding_box_());
	for (int i = 0; i < cells.size(); ++i)
	{
		ps.draw_point_(sites[i], 0.005, 1, 0, 0);
		ps.draw_polygon_(cells[i], 0.005);
	}
	ps.end_();
	out.close();

	//return;
	/*auto VCell = voronoi.get_cells_();
	cout << VCell.size();
	//map<pair<int,int>, int> CellsConnection;
	vector<set<int>> Connections;
	for (int i = 0; i < num; ++i)
	{
		set<int> tmp;
		Connections.push_back(tmp);
	}
	for (int i = 0; i < VCell.size(); ++i)
	{
		for (auto j : VCell[i])
		{
			//CellsConnection[make_pair(min(i,j.second),max(i,j.second))] = 1;
			if (j.second != -1)
			{
				Connections[i].insert(j.second);
				Connections[j.second].insert(i);
			}
		}
	}
	out.open("..\\..\\data\\Connections.obj");
	for(int i = 0;i<num ;++i)
	{
		out << "v " << sites[i].x() << " " << sites[i].y() << " 0" << endl;
	}
	for (int i = 0; i < num; ++i)
	{
		for(auto j: Connections[i])
			out << "l " << i+1 << ' ' << j+1 << "\n";
	}
	out.close();*/
	

	
	// CapVT test
	/*sites.clear();
	for (int i = 0; i < num; ++i)
	{
		sites.push_back(BGAL::_Point2(BGAL::_BOC::rand_(), BGAL::_BOC::rand_()));
	}*/
	sites[0] = BGAL::_Point2(BGAL::_BOC::rand_(), BGAL::_BOC::rand_());
	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	
	out.open("..\\..\\data\\CVTLBFGSTest.ps");
	BGAL::_PS ps1(out);
	ps1.set_bbox_(boundary.bounding_box_());
	for (int i = 0; i < cells.size(); ++i)
	{
		ps1.draw_point_(sites[i], 0.005, 1, 0, 0);
		ps1.draw_polygon_(cells[i], 0.005);
	}
	ps1.end_();
	out.close();
	

	
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgCapVT
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		for (int i = 0; i < num; ++i)
		{
			BGAL::_Point2 p(X(i * 2), X(i * 2 + 1));
			if (!boundary.is_in_(p))
			{
				p = boundary.nearest_point_(p);
				//return std::numeric_limits<double>::max();
			}
			sites[i] = p;
		}
		voronoi.calculate_(boundary, sites);
		cells = voronoi.get_cell_polygons_();
		double energy = 0;
		g.setZero();
		for (int i = 0; i < num; ++i)
		{
			energy += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
		}

		auto VCell = voronoi.get_cells_();
		//cout << VCell.size();
		vector<set<int>> Connections;
		for (int i = 0; i < num; ++i)
		{
			set<int> tmp;
			Connections.push_back(tmp);
		}
		for (int i = 0; i < VCell.size(); ++i)
		{
			for (auto j : VCell[i])
			{
				if (j.second != -1)
				{
					Connections[i].insert(j.second);
					Connections[j.second].insert(i);
				}
			}
		}
		map<pair<int, int>, pair<int, int>> PolyEdge;
		for (int i = 0; i < num; ++i)
		{
			for (auto j : Connections[i])
			{
				int cnt = 0;
				vector<int> tpoints;
				bool flag = 0;
				for (int pi = 0; pi < cells[i].num_(); pi++)
				{
					for (int pj = 0; pj < cells[j].num_(); pj++)
					{
						if (cells[i][pi] == cells[j][pj])
						{
							cnt++;
							int Nxtpi = pi + 1;
							int Nxtpj = pj - 1;
							if (Nxtpi >= cells[i].num_())
							{
								Nxtpi = 0;
							}
							if (Nxtpj < 0)
							{
								Nxtpj = cells[j].num_() - 1;
							}
							if (cells[i][Nxtpi] == cells[j][Nxtpj])
							{
								tpoints.push_back(pi);
								tpoints.push_back(Nxtpi);
							}
							else
							{
								Nxtpi = pi - 1;
								Nxtpj = pj + 1;
								if (Nxtpi < 0)
								{
									Nxtpi = cells[i].num_() - 1;
								}
								if (Nxtpj >= cells[j].num_())
								{
									Nxtpj = 0;
								}
								if (cells[i][Nxtpi] == cells[j][Nxtpj])
								{
									tpoints.push_back(Nxtpi);
									tpoints.push_back(pi);
								}
							}
							flag = 1;
							break;
							/*if (min(i, j) == i)
								tpoints.push_back(pi);
							else
								tpoints.push_back(pj);*/

						}


					}
					if (flag)
						break;

				}
				if (tpoints.size() != 2)
				{
					cout << "Error!!!!\n";
				}
				PolyEdge[make_pair(i, j)] = make_pair(tpoints[0], tpoints[1]);

			}
		}


		for (int i = 0; i < num; ++i)
		{
			double sumx = 0.0, sumy = 0.0;
			for (auto j : Connections[i])
			{
				double tmp = (cells[i].area_() - cells[j].area_()) / (sites[j] - sites[i]).length_();

				auto pts = PolyEdge[make_pair(i, j)];


				auto Epts1 = cells[i][pts.first];
				auto Epts2 = cells[i][pts.second];

				double l = (Epts1 - Epts2).length_();
				//double l = (Epts1 - Epts2).length_();
				auto vecx = l * ((Epts1.x() + Epts2.x()) / 2.0 - sites[i].x());
				auto vecy = l * ((Epts1.y() + Epts2.y()) / 2.0 - sites[i].y());
				//auto vec = ((Epts1 + Epts2) / 2.0 - sites[i]);
				sumx += tmp * vecx;
				sumy += tmp * vecy;
			}

			g(i * 2) = 2.0 * sumx;
			g(i * 2 + 1) = 2.0 * sumy;

			//Eigen::VectorXd inte = BGAL::_Integral::integral_polygon_fast(
			//	[&](BGAL::_Point2 p)
			//	{
			//		Eigen::VectorXd r(3);
			//		r(0) = rho(p) * ((sites[i] - p).sqlength_());
			//		r(1) = 2 * rho(p) * (sites[i].x() - p.x());
			//		r(2) = 2 * rho(p) * (sites[i].y() - p.y());
			//		return r;
			//	}, cells[i]
			//		);
			////energy += inte(0);
			/*g(i * 2) = inte(1);
			g(i * 2 + 1) = inte(2);*/
		}
		return energy;
	};


	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		//cout << cell.area_() << endl;
		AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "AreaDiff before: " << AreaDiff << endl;

	para.max_linearsearch = 1000;

	para.epsilon = 1e-8;
	BGAL::_LBFGS lbfgs2(para);
	Eigen::VectorXd iterX2(num * 2);
	for (int i = 0; i < num; ++i)
	{
		iterX2(i * 2) = sites[i].x();
		iterX2(i * 2 + 1) = sites[i].y();
	}
	lbfgs2.minimize(fgCapVT, iterX2);
	for (int i = 0; i < num; ++i)
	{
		sites[i] = BGAL::_Point2(iterX2(i * 2), iterX2(i * 2 + 1));
	}
	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	AreaDiff = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		auto cell = cells[i];
		//cout<<cell.area_()<<endl;
		AreaDiff += (cells[i].area_() - boundary.area_() / num) * (cells[i].area_() - boundary.area_() / num);
	}
	cout << "AreaDiff after: " << AreaDiff << endl;

	out.open("..\\..\\data\\CapCVTLBFGSTest.ps");
	BGAL::_PS ps2(out);
	ps2.set_bbox_(boundary.bounding_box_());
	for (int i = 0; i < cells.size(); ++i)
	{
		ps2.draw_point_(sites[i], 0.005, 1, 0, 0);
		ps2.draw_polygon_(cells[i], 0.005);
	}
	ps2.end_();
	out.close();

	
}
//

//DrawTest
void DrawTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(BGAL::_Point2(0, 0));
	boundary.insert_(BGAL::_Point2(1, 0));
	boundary.insert_(BGAL::_Point2(1, 1));
	boundary.insert_(BGAL::_Point2(0, 1));
	boundary.end_();
	BGAL::_Polygon poly;
	poly.start_();
	poly.insert_(BGAL::_Point2(0.25, 0.25));
	poly.insert_(BGAL::_Point2(0.75, 0.25));
	poly.insert_(BGAL::_Point2(0.75, 0.75));
	poly.insert_(BGAL::_Point2(0.25, 0.75));
	poly.end_();
	std::ofstream out("..\\..\\data\\DrawTest.ps");
	BGAL::_PS ps(out);
	ps.set_bbox_(boundary.bounding_box_());
	ps.draw_polygon_(poly, 0.01);
	ps.end_();
	out.close();
}
//***********************************

//IntegralTest
void IntegralTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(BGAL::_Point2(0, 0));
	boundary.insert_(BGAL::_Point2(1, 0));
	boundary.insert_(BGAL::_Point2(1, 1));
	boundary.insert_(BGAL::_Point2(0, 1));
	boundary.end_();
	Eigen::VectorXd r = BGAL::_Integral::integral_polygon_fast(
		[](BGAL::_Point2 p)
		{
			Eigen::VectorXd res(1);
			res(0) = p.x();
			return res;
		}, boundary
	);
	std::cout << r << std::endl;
}
//***********************************

//ModelTest
void ModelTest()
{
	BGAL::_ManifoldModel model("..\\..\\data\\sphere.obj");
	std::cout << "V number: " << model.number_vertices_() << std::endl;
	std::cout << "F number: " << model.number_faces_() << std::endl;
	model.initialization_PQP_();
	if (model.is_in_(BGAL::_Point3(1.25, 0.1, 0.05)))
	{
		std::cout << "in" << std::endl;
	}
}
//***********************************

//Tessellation3DTest
void Tessellation3DTest()
{
	BGAL::_ManifoldModel model("..\\..\\data\\sphere.obj");
	int num = 20;
	std::vector<BGAL::_Point3> sites;
	for (int i = 0; i < num; ++i)
	{
		double phi = BGAL::_BOC::PI() * 2.0 * BGAL::_BOC::rand_();
		double theta = BGAL::_BOC::PI() * BGAL::_BOC::rand_();
		sites.push_back(BGAL::_Point3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)));
	}
	std::vector<double> weights(num, 0);
	BGAL::_Restricted_Tessellation3D RVD(model, sites, weights);
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RVD.get_cells_();
	std::ofstream out("..\\..\\data\\Tessellation3DTest.obj");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RVD.number_vertices_(); ++i)
	{
		out << "v " << RVD.vertex_(i) << std::endl;
	}
	for (int i = 0; i < cells.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		for (int j = 0; j < cells[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
		}
	}
	//for (int i = 0; i < cells.size(); ++i)
	//{
	//	double color = (double)(i % 16) / (15.0);
	//	out << "vt " << color << " 0" << std::endl;
	//	for (int j = 0; j < cells[i].size(); ++j)
	//	{
	//		out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
	//			<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
	//			<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
	//	}
	//}
	out.close();
}
//***********************************

////ReadFileTest
//void ReadFileTest()
//{
//	std::string path("..\\..\\data");
//	std::vector<string> files;
//	int num = BGAL::_BOC::search_files_(path, ".obj", files);
//	std::cout << "num: " << num << std::endl;
//	for (int i = 0; i < files.size(); ++i)
//	{
//		std::cout << files[i] << std::endl;
//	}
//}
////***********************************

//KDTreeTest
void KDTreeTest()
{
	std::vector<BGAL::_Point3> pts;
	int num = 100;
	std::ifstream ip("..\\..\\data\\KDTreeTest.txt");
	for (int i = 0; i < num; ++i)
	{
		double x, y, z;
		ip >> x >> y >> z;
		pts.push_back(BGAL::_Point3(x, y, z));
	}
	BGAL::_KDTree kdt(pts);
	double mind = 0;
	BGAL::_Point3 query(-0.03988281, -0.9964109, 0.02454429);
	int id = kdt.search_(query, mind);
	std::cout << id << " " << setprecision(20) << pts[id] << std::endl;
	std::cout << mind << "    " << (query - pts[id]).length_() << std::endl;
	double bmind = 10000;
	int bid = -1;
	for (int i = 0; i < num; ++i)
	{
		if ((pts[i] - query).length_() < bmind)
		{
			bmind = (pts[i] - query).length_();
			bid = i;
		}
	}
	std::cout << "my:  " << bid << "\t" << bmind << std::endl;
}
//***********************************

//ICPTest
void ICPTest()
{
	std::vector<BGAL::_Point3> pts;
	int num = 2895;
	std::ifstream ip("..\\..\\data\\ICPTest.txt");
	for (int i = 0; i < num; ++i)
	{
		double x, y, z;
		ip >> x >> y >> z;
		pts.push_back(BGAL::_Point3(x, y, z));
	}
	std::vector<BGAL::_Point3> dpts(num);
	Eigen::Matrix3d R;
	R.setIdentity();
	double theta = BGAL::_BOC::PI() * 0.5 * 0.125;
	R(0, 0) = 1 - 2 * sin(theta) * sin(theta);
	R(0, 1) = -2 * cos(theta) * sin(theta);
	R(1, 0) = 2 * cos(theta) * sin(theta);;
	R(1, 1) = 1 - 2 * sin(theta) * sin(theta);
	for (int i = 0; i < num; ++i)
	{
		dpts[i] = pts[i].rotate_(R) + BGAL::_Point3(0.8, 0.2, -0.15);
	}
	BGAL::_ICP icp(pts);
	Eigen::Matrix4d RTM = icp.registration_(dpts);
	std::cout << RTM << std::endl;
	Eigen::Matrix4d RRTM;
	RRTM.setIdentity();
	RRTM.block<3, 3>(0, 0) = R;
	RRTM(0, 3) = 0.8;
	RRTM(1, 3) = 0.2;
	RRTM(2, 3) = -0.15;
	std::cout << RRTM * RTM << std::endl;
}
//***********************************

//MarchingTetrahedraTest
void MarchingTetrahedraTest()
{
	double bboxl = 4.35;
	std::pair<BGAL::_Point3, BGAL::_Point3> bbox(
		BGAL::_Point3(-bboxl, -bboxl, -bboxl), BGAL::_Point3(bboxl, bboxl, bboxl));
	BGAL::_Marching_Tetrahedra MT(bbox, 6);
	MT.set_method_(0);
	function<std::pair<double, BGAL::_Point3>(BGAL::_Point3 p)> dis
		= [](BGAL::_Point3 p)
	{
		double x, y, z;
		x = p.x();
		y = p.y();
		z = p.z();
		double len = (x * x * x * x + y * y * y * y + z * z * z * z) / 16 - (x * x + y * y + z * z) / 4 + 0.4;
		//len = (1 - sqrt(x * x + y * y)) * (1 - sqrt(x * x + y * y)) + z * z - 0.16;
		double dx, dy, dz;
		dx = 4 * x * x * x / 16 - 2 * x / 4;
		dy = 4 * y * y * y / 16 - 2 * y / 4;
		dz = 4 * z * z * z / 16 - 2 * z / 4;
		//dx = -2 * (x / (sqrt(x * x + y * y)) - x);
		//dy = -2 * (y / (sqrt(x * x + y * y)) - y);
		//dz = 2 * z;
		double dl = (dx * dx + dy * dy + dz * dz);
		dx /= dl;
		dy /= dl;
		dz /= dl;
		BGAL::_Point3 rp(x - dx * len, y - dy * len, z - dz * len);
		std::pair<double, BGAL::_Point3> res(len, rp);
		return res;
	};
	BGAL::_ManifoldModel model = MT.reconstruction_(dis);
	model.save_obj_file_("..\\..\\data\\MarchingTetrahedraTest.obj");
}
//***********************************

void GeodesicDijkstraTest()
{
	BGAL::_ManifoldModel model("..\\..\\data\\sphere.obj");
	std::map<int, double> source;
	source[0] = 0;
	BGAL::Geodesic::_Dijkstra dijk(model, source);
	dijk.execute_();
	std::vector<double> distance = dijk.get_distances_();
	double maxd = *(std::max_element(distance.begin(), distance.end()));
	for (int i = 0; i < distance.size(); ++i)
	{
		distance[i] = distance[i] / maxd;
	}
	model.save_scalar_field_obj_file_("..\\..\\data\\GeodesicDijkstraTest.obj", distance);
}
//************************************

//void CVTBasedNewtonTest()
//{
//	BGAL::_Polygon boundary;
//	boundary.start_();
//	boundary.insert_(BGAL::_Point2(0, 0));
//	boundary.insert_(BGAL::_Point2(1, 0));
//	boundary.insert_(BGAL::_Point2(1, 1));
//	boundary.insert_(BGAL::_Point2(0, 1));
//	boundary.end_();
//	std::vector<BGAL::_Point2> sites;
//	int num = 1000;
//	for (int i = 0; i < num; ++i)
//	{
//		sites.push_back(BGAL::_Point2(BGAL::_BOC::rand_(), BGAL::_BOC::rand_()));
//	}
//	auto result = calculate_cvt(boundary, sites);
//
//}

void CPDTest()
{
	BGAL::_ManifoldModel model("..\\..\\data\\sphere.obj");
	std::function<double(BGAL::_Point3& p)> rho = [](BGAL::_Point3& p)
	{
		return 1;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 5e-4;
	BGAL::_CPD3D cpd(model, rho, para);
	cpd._omt_eps = 5e-4;
	double sum_mass = 0;
	for (auto fit = model.face_begin(); fit != model.face_end(); ++fit)
	{
		auto tri = model.face_(fit.id());
		sum_mass += tri.area_();
	}
	int num = 200;
	std::vector<double> capacity(num, sum_mass / num);
	std::vector<BGAL::_Point3> _sites;
	_sites.resize(num);
	for (int i = 0; i < num; ++i)
	{
		int fid = rand() % model.number_faces_();
		double l0, l1, l2, sum;
		l0 = BGAL::_BOC::rand_();
		l1 = BGAL::_BOC::rand_();
		l2 = BGAL::_BOC::rand_();
		sum = l0 + l1 + l2;
		l0 /= sum;
		l1 /= sum;
		l2 /= sum;
		_sites[i] = model.face_(fid).point(0) * l0 + model.face_(fid).point(1) * l1 + model.face_(fid).point(2) * l2;
	}
	cpd.calculate_(capacity, _sites);
	const std::vector<BGAL::_Point3>& sites = cpd.get_sites();
	const std::vector<double>& weights = cpd.get_weights();
	const BGAL::_Restricted_Tessellation3D& RPD = cpd.get_RPD();
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RPD.get_cells_();
	std::cout << "mass - capacity" << std::endl;
	for (int i = 0; i < num; ++i)
	{
		double cal_mass = 0;
		for (int j = 0; j < cells[i].size(); ++j)
		{
			BGAL::_Triangle3 tri(RPD.vertex_(std::get<0>(cells[i][j])), RPD.vertex_(std::get<1>(cells[i][j])), RPD.vertex_(std::get<2>(cells[i][j])));
			cal_mass += tri.area_();
		}
		std::cout << cal_mass << "\t" << capacity[i] << "\t" << cal_mass - capacity[i] << std::endl;
	}
	std::ofstream out("..\\..\\data\\CPD3DTest.obj");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RPD.number_vertices_(); ++i)
	{
		out << "v " << RPD.vertex_(i) << std::endl;
	}
	for (int i = 0; i < cells.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		for (int j = 0; j < cells[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
		}
	}
	out.close();
}


void CVTCPDTest(string modelName, std::vector<BGAL::_Point3> sites)
{
	BGAL::_ManifoldModel model("..\\..\\data\\" + modelName + ".obj");
	std::function<double(BGAL::_Point3& p)> rho = [](BGAL::_Point3& p)
	{
		return 1;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 5e-4;
	BGAL::_CPD3D cpd(model, rho, para);
	cpd._omt_eps = 5e-4;
	double sum_mass = 0;
	for (auto fit = model.face_begin(); fit != model.face_end(); ++fit)
	{
		auto tri = model.face_(fit.id());
		sum_mass += tri.area_();
	}
	int num = sites.size();
	std::vector<double> capacity(num, sum_mass / num);

	cpd.calculate_(capacity, sites);
	//const std::vector<BGAL::_Point3>& sites = cpd.get_sites();
	const std::vector<double>& weights = cpd.get_weights();
	const BGAL::_Restricted_Tessellation3D& RPD = cpd.get_RPD();
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RPD.get_cells_();
	

	auto Vs = RPD.get_sites_();
	auto Edges = RPD.get_edges_();
	set<pair<int, int>> RDT_Edges;
	vector<set<int>> neibors;
	neibors.resize(Vs.size());
	for (int i = 0; i < Edges.size(); i++)
	{
		for (auto ee : Edges[i])
		{
			RDT_Edges.insert(make_pair(min(i, ee.first), max(i, ee.first)));
			neibors[i].insert(ee.first);
			neibors[ee.first].insert(i);
			//cout << ee.first << endl;

		}
	}
	std::ofstream outRDT("..\\..\\data\\" + modelName + "RDT_v=" + to_string(num) + ".obj");

	for (auto v : Vs)
	{
		outRDT << "v " << v << endl;
	}
	/*for (auto e : RDT_Edges)
	{
		outRDT << "l " << e.first+1 <<" "<< e.second+1  << endl;
	}*/

	set<MyFace> rdtFaces;

	for (auto e : RDT_Edges)
	{
		for (int pid : neibors[e.first])
		{
			if (RDT_Edges.find(make_pair(min(pid, e.first), max(pid, e.first))) != RDT_Edges.end())
			{
				if (RDT_Edges.find(make_pair(min(pid, e.second), max(pid, e.second))) != RDT_Edges.end())
				{
					int f1 = pid, f2 = e.first, f3 = e.second;

					int mid;
					if (f1 != max(f1, max(f2, f3)) && f1 != min(f1, min(f2, f3)))
					{
						mid = f1;
					}
					if (f2 != max(f1, max(f2, f3)) && f2 != min(f1, min(f2, f3)))
					{
						mid = f2;
					}
					if (f3 != max(f1, max(f2, f3)) && f3 != min(f1, min(f2, f3)))
					{
						mid = f3;
					}
					rdtFaces.insert(MyFace(max(f1, max(f2, f3)), mid, min(f1, min(f2, f3))));
				}
			}
		}
	}
	for (auto f : rdtFaces)
	{
		outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << endl;
	}

	outRDT.close();
	
	
	std::cout << "mass - capacity" << std::endl;
	for (int i = 0; i < num; ++i)
	{
		double cal_mass = 0;
		for (int j = 0; j < cells[i].size(); ++j)
		{
			BGAL::_Triangle3 tri(RPD.vertex_(std::get<0>(cells[i][j])), RPD.vertex_(std::get<1>(cells[i][j])), RPD.vertex_(std::get<2>(cells[i][j])));
			cal_mass += tri.area_();
		}
		std::cout << cal_mass << "\t" << capacity[i] << "\t" << cal_mass - capacity[i] << std::endl;
	}
	std::ofstream out("..\\..\\data\\" + modelName + "CPD_v=" + to_string(num) + ".obj");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RPD.number_vertices_(); ++i)
	{
		out << "v " << RPD.vertex_(i) << std::endl;
	}
	for (int i = 0; i < cells.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		for (int j = 0; j < cells[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
		}
	}
	out.close();
}

void CVT3DTest()
{
	string modelName = "bunny";
	BGAL::_ManifoldModel model("..\\..\\data\\" + modelName + ".obj");
	
	std::function<double(BGAL::_Point3& p)> rho = [](BGAL::_Point3& p)
	{
		return 1;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 1e-4;
	BGAL::_CVT3D cvt(model, rho, para);
	int num = 15000;
	double TotArea = 0;
	for (int i = 0; i < model.number_faces_(); i++)
	{
		model.vertex_(model.face_(i)[0]);
		
		double side[3];//存储三条边的长度;

		auto a = model.vertex_(model.face_(i)[0]);
		auto b = model.vertex_(model.face_(i)[1]);
		auto c = model.vertex_(model.face_(i)[2]);

		side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
		side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
		side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
		double p = (side[0] + side[1] + side[2]) / 2;
		double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
		TotArea += area;
	}
	std::cout << "TotArea = " << TotArea << std::endl;

	cvt.calculate_(num);
	const std::vector<BGAL::_Point3>& sites = cvt.get_sites();
	const BGAL::_Restricted_Tessellation3D& RVD = cvt.get_RVD();
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RVD.get_cells_();
	
	
	auto Vs = RVD.get_sites_();
	auto Edges = RVD.get_edges_();
	set<pair<int, int>> RDT_Edges;
	vector<set<int>> neibors;
	neibors.resize(Vs.size());
	for (int i = 0; i < Edges.size(); i++)
	{
		for (auto ee : Edges[i])
		{
			RDT_Edges.insert(make_pair(min(i, ee.first), max(i, ee.first)));
			neibors[i].insert(ee.first);
			neibors[ee.first].insert(i);
			//cout << ee.first << endl;

		}
	}
	std::ofstream outRDT("..\\..\\data\\" + modelName + "RDT_v=" + to_string(num) + ".obj");

	for (auto v : Vs)
	{
		outRDT << "v " << v << endl;
	}
	/*for (auto e : RDT_Edges)
	{
		outRDT << "l " << e.first+1 <<" "<< e.second+1  << endl;
	}*/

	set<MyFace> rdtFaces;

	for (auto e : RDT_Edges)
	{
		for (int pid : neibors[e.first])
		{
			if (RDT_Edges.find(make_pair(min(pid, e.first), max(pid, e.first))) != RDT_Edges.end())
			{
				if (RDT_Edges.find(make_pair(min(pid, e.second), max(pid, e.second))) != RDT_Edges.end())
				{
					int f1 = pid, f2 = e.first, f3 = e.second;

					int mid;
					if (f1 != max(f1, max(f2, f3)) && f1 != min(f1, min(f2, f3)))
					{
						mid = f1;
					}
					if (f2 != max(f1, max(f2, f3)) && f2 != min(f1, min(f2, f3)))
					{
						mid = f2;
					}
					if (f3 != max(f1, max(f2, f3)) && f3 != min(f1, min(f2, f3)))
					{
						mid = f3;
					}
					rdtFaces.insert(MyFace(max(f1, max(f2, f3)), mid, min(f1, min(f2, f3))));
				}
			}
		}
	}
	for (auto f : rdtFaces)
	{
		outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << endl;
	}

	outRDT.close();



	double AreaDiff = 0;

	std::ofstream out("..\\..\\data\\" + modelName + "CVT_v=" + to_string(num) + ".obj");
	std::ofstream outsites("..\\..\\data\\" + modelName + "Sites_CVT_v=" + to_string(num) + ".txt");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RVD.number_vertices_(); ++i)
	{
		out << "v " << RVD.vertex_(i) << std::endl;
		
	}
	for (int i = 0; i < num; i++)
	{
		outsites << sites[i] << endl;
	}
	double TotArea1 = 0;
	for (int i = 0; i < cells.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		double cellarea = 0;
		for (int j = 0; j < cells[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;

			double side[3];//存储三条边的长度;

			auto a = RVD.vertex_(std::get<0>(cells[i][j]));
			auto b = RVD.vertex_(std::get<1>(cells[i][j]));
			auto c = RVD.vertex_(std::get<2>(cells[i][j]));
			
			side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
			side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
			side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
			double p = (side[0] + side[1] + side[2]) / 2;
			double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
			//cout << area << endl;
			cellarea += area;
		}
		AreaDiff += (cellarea - TotArea / num) * (cellarea - TotArea / num);
		TotArea1 += cellarea;
	}
	
	std::cout	<< "TotArea1 = " << TotArea1 << std::endl;
	std::cout << "AreaDiff = " << AreaDiff << std::endl;
	out.close();
	outsites.close();
	//CVTCPDTest(modelName, sites);

	

	//CapVT
	std::vector<BGAL::_Point3> sts = sites;

	cvt.calculate_CapVT(sts);
	const std::vector<BGAL::_Point3>& sites_cap = cvt.get_sites();
	const BGAL::_Restricted_Tessellation3D& RVD_cap = cvt.get_RVD();
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells_cap = RVD_cap.get_cells_();


	out.open("..\\..\\data\\" + modelName + "CapCVT_v=" + to_string(num) + ".obj");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RVD_cap.number_vertices_(); ++i)
	{
		out << "v " << RVD_cap.vertex_(i) << std::endl;
	}
	TotArea1 = 0; AreaDiff = 0;
	for (int i = 0; i < cells_cap.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		double cellarea = 0;
		for (int j = 0; j < cells_cap[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells_cap[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells_cap[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells_cap[i][j]) + 1 << "/" << i + 1 << std::endl;

			double side[3];//存储三条边的长度;

			auto a = RVD_cap.vertex_(std::get<0>(cells_cap[i][j]));
			auto b = RVD_cap.vertex_(std::get<1>(cells_cap[i][j]));
			auto c = RVD_cap.vertex_(std::get<2>(cells_cap[i][j]));

			side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
			side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
			side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
			double p = (side[0] + side[1] + side[2]) / 2;
			double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
			//cout << area << endl;
			cellarea += area;
		}
		//cout << cellarea << endl;
		AreaDiff += (cellarea - TotArea / num) * (cellarea - TotArea / num);
		TotArea1 += cellarea;
	}

	std::cout << "TotArea1 = " << TotArea1 << std::endl;
	std::cout << "AreaDiff = " << AreaDiff << std::endl;
	out.close();


	Vs = RVD_cap.get_sites_();
	Edges = RVD_cap.get_edges_();
	RDT_Edges.clear();
	neibors.clear();
	neibors.resize(Vs.size());
	for (int i = 0; i < Edges.size(); i++)
	{
		for (auto ee : Edges[i])
		{
			RDT_Edges.insert(make_pair(min(i, ee.first), max(i, ee.first)));
			neibors[i].insert(ee.first);
			neibors[ee.first].insert(i);
			//cout << ee.first << endl;

		}
	}
	outRDT.open("..\\..\\data\\" + modelName + "CapRDT_v=" + to_string(num) + ".obj");

	for (auto v : Vs)
	{
		outRDT << "v " << v << endl;
	}
	/*for (auto e : RDT_Edges)
	{
		outRDT << "l " << e.first+1 <<" "<< e.second+1  << endl;
	}*/

	rdtFaces.clear();

	for (auto e : RDT_Edges)
	{
		for (int pid : neibors[e.first])
		{
			if (RDT_Edges.find(make_pair(min(pid, e.first), max(pid, e.first))) != RDT_Edges.end())
			{
				if (RDT_Edges.find(make_pair(min(pid, e.second), max(pid, e.second))) != RDT_Edges.end())
				{
					int f1 = pid, f2 = e.first, f3 = e.second;

					int mid;
					if (f1 != max(f1, max(f2, f3)) && f1 != min(f1, min(f2, f3)))
					{
						mid = f1;
					}
					if (f2 != max(f1, max(f2, f3)) && f2 != min(f1, min(f2, f3)))
					{
						mid = f2;
					}
					if (f3 != max(f1, max(f2, f3)) && f3 != min(f1, min(f2, f3)))
					{
						mid = f3;
					}
					rdtFaces.insert(MyFace(max(f1, max(f2, f3)), mid, min(f1, min(f2, f3))));
				}
			}
		}
	}
	for (auto f : rdtFaces)
	{
		outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << endl;
	}

	outRDT.close();



}
void CapVT3DTest()
{
	
	string modelName = "bunny";
	BGAL::_ManifoldModel model("..\\..\\data\\" + modelName + ".obj");

	std::function<double(BGAL::_Point3& p)> rho = [](BGAL::_Point3& p)
	{
		return 1;
	};
	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 1e-4;
	para.max_linearsearch = 20;
	BGAL::_CVT3D cvt(model, rho, para);
	int num = 8000;
	double TotArea = 0;
	//cvt.calculate_(num);
	for (int i = 0; i < model.number_faces_(); i++)
	{
		model.vertex_(model.face_(i)[0]);

		double side[3];//存储三条边的长度;

		auto a = model.vertex_(model.face_(i)[0]);
		auto b = model.vertex_(model.face_(i)[1]);
		auto c = model.vertex_(model.face_(i)[2]);

		side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
		side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
		side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
		double p = (side[0] + side[1] + side[2]) / 2;
		double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
		TotArea += area;
	}
	std::cout << "TotArea = " << TotArea << std::endl;

	
	std::vector<BGAL::_Point3> sts;

	std::ifstream inn("..\\..\\data\\" + modelName + "Sites_CVT_v=" + to_string(num) + ".txt");
	//cout << "..\\..\\data\\" + modelName + "Sites_CVT_v=" + to_string(num) + ".txt" << endl;
	double x, y, z;
	while (inn >> x >> y >> z)
	{
		sts.push_back(BGAL::_Point3(x, y, z));
	}
	cout << sts.size();
	// error here.
	cvt.calculate_CapVT(sts);
	const std::vector<BGAL::_Point3>& sites_cap = cvt.get_sites();
	const BGAL::_Restricted_Tessellation3D& RVD_cap = cvt.get_RVD();
	const std::vector<std::vector<std::tuple<int, int, int>>>& cells_cap = RVD_cap.get_cells_();


	ofstream out("..\\..\\data\\" + modelName + "CapCVT_v=" + to_string(num) + ".obj");
	out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
	for (int i = 0; i < RVD_cap.number_vertices_(); ++i)
	{
		out << "v " << RVD_cap.vertex_(i) << std::endl;
	}
	double TotArea1 = 0,AreaDiff = 0;
	for (int i = 0; i < cells_cap.size(); ++i)
	{
		double color = (double)BGAL::_BOC::rand_();
		out << "vt " << color << " 0" << std::endl;
		double cellarea = 0;
		for (int j = 0; j < cells_cap[i].size(); ++j)
		{
			out << "f " << std::get<0>(cells_cap[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<1>(cells_cap[i][j]) + 1 << "/" << i + 1
				<< " " << std::get<2>(cells_cap[i][j]) + 1 << "/" << i + 1 << std::endl;

			double side[3];//存储三条边的长度;

			auto a = RVD_cap.vertex_(std::get<0>(cells_cap[i][j]));
			auto b = RVD_cap.vertex_(std::get<1>(cells_cap[i][j]));
			auto c = RVD_cap.vertex_(std::get<2>(cells_cap[i][j]));

			side[0] = sqrt(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
			side[1] = sqrt(pow(a.x() - c.x(), 2) + pow(a.y() - c.y(), 2) + pow(a.z() - c.z(), 2));
			side[2] = sqrt(pow(c.x() - b.x(), 2) + pow(c.y() - b.y(), 2) + pow(c.z() - b.z(), 2));
			double p = (side[0] + side[1] + side[2]) / 2;
			double area = sqrt(p * (p - side[0]) * (p - side[1]) * (p - side[2]));
			//cout << area << endl;
			cellarea += area;
		}
		//cout << cellarea << endl;
		AreaDiff += (cellarea - TotArea / num) * (cellarea - TotArea / num);
		TotArea1 += cellarea;
	}

	std::cout << "TotArea1 = " << TotArea1 << std::endl;
	std::cout << "AreaDiff = " << AreaDiff << std::endl;
	out.close();


	auto Vs = RVD_cap.get_sites_();
	auto Edges = RVD_cap.get_edges_();
	set<pair<int, int>> RDT_Edges;
	vector<set<int>> neibors;
	neibors.resize(Vs.size());
	for (int i = 0; i < Edges.size(); i++)
	{
		for (auto ee : Edges[i])
		{
			RDT_Edges.insert(make_pair(min(i, ee.first), max(i, ee.first)));
			neibors[i].insert(ee.first);
			neibors[ee.first].insert(i);
			//cout << ee.first << endl;

		}
	}
	std::ofstream outRDT("..\\..\\data\\" + modelName + "CapRDT_v=" + to_string(num) + ".obj");

	for (auto v : Vs)
	{
		outRDT << "v " << v << endl;
	}
	/*for (auto e : RDT_Edges)
	{
		outRDT << "l " << e.first+1 <<" "<< e.second+1  << endl;
	}*/

	set<MyFace> rdtFaces;

	for (auto e : RDT_Edges)
	{
		for (int pid : neibors[e.first])
		{
			if (RDT_Edges.find(make_pair(min(pid, e.first), max(pid, e.first))) != RDT_Edges.end())
			{
				if (RDT_Edges.find(make_pair(min(pid, e.second), max(pid, e.second))) != RDT_Edges.end())
				{
					int f1 = pid, f2 = e.first, f3 = e.second;

					int mid;
					if (f1 != max(f1, max(f2, f3)) && f1 != min(f1, min(f2, f3)))
					{
						mid = f1;
					}
					if (f2 != max(f1, max(f2, f3)) && f2 != min(f1, min(f2, f3)))
					{
						mid = f2;
					}
					if (f3 != max(f1, max(f2, f3)) && f3 != min(f1, min(f2, f3)))
					{
						mid = f3;
					}
					rdtFaces.insert(MyFace(max(f1, max(f2, f3)), mid, min(f1, min(f2, f3))));
				}
			}
		}
	}
	for (auto f : rdtFaces)
	{
		outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << endl;
	}

	outRDT.close();


}



pair<double, double>  V3toV2(Eigen::Vector3d nor)
{
	double u = 0.0, v = 0.0;
	u = acos(nor.z());
	if (u == 0)
	{
		v = 0;
	}
	else
	{
		double tmp1 = abs(acos(nor.x() / sin(acos(nor.z()))));
		double tmp2 = abs(asin(nor.y() / sin(acos(nor.z()))));
		if (isnan(tmp1))
		{
			if (isnan(tmp2))
			{
				v = 0.0;
				pair<double, double> p(u, v);
				return p;
			}
			else
			{
				v = tmp2;
				pair<double, double> p(u, v);
				return p;
			}

		}
		if (isnan(tmp2))
		{
			if (isnan(tmp1))
			{
				v = 0.0;
				pair<double, double> p(u, v);
				return p;
			}
			else
			{
				v = tmp1;
				pair<double, double> p(u, v);
				return p;
			}
		}
		Eigen::Vector3d n11(sin(u) * cos(tmp1), sin(u) * sin(tmp1), cos(u));
		Eigen::Vector3d n22(sin(u) * cos(tmp2), sin(u) * sin(tmp2), cos(u));
		double tot1, tot2;
		tot1 = (n11 - nor).x() * (n11 - nor).x() + (n11 - nor).y() * (n11 - nor).y() + (n11 - nor).z() * (n11 - nor).z();
		tot2 = (n22 - nor).x() * (n22 - nor).x() + (n22 - nor).y() * (n22 - nor).y() + (n22 - nor).z() * (n22 - nor).z();
		if (tot1 < tot2)
		{
			v = tmp1;
		}
		else
		{
			v = tmp2;
		}
		if (abs(sin(u) * cos(v) - nor.x()) > 0.1)
		{
			u = -1.0 * u;
		}
		if (abs(sin(u) * sin(v) - nor.y()) > 0.1)
		{
			v = -1.0 * v;
		}

	}
	pair<double, double> p(u, v);
	return p;
}




void RFEPSTest(string model)
{
	clock_t start, end;
	vector<Eigen::Vector3d> Vall, Nall;
	vector<vector<int>> neighboor;

	bool debugOutput = 0, IfoutputFile =1;

	ofstream out;
	//alglib::setglobalthreading(alglib::parallel);

	//string model = "angleWithNor";
	//string model = "0.005_50000_00040123_8fc7d06caf264003a242597a_trimesh_000";
	//string modelnn = "DenoisePoints";

	string outputFile = "Results_noise0.0025";
	string outputPath = "E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\";

	ifstream in("E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\"+ outputFile +"\\" +model+"\\DenoisePoints.xyz");
	//ifstream in("E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\" + outputFile + "\\" + model + "\\"+model+".xyz");

	while (!in.eof())
	{
		Eigen::Vector3d p, n;
		in >> p.x() >> p.y() >> p.z() >> n.x() >> n.y() >> n.z();
		Vall.push_back(p);
		Nall.push_back(n);
	}
	cout << "Read PointCloud. xyz File. \n";
	int r = Vall.size();
	Eigen::Vector3d maxp(-99999, -99999, -99999);
	Eigen::Vector3d minp(99999, 99999, 99999);

	//for (int i = 0; i < r; i++)
	//{
	//	maxp.x() = max(maxp.x(), Vall[i].x());
	//	maxp.y() = max(maxp.y(), Vall[i].y());
	//	maxp.z() = max(maxp.z(), Vall[i].z());
	//	minp.x() = min(minp.x(), Vall[i].x());
	//	minp.y() = min(minp.y(), Vall[i].y());
	//	minp.z() = min(minp.z(), Vall[i].z());
	//	Nall[i].normalize();
	//}
	//double minn = min(minp.x(), min(minp.y(), minp.z()));
	//double maxn = max(maxp.x(), max(maxp.y(), maxp.z()));
	//double maxl = max(maxp.x() - minp.x(), max(maxp.y() - minp.y(), maxp.z() - minp.z()));
	//for (int i = 0; i < r; i++)
	//{
	//	Vall[i].x() = (Vall[i].x() - minp.x()) / (maxl);
	//	Vall[i].y() = (Vall[i].y() - minp.y()) / (maxl);
	//	Vall[i].z() = (Vall[i].z() - minp.z()) / (maxl);
	//} //
	//out.open("01_" + model + ".xyz");
	//for (size_t i = 0; i < r; i++)
	//{
	//	out << Vall[i].transpose() << " " << Nall[i].transpose() << endl;
	//}
	//out.close();
	//return;
	

	double radis = 0.001;
	double lambda = 0.05;

	// rnn

	PointCloud<double> cloud;
	int mink = 9999999, maxk = -9999999;

	cloud.pts.resize(r);
	for (int i = 0; i < r; i++)
	{
		cloud.pts[i].x = Vall[i].x();
		cloud.pts[i].y = Vall[i].y();
		cloud.pts[i].z = Vall[i].z();
	}
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t   index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	do
	{
		radis += 0.001;
		neighboor.clear();
		mink = 99999; maxk = -99999;
		for (int i = 0; i < r; i++)
		{
			vector<int> tmp;
			neighboor.push_back(tmp);
			double query_pt[3] = { Vall[i].x(), Vall[i].y(), Vall[i].z() };
			const double search_radius = static_cast<double>((radis) * (radis));
			std::vector<std::pair<uint32_t, double> >   ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
			for (size_t j = 0; j < nMatches; j++)
			{
				neighboor[i].push_back(ret_matches[j].first);
			}
			maxk = max(maxk, int(nMatches));
			mink = min(mink, int(nMatches));
		}
		mink = 99999; maxk = -99999;
		auto neighboor_denoise = neighboor;
		for (int i = 0; i < r; i++)
		{
			int k = neighboor[i].size();
			neighboor_denoise[i].clear();
			for (int j = 0; j < k; j++)
			{
				int pid = neighboor[i][j];
				double dis = (Nall[i] - Nall[pid]).norm();
				//if (dis < 1.5)
				{
					neighboor_denoise[i].push_back(pid);
				}
			}
			maxk = max(maxk, int(neighboor_denoise[i].size()));
			mink = min(mink, int(neighboor_denoise[i].size()));
		}
		neighboor = neighboor_denoise;
		
		cout << "maxk: " << maxk << "   mink:" << mink << endl;
	}  while (maxk < rnnnum);

	// add par later

	start = clock();

	map<int, double> R2, R3, S;
	for (int iter = 0; iter < r; iter++)
	{
		R2[iter] = 0.0; R3[iter] = 0; S[iter] = 0;
	}

	int omp_cnt = 0;
	omp_set_num_threads(24);
#pragma omp parallel for schedule(dynamic, 20)  //part 1
	for (int iter = 0; iter < r; iter++) // part 1
	{

		omp_cnt++;
		std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)> fop_lambda
			= [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
		{
			int k = neighboor[iter].size();
			int r = Vall.size();
			vector<double > w;
			for (int i = 0; i < k; i++)
			{
				w.push_back(0);
			}
			double maxw = -99999999.0;
			for (int i = 0; i < k; i++)
			{
				double dis = (Vall[iter] - Vall[neighboor[iter][i]]).norm();
				if (dis < 0.0001)
				{
					w[i] = 0.0;
				}
				else
				{
					w[i] = 1.0 / (dis * dis);
					maxw = max(maxw, w[i]);
				}
			}
			for (int i = 0; i < k; i++)
			{
				w[i] = 1.0;
				//w[i] = w[i] / maxw;
			}

			double u1 = x[k], v1 = x[k + 1], u2 = x[k + 2], v2 = x[k + 3];
			Eigen::Vector3d n1(sin(u1) * cos(v1), sin(u1) * sin(v1), cos(u1));
			Eigen::Vector3d n2(sin(u2) * cos(v2), sin(u2) * sin(v2), cos(u2));
			n1.normalize();
			n2.normalize();
			func = 0;
			for (int i = 0; i < k; i++)
			{
				auto Nj = Nall[neighboor[iter][i]];
				func += x[i] * w[i] * (pow((Nj.x() - sin(u1) * cos(v1)), 2) + pow(Nj.y() - sin(u1) * sin(v1), 2) + pow(Nj.z() - cos(u1), 2))
					+ (1.0 - x[i]) * w[i] * (pow((Nj.x() - sin(u2) * cos(v2)), 2) + pow(Nj.y() - sin(u2) * sin(v2), 2) + pow(Nj.z() - cos(u2), 2));
				//func += x[i] * w[i] * (Nall[neighboor[iter][i]] - n1).squaredNorm()  + (1 - x[i]) * w[i] * (Nall[neighboor[iter][i]] - n2).squaredNorm();
			}

			for (int i = 0; i < k; i++)
			{
				//double a = abs(cos(x[k]))* abs(cos(x[k]));
				auto Nj = Nall[neighboor[iter][i]];
				grad[i] = w[i] * (pow((Nj.x() - sin(u1) * cos(v1)), 2) + pow(Nj.y() - sin(u1) * sin(v1), 2) + pow(Nj.z() - cos(u1), 2))
					- w[i] * (pow((Nj.x() - sin(u2) * cos(v2)), 2) + pow(Nj.y() - sin(u2) * sin(v2), 2) + pow(Nj.z() - cos(u2), 2));
			}

			grad[k] = 0; grad[k + 1] = 0; grad[k + 2] = 0; grad[k + 3] = 0;

			for (int i = 0; i < k; i++)
			{
				auto Nj = Nall[neighboor[iter][i]];
				grad[k] += 2 * x[i] * sin(u1) * w[i] * (Nj.z() - cos(u1))
					- (2 * x[i] * cos(v1) * cos(u1) * w[i] * (Nj.x() - sin(u1) * cos(v1))
						+ 2 * x[i] * cos(u1) * w[i] * sin(v1) * (Nj.y() - sin(u1) * sin(v1)));

				grad[k + 1] += 2 * x[i] * sin(u1) * sin(v1) * w[i] * (Nj.x() - sin(u1) * cos(v1))
					- 2 * x[i] * sin(u1) * cos(v1) * w[i] * (Nj.y() - sin(u1) * sin(v1));

				grad[k + 2] += 2 * w[i] * sin(u2) * (1 - x[i]) * (Nj.z() - cos(u2))
					- (2 * w[i] * cos(u2) * cos(v2) * (1 - x[i]) * (Nj.x() - sin(u2) * cos(v2))
						+ 2 * w[i] * cos(u2) * sin(v2) * (1 - x[i]) * (Nj.y() - sin(u2) * sin(v2)));

				grad[k + 3] += 2 * sin(u2) * sin(v2) * (1 - x[i]) * w[i] * (Nj.x() - sin(u2) * cos(v2))
					- 2 * sin(u2) * cos(v2) * (1 - x[i]) * w[i] * (Nj.y() - sin(u2) * sin(v2));
			}
		};


		if (omp_cnt % 10000 == 0)
		{
			cout << "Part 1 --- Point iter: " << omp_cnt << " \n";
		}


		int k = neighboor[iter].size();
		if (k < 5)
		{
			continue;
		}
		// kmeans

		clusterizerstate s;
		kmeansreport rep;
		//real_2d_array xy = "[[1,1],[1,2],[4,1],[2,3],[4,1.5]]";
		real_2d_array xy;
		xy.setlength(k, 3);
		for (int i = 0; i < k; i++)
		{
			xy[i][0] = Nall[neighboor[iter][i]].x();
			xy[i][1] = Nall[neighboor[iter][i]].y();
			xy[i][2] = Nall[neighboor[iter][i]].z();
			if (debugOutput)
				cout << xy[i][0] << " " << xy[i][1] << " " << xy[i][2] << endl;
		}

		alglib::clusterizercreate(s);
		alglib::clusterizersetpoints(s, xy, 2);
		alglib::clusterizersetkmeanslimits(s, 10, 0);
		alglib::clusterizerrunkmeans(s, 2, rep);// ?
		if (debugOutput)
			printf("%d\n", int(rep.terminationtype)); // EXPECTED: 1
		Eigen::Vector3d C1, C2;
		if (int(rep.terminationtype) == -3)
		{
			C1 = Nall[neighboor[iter][0]];
			C2 = Nall[neighboor[iter][0]];
			//其实可以直接结束
		}
		else
		{
			C1.x() = rep.c[0][0];
			C1.y() = rep.c[0][1];
			C1.z() = rep.c[0][2];
			C2.x() = rep.c[1][0];
			C2.y() = rep.c[1][1];
			C2.z() = rep.c[1][2];

		}
		if (debugOutput)
		{
			cout << rep.c[0][0] << " " << rep.c[0][1] << " " << rep.c[0][2] << endl;
			cout << rep.c[1][0] << " " << rep.c[1][1] << " " << rep.c[1][2] << endl;
		}


		real_1d_array x0;
		x0.setlength(k + 4);
		real_1d_array s0;
		s0.setlength(k + 4);
		for (int i = 0; i < k; i++)
		{
			x0[i] = 0.5;
			s0[i] = 1;
		}
		for (int i = k; i < k + 4; i++)
		{
			x0[i] = 0;
			s0[i] = 1;
		}

		C2.normalize();
		C1.normalize();

		auto Q = V3toV2(C1);
		x0[k] = Q.first;
		x0[k + 1] = Q.second;
		Q = V3toV2(C2);
		x0[k + 2] = Q.first;
		x0[k + 3] = Q.second;
		//kmeans[iter] = make_pair(make_pair(x0[k], x0[k + 1]), make_pair(x0[k + 2], x0[k + 3]));

		real_1d_array bndl;
		real_1d_array bndu;
		bndl.setlength(k + 4);
		bndu.setlength(k + 4);

		real_2d_array c;
		c.setlength(1, neighboor[iter].size() + 5);
		integer_1d_array ct = "[0]";
		for (int i = 0; i < k; i++)
		{
			bndl[i] = 0;
			bndu[i] = 1;
			c[0][i] = 1;
		}
		for (int i = k; i < k + 4; i++)
		{
			bndl[i] = -99999999999;
			bndu[i] = 99999999999;
			c[0][i] = 0;
		}
		c[0][k + 4] = double(k) / 2.0;
		minbleicstate state;
		double epsg = 0;
		double epsf = 0;
		double epsx = 0;
		ae_int_t maxits = 0.000001;
		alglib::minbleiccreate(x0, state);
		alglib::minbleicsetlc(state, c, ct);
		alglib::minbleicsetbc(state, bndl, bndu);
		alglib::minbleicsetscale(state, s0);
		alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);

		alglib::minbleicoptguardsmoothness(state);
		alglib::minbleicoptguardgradient(state, 0.0001);

		if (debugOutput)
		{
			for (int i = 0; i < k + 4; i++)
			{
				cout << x0[i] << " ";
			}cout << endl;
		}



		minbleicreport rep2;
		//minbleicoptimize(state, fop, nullptr, nullptr, alglib::parallel);
		alglib::minbleicoptimize(state, fop_lambda);
		alglib::minbleicresults(state, x0, rep2);
		//cout << rep2.debugff << endl;
		double mn = 0;
		real_1d_array G_tmp;
		G_tmp.setlength(k + 4);
		fop_lambda(x0, mn, G_tmp, nullptr);
		if (debugOutput)
			cout << mn << endl;
		if (debugOutput)
		{
			printf("%d\n", int(rep2.terminationtype)); // EXPECTED: 4
			printf("%s\n", x0.tostring(2).c_str()); // EXPECTED: [2,4]

			optguardreport ogrep;
			minbleicoptguardresults(state, ogrep);
			printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
			printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
			printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false
		}


		double u1 = x0[k], v1 = x0[k + 1], u2 = x0[k + 2], v2 = x0[k + 3];
		Eigen::Vector3d n1(sin(u1) * cos(v1), sin(u1) * sin(v1), cos(u1));
		Eigen::Vector3d n2(sin(u2) * cos(v2), sin(u2) * sin(v2), cos(u2));
		/*n1.normalize();
		n2.normalize();*/
		if (debugOutput)
		{
			cout << n1 << endl;
			cout << n2 << endl;
		}


		double dis = sqrt((n1.x() - n2.x()) * (n1.x() - n2.x()) + (n1.y() - n2.y()) * (n1.y() - n2.y()) + (n1.z() - n2.z()) * (n1.z() - n2.z()));
		if (debugOutput)
			cout << "dis " << dis << endl;
		if (dis < 30 / 100.0) // importent
		{
			R3[iter] = 1; // normal point
		}
		else
		{
			R3[iter] = 0;
		}
		if (mn < 0.25)
		{
			R2[iter] = 1;
		}
	}

	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T2 Running Time: " << endtime << endl;
	cout << "T2 Running Time: " << endtime * 1000 << " ms " << endl;
	

	if (IfoutputFile)
	{
		out.open(outputPath + outputFile + "\\" + model + "\\ShowColorR2.txt");
		for (int i = 0; i < r; i++)
		{
			out << Vall[i].x() << " " << Vall[i].y() << " " << Vall[i].z() << " " << R2[i] << " 0.1 0.1 \n";
		}
		out.close();
		out.open(outputPath + outputFile + "\\" + model + "\\ShowColorR3.txt");
		for (int i = 0; i < r; i++)
		{
			out << Vall[i].x() << " " << Vall[i].y() << " " << Vall[i].z() << " " << R3[i] << " 0.1 0.1 \n";
		}
		out.close();
	}

	//return;
	// part2  normal
	//KNNSC
	auto neighboor_M = neighboor;
	cout << "ASDA\n";
	for (int i = 0; i < r; i++)
	{
		int k = neighboor[i].size();
		if (k < 5)
		{
			continue;
		}

		bool flag = 0;
		if (R3[i] == 1)
		{
			flag = 1;
		}
		/*for (auto p : neighboor_M[i])
		{
			if (R2[p] == 1)
			{
				flag = 1;
				break;
			}
		}*/
		if (flag)
		{
			
			continue;
		}
		int pid = 1;

		
		k = neighboor_M[i].size();
		neighboor_M[i].clear();

		double query_pt[3] = { Vall[i].x(), Vall[i].y(), Vall[i].z() };
		const double search_radius = static_cast<double>((radis * 3) * (radis * 3));
		std::vector<std::pair<uint32_t, double> > ret_matches;
		nanoflann::SearchParams params;
		const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

		neighboor_M[i].push_back(i);
		for (size_t j = 0; j < nMatches; j++)
		{
			if (R3[ret_matches[j].first] == 1)
			{
				neighboor_M[i].push_back(ret_matches[j].first);
				pid++;
				if (pid == k)
				{
					break;
				}
			}
		}
		double tmpr = radis * 3;
		while (pid != k)
		{
			tmpr *= 4;
			pid = 1;
			neighboor_M[i].clear();
			const double search_radius = static_cast<double>((tmpr) * (tmpr));
			std::vector<std::pair<uint32_t, double> > ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

			neighboor_M[i].push_back(i);
			for (size_t j = 0; j < nMatches; j++)
			{
				if (R3[ret_matches[j].first] == 1)
				{
					neighboor_M[i].push_back(ret_matches[j].first);
					pid++;
					if (pid == k)
					{
						break;
					}
				}
			}
		}
	}
	cout << "ASDA\n";

	omp_cnt = 0;
	map<int, Eigen::Vector3d> Nall_new;
	for (int iter = 0; iter < r; iter++)
	{
		Nall_new[iter] = Nall[iter];
	}
	
	start = clock();
	debugOutput = 0;
#pragma omp parallel for schedule(dynamic, 20)  //part 2
	for (int iter = 0; iter < r; iter++) // part 2
	{
		omp_cnt++;

		if (omp_cnt % 10000 == 0)
			cout << "Part 2 --- Point iter: " << omp_cnt << " \n";

		int k = neighboor[iter].size();
		if (k < 5)
		{
			continue;
		}
		k = neighboor_M[iter].size();
		/*if (R2[iter] == 1 && R3[iter] == 1)
		{
			S[iter] = 1;
			continue;
		}*/

		if (1)
		{
			vector<int> v1, cnt;
			for (int i = 0; i < k; i++)
			{
				v1.push_back(i);
				cnt.push_back(0);
			}
			for (int i = 0; i < k - 1; i++)
			{
				for (int j = i + 1; j < k; j++)
				{
					double dij = (Nall[neighboor_M[iter][i]] - Nall[neighboor_M[iter][j]]).norm();
					if (dij < lambda)
					{
						v1[j] = v1[i];
					}
				}
			}
			for (int i = 0; i < k; i++)
			{
				cnt[v1[i]] = cnt[v1[i]] + 1;
			}
			int mx = -99999;
			int center = 0;
			for (int i = 0; i < k; i++)
			{
				if (cnt[i] > mx)
				{
					mx = cnt[i];
					center = i;
				}
			}
			Eigen::Vector3d nor(0, 0, 0);
			for (int i = 0; i < k; i++)
			{
				if (v1[i] == center)
				{
					nor = nor + Nall[neighboor_M[iter][i]];
				}
			}
			nor.x() /= double(mx);
			nor.y() /= double(mx);
			nor.z() /= double(mx);
			nor.normalize();
			S[iter] = 1;

			Nall_new[iter] = nor;
			continue;
		}

		// opt .  no use.


		clusterizerstate s;
		kmeansreport rep;
		//real_2d_array xy = "[[1,1],[1,2],[4,1],[2,3],[4,1.5]]";
		real_2d_array xy;
		xy.setlength(k, 3);
		for (int i = 0; i < k; i++)
		{
			xy[i][0] = Nall[neighboor_M[iter][i]].x();
			xy[i][1] = Nall[neighboor_M[iter][i]].y();
			xy[i][2] = Nall[neighboor_M[iter][i]].z();
			if (debugOutput)
				cout << xy[i][0] << " " << xy[i][1] << " " << xy[i][2] << endl;
		}

		alglib::clusterizercreate(s);
		alglib::clusterizersetpoints(s, xy, 2);
		alglib::clusterizersetkmeanslimits(s, 10, 0);
		alglib::clusterizerrunkmeans(s, 2, rep);// ?
		if (debugOutput)
			printf("%d\n", int(rep.terminationtype)); // EXPECTED: 1
		Eigen::Vector3d C1, C2;
		if (int(rep.terminationtype) == -3)
		{
			C1 = Nall[neighboor[iter][0]];
			C2 = Nall[neighboor[iter][0]];
			//其实可以直接结束
		}
		else
		{
			C1.x() = rep.c[0][0];
			C1.y() = rep.c[0][1];
			C1.z() = rep.c[0][2];
			C2.x() = rep.c[1][0];
			C2.y() = rep.c[1][1];
			C2.z() = rep.c[1][2];

		}
		if (debugOutput)
		{
			cout << rep.c[0][0] << " " << rep.c[0][1] << " " << rep.c[0][2] << endl;
			cout << rep.c[1][0] << " " << rep.c[1][1] << " " << rep.c[1][2] << endl;
		}


		real_1d_array x0;
		x0.setlength(k + 4);
		real_1d_array s0;
		s0.setlength(k + 4);
		for (int i = 0; i < k; i++)
		{
			x0[i] = 0.5;
			s0[i] = 1;
		}
		for (int i = k; i < k + 4; i++)
		{
			x0[i] = 0;
			s0[i] = 1;
		}

		C2.normalize();
		C1.normalize();

		auto Q = V3toV2(C1);
		x0[k] = Q.first;
		x0[k + 1] = Q.second;
		Q = V3toV2(C2);
		x0[k + 2] = Q.first;
		x0[k + 3] = Q.second;



		real_1d_array bndl;
		real_1d_array bndu;
		bndl.setlength(k + 4);
		bndu.setlength(k + 4);

		/*real_2d_array c;
		c.setlength(1, neighboor[iter].size() + 5);
		integer_1d_array ct = "[0]";*/
		for (int i = 0; i < k; i++)
		{
			bndl[i] = 0;
			bndu[i] = 1;
			//c[0][i] = 1;
		}
		for (int i = k; i < k + 4; i++)
		{
			bndl[i] = -9999999999;
			bndu[i] = 9999999999;
			//c[0][i] = 0;
		}
		//c[0][k + 4] = double(k) / 2.0;
		minbleicstate state;
		double epsg = 0;
		double epsf = 0;
		double epsx = 0.000001;
		ae_int_t maxits = 0;
		alglib::minbleiccreate(x0, state);
		//alglib::minbleicsetlc(state, c, ct);
		alglib::minbleicsetbc(state, bndl, bndu);
		alglib::minbleicsetscale(state, s0);
		alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);

		alglib::minbleicoptguardsmoothness(state);
		alglib::minbleicoptguardgradient(state, 0.0001);

		/*	for (int i = 0; i < k+4; i++)
			{
				cout << x0[i] << " ";
			}cout << endl;*/



		std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)> fop_w_lambda
			= [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
		{
			int k = neighboor_M[iter].size();
			int r = Vall.size();
			vector<double > w;
			for (int i = 0; i < k; i++)
			{
				w.push_back(R3[neighboor_M[iter][i]]);
			}
			double maxw = -99999999.0;
			for (int i = 0; i < k; i++)
			{
				double dis = (Vall[iter] - Vall[neighboor_M[iter][i]]).norm();
				if (dis < 0.0001)
				{
					w[i] = 0.0;
				}
				else
				{
					w[i] = 1.0 / (dis * dis);
					maxw = max(maxw, w[i]);
				}
			}
			for (int i = 0; i < k; i++)
			{
				//w[i] = 1.0;
				w[i] = w[i] / maxw;
			}

			double u1 = x[k], v1 = x[k + 1], u2 = x[k + 2], v2 = x[k + 3];
			Eigen::Vector3d n1(sin(u1) * cos(v1), sin(u1) * sin(v1), cos(u1));
			Eigen::Vector3d n2(sin(u2) * cos(v2), sin(u2) * sin(v2), cos(u2));
			n1.normalize();
			n2.normalize();
			func = 0;
			for (int i = 0; i < k; i++)
			{
				auto Nj = Nall[neighboor_M[iter][i]];
				func += x[i] * w[i] * (pow((Nj.x() - sin(u1) * cos(v1)), 2) + pow(Nj.y() - sin(u1) * sin(v1), 2) + pow(Nj.z() - cos(u1), 2))
					+ (1.0 - x[i]) * w[i] * (pow((Nj.x() - sin(u2) * cos(v2)), 2) + pow(Nj.y() - sin(u2) * sin(v2), 2) + pow(Nj.z() - cos(u2), 2));
				//func += x[i] * w[i] * (Nall[neighboor[iter][i]] - n1).squaredNorm()  + (1 - x[i]) * w[i] * (Nall[neighboor[iter][i]] - n2).squaredNorm();
			}

			for (int i = 0; i < k; i++)
			{
				//double a = abs(cos(x[k]))* abs(cos(x[k]));
				auto Nj = Nall[neighboor_M[iter][i]];
				grad[i] = w[i] * (pow((Nj.x() - sin(u1) * cos(v1)), 2) + pow(Nj.y() - sin(u1) * sin(v1), 2) + pow(Nj.z() - cos(u1), 2))
					- w[i] * (pow((Nj.x() - sin(u2) * cos(v2)), 2) + pow(Nj.y() - sin(u2) * sin(v2), 2) + pow(Nj.z() - cos(u2), 2));
			}

			grad[k] = 0; grad[k + 1] = 0; grad[k + 2] = 0; grad[k + 3] = 0;

			for (int i = 0; i < k; i++)
			{
				auto Nj = Nall[neighboor_M[iter][i]];
				grad[k] += 2 * x[i] * sin(u1) * w[i] * (Nj.z() - cos(u1))
					- (2 * x[i] * cos(v1) * cos(u1) * w[i] * (Nj.x() - sin(u1) * cos(v1))
						+ 2 * x[i] * cos(u1) * w[i] * sin(v1) * (Nj.y() - sin(u1) * sin(v1)));

				grad[k + 1] += 2 * x[i] * sin(u1) * sin(v1) * w[i] * (Nj.x() - sin(u1) * cos(v1))
					- 2 * x[i] * sin(u1) * cos(v1) * w[i] * (Nj.y() - sin(u1) * sin(v1));

				grad[k + 2] += 2 * w[i] * sin(u2) * (1 - x[i]) * (Nj.z() - cos(u2))
					- (2 * w[i] * cos(u2) * cos(v2) * (1 - x[i]) * (Nj.x() - sin(u2) * cos(v2))
						+ 2 * w[i] * cos(u2) * sin(v2) * (1 - x[i]) * (Nj.y() - sin(u2) * sin(v2)));

				grad[k + 3] += 2 * sin(u2) * sin(v2) * (1 - x[i]) * w[i] * (Nj.x() - sin(u2) * cos(v2))
					- 2 * sin(u2) * cos(v2) * (1 - x[i]) * w[i] * (Nj.y() - sin(u2) * sin(v2));
			}
		};


		minbleicreport rep2;
		//minbleicoptimize(state, fop, nullptr, nullptr, alglib::parallel);
		alglib::minbleicoptimize(state, fop_w_lambda);
		alglib::minbleicresults(state, x0, rep2);
		//cout << rep2.debugff << endl;
		//double mn = 0;
		//real_1d_array G_tmp;
		//G_tmp.setlength(k + 4);
		//fop_w_lambda(x0, mn, G_tmp, nullptr);
		//cout << mn << endl;
		//printf("%d\n", int(rep2.terminationtype)); // EXPECTED: 4
		//printf("%s\n", x0.tostring(2).c_str()); // EXPECTED: [2,4]

		//optguardreport ogrep;
		//minbleicoptguardresults(state, ogrep);
		//printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
		//printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
		//printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false

		double u1 = x0[k], v1 = x0[k + 1], u2 = x0[k + 2], v2 = x0[k + 3];
		Eigen::Vector3d n1(sin(u1) * cos(v1), sin(u1) * sin(v1), cos(u1));
		Eigen::Vector3d n2(sin(u2) * cos(v2), sin(u2) * sin(v2), cos(u2));
		n1.normalize();
		n2.normalize();
		//cout << n1 << endl;
		//cout << n2 << endl;

		double sum = 0.0;
		for (int i = 0; i < k; i++)
		{
			sum += x0[i];
		}
		if (sum < double(k) / 2.0)
		{
			Nall_new[iter] = n2;
		}
		else
		{
			Nall_new[iter] = n1;
		}
	}

	if (IfoutputFile)
	{
		out.open(outputPath + outputFile + "\\" + model + "\\ShowNormal.xyz");
		for (int i = 0; i < r; i++)
		{
			int k = neighboor[i].size();
			if (k < 5)
			{
				continue;
			}
			out << Vall[i].x() << " " << Vall[i].y() << " " << Vall[i].z() << " " << Nall_new[i].x() << " " << Nall_new[i].y() << " " << Nall_new[i].z() << "\n";
		}
		out.close();
	}

	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T3 Running Time: " << endtime << endl;
	cout << "T3 Running Time: " << endtime * 1000 << " ms " << endl;
	
	// part 2.5 denoise 

	
	auto neighboor_denoise = neighboor;

	std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)> fop_denoise
		= [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
	{
		func = 0.0;


		for (int i = 0; i < r; i++)
		{
			grad[i] = 0;
		}
		for (int i = 0; i < r; i++)
		{

			int k = neighboor_denoise[i].size();
			if (k < 5)
			{
				continue;
			}

			for (int j = 0; j < k; j++)
			{
				int pj = neighboor_denoise[i][j];
				Eigen::Vector3d Pj = Vall[pj] + x[pj] * Nall_new[pj];
				Eigen::Vector3d Pi = Vall[i] + x[i] * Nall_new[i];
				Eigen::Vector3d Vij = Pj - Pi;

				Eigen::Matrix3d a;
				a = Vij * Vij.transpose();
				Eigen::Vector3d ans = a * Nall_new[i];
				func += ans.norm();

				auto t0 = Vij;

				double t1 = t0.transpose() * Nall_new[i];
				double t2 = Nall_new[i].transpose() * t0;
				double t3 = t0.transpose() * t0;
				double t4 = Nall_new[i].transpose() * Nall_new[i];


				grad[i] += -1.0 * (2.0 * t1 * t1 * t2 + 2.0 * t1 * t3 * t4);
				double t5 = t0.transpose() * Nall_new[pj];
				double t6 = Nall_new[i].transpose() * Nall_new[pj];

				grad[pj] += 2.0 * t1 * t5 * t2 + 2.0 * t1 * t3 * t6;
			}
		}


		//return loss;
	};

	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fg
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		double func = 0.0;


		for (int i = 0; i < r; i++)
		{
			g(i) = 0;
		}
		for (int i = 0; i < r; i++)
		{

			int k = neighboor_denoise[i].size();
			if (k < 5)
			{
				continue;
			}

			for (int j = 0; j < k; j++)
			{
				int pj = neighboor_denoise[i][j];
				Eigen::Vector3d Pj = Vall[pj] + X(pj) * Nall_new[pj];
				Eigen::Vector3d Pi = Vall[i] + X(i) * Nall_new[i];
				Eigen::Vector3d Vij = Pj - Pi;

				Eigen::Matrix3d a;
				a = Vij * Vij.transpose();
				Eigen::Vector3d ans = a * Nall_new[i];
				func += ans.norm();

				auto t0 = Vij;

				double t1 = t0.transpose() * Nall_new[i];
				double t2 = Nall_new[i].transpose() * t0;
				double t3 = t0.transpose() * t0;
				double t4 = Nall_new[i].transpose() * Nall_new[i];


				g[i] += -1.0 * (2.0 * t1 * t1 * t2 + 2.0 * t1 * t3 * t4);
				double t5 = t0.transpose() * Nall_new[pj];
				double t6 = Nall_new[i].transpose() * Nall_new[pj];

				g[pj] += 2.0 * t1 * t5 * t2 + 2.0 * t1 * t3 * t6;
			}
		}
		return func;
	};

	for (int i = 0; i < r; i++)
	{
		int k = neighboor[i].size();
		if (k < 5)
		{
			continue;
		}
		neighboor_denoise[i].clear();
		for (int j = 0; j < k; j++)
		{
			int pid = neighboor[i][j];
			double dis = (Nall_new[i] - Nall_new[pid]).norm();
			if (dis < 0.2)
			{
				neighboor_denoise[i].push_back(pid);
			}
		}
	}


	


	//cout << "loss = " << fop_denoise() << endl;
	// opt.

	vector<Eigen::Vector3d> Vall_new = Vall;
	start = clock();
	// lbfgs test
	if(1)
	{
		
		BGAL::_LBFGS::_Parameter param = BGAL::_LBFGS::_Parameter();
		param.epsilon = 5e-5;
		param.is_show = true;
		param.max_iteration = 100;
		param.max_linearsearch = 20;
		BGAL::_LBFGS lbfgs(param);


		Eigen::VectorXd iterX(r);
		for (int i = 0; i < r; i++)
		{
			iterX(i) = 0;
		}
		int n = lbfgs.minimize(fg, iterX);
		//int a = 53;
		//int n = lbfgs.test(a);
		//std::cout << iterX << std::endl;
		std::cout << "n: " << n << std::endl;


		for (int i = 0; i < r; i++)
		{
			Vall_new[i] = Vall[i] + iterX(i) * Nall_new[i];
		}


	}
	
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T4 Running Time: " << endtime << endl;
	cout << "T4 Running Time: " << endtime * 1000 << " ms " << endl;

	// rnn

	PointCloud<double> cloud1;
	mink = 9999999; maxk = -9999999;

	cloud1.pts.resize(r);
	for (int i = 0; i < r; i++)
	{
		cloud1.pts[i].x = Vall_new[i].x();
		cloud1.pts[i].y = Vall_new[i].y();
		cloud1.pts[i].z = Vall_new[i].z();
	}

	my_kd_tree_t   index1(3 /*dim*/, cloud1, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index1.buildIndex();


	auto neighboor_qc = neighboor;
	neighboor_qc.clear();
		mink = 99999; maxk = -99999;
		for (int i = 0; i < r; i++)
		{
			
			
			vector<int> tmp;
			neighboor_qc.push_back(tmp);
			double query_pt[3] = { Vall_new[i].x(), Vall_new[i].y(), Vall_new[i].z() };
			const double search_radius = static_cast<double>((radis) * (radis));
			std::vector<std::pair<uint32_t, double> >   ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index1.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
			for (size_t j = 0; j < nMatches; j++)
			{
				neighboor_qc[i].push_back(ret_matches[j].first);
			}
			maxk = max(maxk, int(nMatches));
			mink = min(mink, int(nMatches));
		}
		cout << "maxk: " << maxk << "   mink:" << mink << endl;
	

	ofstream fout(outputPath + outputFile + "\\" + model + "\\DenoisePoints_Final.xyz");
	for (size_t i = 0; i < r; i++)
	{
		int k = neighboor_denoise[i].size();
		if (k < 5)
		{
			continue;
		}
	/*	k = neighboor_qc[i].size();
		if (k < 55555)
		{
			continue;
		}*/
		
		
		fout << Vall_new[i].transpose() << " " << Nall_new[i].transpose() << endl;
	}

	Vall = Vall_new;

	
	



	// part 3
	map<int, bool> flag3;

	for (int iter = 0; iter < r; iter++)
	{
		flag3[iter] = 0;
		if (R3[iter] == 1)
		{
			flag3[iter] = 1;
			//continue;
		}
	}
	for (int iter = 0; iter < r; iter++)
	{
		if (flag3[iter])
		{
			continue;
		}
		int k = neighboor[iter].size();
		if (k < 5)
		{
			continue;
		}
	
		double maxAng = -9999.0;
		for (int i = 0; i < k - 1; i++)
		{
			for (int j = i + 1; j < k; j++)
			{
				double sigma = acos(Nall_new[neighboor[iter][i]].dot(Nall_new[neighboor[iter][j]]) / (Nall_new[neighboor[iter][i]].norm() * Nall_new[neighboor[iter][j]].norm()));
				sigma = sigma / EIGEN_PI * 180.0;
				maxAng = max(maxAng, sigma);
			}
		}

		if (maxAng < 20)
		{
			flag3[iter] = 1;
			continue;
		}

	}
	bool iw = 0;
	/*while (!iw)
	{
		iw = 1;
		for (int iter = 0; iter < r; iter++)
		{
			if (flag3[iter])
			{
				continue;
			}
			int cnt = 0;
			for (auto p : neighboor[iter])
			{
				if (flag3[p] == 0)
				{
					cnt++;
				}
			}
			if (cnt <= 3)
			{
				flag3[iter] = 1;
				iw = 0;
			}
		}
	}*/
	omp_cnt = 0;
	start = clock();
	//debugOutput = 1;
	map<int, Eigen::Vector3d> NewPoints;
//#pragma omp parallel for schedule(dynamic, 20) //part 3
	for (int iter = 0; iter < r; iter++)
	{
		if (flag3[iter])
		{
			continue;
		}
		int k = neighboor[iter].size();
		if (k < 5)
		{
			continue;
		}
	


		omp_cnt++;

		if (omp_cnt % 1000 == 0)
			cout << "Part 3 --- Point iter: " << omp_cnt << " \n";

		real_1d_array x0;
		x0.setlength(3);
		x0[0] = Vall[iter].x();
		x0[1] = Vall[iter].y();
		x0[2] = Vall[iter].z();
		real_1d_array s0 = "[1,1,1]";



		std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)> fop_z_lambda
			= [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
		{
			int k = neighboor[iter].size();
			int r = Vall.size();


			Eigen::Vector3d z(x[0], x[1], x[2]);
			double mu = 0.01;
			func = 0;
			for (int i = 0; i < k; i++)
			{
				auto zp = Vall[neighboor[iter][i]] - z;
				func += pow(zp.dot(Nall_new[neighboor[iter][i]]), 2);
				//func += x[i] * w[i] * (Nall[neighboor[iter][i]] - n1).squaredNorm()  + (1 - x[i]) * w[i] * (Nall[neighboor[iter][i]] - n2).squaredNorm();
			}
			func = func + mu * (Vall[iter] - z).squaredNorm();

			Eigen::Vector3d g(0, 0, 0);
			g = 2 * mu * (z - Vall[iter]);
			for (int i = 0; i < k; i++)
			{
				//double a = abs(cos(x[k]))* abs(cos(x[k]));
				auto Pj = Vall[neighboor[iter][i]];
				auto nj = Nall_new[neighboor[iter][i]];
				g = g + 2 * (z - Pj).dot(nj) * nj;
			}

			grad[0] = g.x();
			grad[1] = g.y();
			grad[2] = g.z();


		};

		minbleicstate state;
		double epsg = 0;
		double epsf = 0;
		double epsx = 0;
		ae_int_t maxits = 0;
		alglib::minbleiccreate(x0, state);
		//alglib::minbleicsetlc(state, c, ct);
		//alglib::minbleicsetbc(state, bndl, bndu);
		alglib::minbleicsetscale(state, s0);
		alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);

		alglib::minbleicoptguardsmoothness(state);
		alglib::minbleicoptguardgradient(state, 0.000001);

		if (debugOutput)
		{
			for (int i = 0; i < 3; i++)
			{
				cout << x0[i] << " ";
			}cout << endl;
		}



		minbleicreport rep2;
		//minbleicoptimize(state, fop, nullptr, nullptr, alglib::parallel);
		alglib::minbleicoptimize(state, fop_z_lambda);
		alglib::minbleicresults(state, x0, rep2);
		//cout << rep2.debugff << endl;
		double mn = 0;
		real_1d_array G_tmp;
		G_tmp.setlength(3);
		fop_z_lambda(x0, mn, G_tmp, nullptr);
		if (debugOutput)
			cout << mn << endl;
		if (debugOutput)
		{
			printf("%d\n", int(rep2.terminationtype)); // EXPECTED: 4
			printf("%s\n", x0.tostring(2).c_str()); // EXPECTED: [2,4]

			optguardreport ogrep;
			minbleicoptguardresults(state, ogrep);
			printf("%s\n", ogrep.badgradsuspected ? "true" : "false"); // EXPECTED: false
			printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
			printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false
		}

		NewPoints[iter] = Eigen::Vector3d(x0[0], x0[1], x0[2]);
	}



	for (auto np : NewPoints)
	{
		if (flag3[np.first])
		{
			continue;
		}
		double dis = (Vall[np.first] - np.second).norm();
		if (dis > radis*2 || dis < 0.00001)
		{
			flag3[np.first] = 1;
		}

	}

	if (IfoutputFile) {
		/*out.open("FinalPointCloud.xyz");
		for (int i = 0; i < r; i++)
		{
			out << Vall[i].transpose() << endl;
		}

		for (auto np : NewPoints)
		{
			if (flag3[np.first])
			{
				continue;
			}
			out << np.second.transpose() << endl;
		}
		out.close();*/

		out.open(outputPath + outputFile + "\\" + model + "\\FeaturePoints_angle.xyz");
		for (auto np : NewPoints)
		{
			if (flag3[np.first])
			{
				continue;
			}
			out << np.second.transpose() << endl;
		}
		out.close();

		out.open(outputPath + outputFile + "\\" + model + "\\FinalPointCloud.xyz");
		vector<bool> flag2(r, 0);
		for (auto np : NewPoints)
		{
			if (flag3[np.first])
			{
				continue;
			}
			out << np.second.transpose() <<" "<< Nall_new[np.first].transpose() << endl;
			flag2[np.first] = 1;
		}
		for (size_t i = 0; i < r; i++)
		{
			int k = neighboor_denoise[i].size();
			if (k < 5)
			{
				continue;
			}
			
			if(!flag2[i])
			out << Vall[i].transpose() << " " << Nall_new[i].transpose() << endl;
		}

		/*for (size_t i = 0; i < r; i++)
		{
			int k = neighboor_denoise[i].size();
			if (k < 5)
			{
				continue;
			}
			out << Vall[i].transpose() << " " << Nall_new[i].transpose() << endl;
		}*/
		out.close();

		out.open(outputPath + outputFile + "\\" + model + "\\FinalPointCloud_forZX.xyz");
		vector<bool> flag22(r, 0);
		for (auto np : NewPoints)
		{
			if (flag3[np.first])
			{
				continue;
			}
			out << np.second.transpose() << " " << Nall_new[np.first].transpose() << endl;
			flag22[np.first] = 1;
		}
		for (size_t i = 0; i < r; i++)
		{
			int k = neighboor_denoise[i].size();
			if (k < 5)
			{
				continue;
			}

			//if (!flag2[i])
			out << Vall[i].transpose() << " " << Nall_new[i].transpose() << endl;
		}

		out.close();


		out.open(outputPath + outputFile + "\\" + model + "\\FinalPointCloud_NoFeature.xyz");
		
		for (auto np : NewPoints)
		{
			if (flag3[np.first])
			{
				continue;
			}
			//out << np.second.transpose() << " " << Nall_new[np.first].transpose() << endl;
			//flag2[np.first] = 1;
		}
		for (size_t i = 0; i < r; i++)
		{
			int k = neighboor_denoise[i].size();
			if (k < 5)
			{
				continue;
			}

			if(!flag2[i])
			out << Vall[i].transpose() << " " << Nall_new[i].transpose() << endl;
		}
		out.close();
		
		out.open(outputPath + outputFile + "\\" + model + "\\FeaturePointsNum.txt");
		int cnt = 0;
		for (auto np : NewPoints)
		{
			if (flag3[np.first])
			{
				continue;
			}
			cnt++;
			
		}
		out << cnt << endl;
		out.close();

		out.open(outputPath + outputFile + "\\" + model + "\\radis.txt");
		
		out << radis << endl;
		out.close();
		
	}


	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T5 Running Time: " << endtime << endl;
	cout << "T5 Running Time: " << endtime * 1000 << " ms " << endl;

}


void DenoiseTest(string model)
{
	clock_t start, end;

	bool debugOutput = 0, IfoutputFile = 1;
	vector<Eigen::Vector3d> Vall, Nall;
	vector<vector<int>> neighboor;

	ofstream out;
	//alglib::setglobalthreading(alglib::parallel);

	//string model = "angleWithNor";
	//string model = "0.005_50000_00040123_8fc7d06caf264003a242597a_trimesh_000";

	string outputFile = "Results_noise0.01";
	string outputPath = "E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\";

	ifstream in("E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\Noise\\abc_chunk4\\0.01\\" + model + ".xyz");

	while (!in.eof())
	{
		Eigen::Vector3d p, n;
		in >> p.x() >> p.y() >> p.z() >> n.x() >> n.y() >> n.z();
		Vall.push_back(p);
		Nall.push_back(n);
	}
	cout << "Read PointCloud. xyz File. \n";
	int r = Vall.size();
	Eigen::Vector3d maxp(-99999, -99999, -99999);
	Eigen::Vector3d minp(99999, 99999, 99999);

	for (int i = 0; i < r; i++)
	{
		maxp.x() = max(maxp.x(), Vall[i].x());
		maxp.y() = max(maxp.y(), Vall[i].y());
		maxp.z() = max(maxp.z(), Vall[i].z());
		minp.x() = min(minp.x(), Vall[i].x());
		minp.y() = min(minp.y(), Vall[i].y());
		minp.z() = min(minp.z(), Vall[i].z());
		Nall[i].normalize();
	}
	double minn = min(minp.x(), min(minp.y(), minp.z()));
	double maxn = max(maxp.x(), max(maxp.y(), maxp.z()));
	double maxl = max(maxp.x() - minp.x(), max(maxp.y() - minp.y(), maxp.z() - minp.z()));
	//for (int i = 0; i < r; i++)
	//{
	//	Vall[i].x() = (Vall[i].x() - minp.x()) / (maxl);
	//	Vall[i].y() = (Vall[i].y() - minp.y()) / (maxl);
	//	Vall[i].z() = (Vall[i].z() - minp.z()) / (maxl);
	//} //
	//out.open("01_" + model + ".xyz");
	//for (size_t i = 0; i < r; i++)
	//{
	//	out << Vall[i].transpose() << " " << Nall[i].transpose() << endl;
	//}
	//out.close();

	double radis = 0.001;
	double lambda = 0.05;

	// rnn

	PointCloud<double> cloud;
	int mink = 9999999, maxk = -9999999;

	cloud.pts.resize(r);
	for (int i = 0; i < r; i++)
	{
		cloud.pts[i].x = Vall[i].x();
		cloud.pts[i].y = Vall[i].y();
		cloud.pts[i].z = Vall[i].z();
	}
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t   index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	do
	{
		radis += 0.001;
		neighboor.clear();
		mink = 99999; maxk = -99999;
		for (int i = 0; i < r; i++)
		{
			vector<int> tmp;
			neighboor.push_back(tmp);
			double query_pt[3] = { Vall[i].x(), Vall[i].y(), Vall[i].z() };
			const double search_radius = static_cast<double>((radis) * (radis));
			std::vector<std::pair<uint32_t, double> >   ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
			for (size_t j = 0; j < nMatches; j++)
			{
				neighboor[i].push_back(ret_matches[j].first);
			}
			maxk = max(maxk, int(nMatches));
			mink = min(mink, int(nMatches));
		}
		cout << "maxk: " << maxk << "   mink:" << mink << endl;

	}  while (maxk < rnnnum);

	// smooth 
	//for (int i = 0; i < r; i++)
	//{
	//	int k = neighboor[i].size();
	//	if (k < 5)
	//	{
	//		continue;
	//	}
	//	
	//	Eigen::Vector3d sum(0, 0, 0);
	//	for (int j = 0; j < k; j++)
	//	{
	//		sum += Nall[neighboor[i][j]];
	//	}
	//	sum /= k;
	//	Nall[i] = sum;
	//}			


	
	// add par later

	//auto neighboor_denoise = neighboor;
	//for (int i = 0; i < r; i++)
	//{
	//	int k = neighboor[i].size();
	//	if (k < 5)
	//	{
	//		continue;
	//	}
	//	neighboor_denoise[i].clear();
	//	for (int j = 0; j < k; j++)
	//	{
	//		int pid = neighboor[i][j];
	//		double dis = (Nall[i] - Nall[pid]).norm();
	//		if (dis < 0.3)
	//		{
	//			neighboor_denoise[i].push_back(pid);
	//		}
	//	}
	//}
	//neighboor = neighboor_denoise;


	start = clock();


	auto Vall_new = Vall;
	auto Nall_new = Nall;

	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fg
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		double func = 0.0;
		double lambda = 0.01;

		for (int i = 0; i < r*3; i++)
		{
			g(i) = 0;
		}
		auto Nall_tmp = Nall_new;
		for (int i = 0; i < r; i++)
		{
			double u = X(r + i * 2);
			double v = X(r + i * 2 + 1);
			Nall_tmp[i].x() = sin(u) * cos(v);
			Nall_tmp[i].y() = sin(u) * sin(v);
			Nall_tmp[i].z() = cos(u);
		}
		
		for (int i = 0; i < r; i++)
		{
			

			int k = neighboor[i].size();
			if (k < 5)
			{
				continue;
			}

			for (int j = 0; j < k; j++)
			{
				int pj = neighboor[i][j];
				Eigen::Vector3d Pj = Vall[pj] + X(pj) * Nall_tmp[pj];
				Eigen::Vector3d Pi = Vall[i] + X(i) * Nall_tmp[i];
				Eigen::Vector3d Vij = Pj - Pi;

				Eigen::Matrix3d a;
				a = Vij * Vij.transpose();
				Eigen::Vector3d ans = a * Nall_tmp[i];
				func += ans.norm();

				auto t0 = Vij;

				double t1 = t0.transpose() * Nall_tmp[i];
				double t2 = Nall_tmp[i].transpose() * t0;
				double t3 = t0.transpose() * t0;
				double t4 = Nall_tmp[i].transpose() * Nall_tmp[i];


				g[i] += -1.0 * (2.0 * t1 * t1 * t2 + 2.0 * t1 * t3 * t4);
				double t5 = t0.transpose() * Nall_tmp[pj];
				double t6 = Nall_tmp[i].transpose() * Nall_tmp[pj];

				g[pj] += 2.0 * t1 * t5 * t2 + 2.0 * t1 * t3 * t6;

				// g of ni
				{
					Eigen::Vector3d t0 = Vij;
					Eigen::Vector3d t1 = t0.transpose() * Nall_tmp[i] * t0;
					double t2 = 2 * X(i);
					double t3 = t0.transpose() * t0;
					double t4 = Nall_tmp[i].transpose() * t0;
					Eigen::Vector3d gni(0, 0, 0);
					
					gni = 2.0 * t3 * t1 -(t2 * t4 * t1 + t2 * t3 * t4 * Nall_tmp[i]);
					
					g[r + i * 2] += gni.x() * cos(X(r + i * 2)) * cos(X(r + i * 2 + 1))  + gni.y() * cos(X(r + i * 2)) * sin(X(r + i * 2 + 1)) + -1.0 * gni.z() * sin(X(r + i * 2));
					g[r + i * 2 + 1] += -1.0* gni.x() * sin(X(r + i * 2)) * sin(X(r + i * 2 + 1)) + gni.y() * sin(X(r + i * 2)) * cos(X(r + i * 2 + 1));
					
				}
				// g of nj
				{
					Eigen::Vector3d t0 = Vij;
					double t1 = 2*X(pj);
					double t2 = Nall_tmp[i].transpose() * t0;
					double t3 = t0.transpose() * Nall_tmp[i];
					double t4 = t0.transpose() * t0;
					Eigen::Vector3d gnj(0,0,0);
					gnj = t1 * t2 * t3 * t0 + t1 * t2 * t4*Nall_tmp[i];
					g[r + pj * 2] += gnj.x() * cos(X(r + pj * 2)) * cos(X(r + pj * 2 + 1)) + gnj.y() * cos(X(r + pj * 2)) * sin(X(r + pj * 2 + 1)) + -1.0 * gnj.z() * sin(X(r + pj * 2));
					g[r + pj * 2 + 1] += -1.0 * gnj.x() * sin(X(r + pj * 2)) * sin(X(r + pj * 2 + 1)) + gnj.y() * sin(X(r + pj * 2)) * cos(X(r + pj * 2 + 1));
				}

			}
		}
		for (int i = 0; i < r; i++)
		{
			Eigen::Vector3d Pi = Vall[i] + X(i) * Nall_tmp[i];
			func += lambda* (Pi - Vall[i]).squaredNorm();
			double tt = (Pi - Vall[i]).transpose() * Nall_tmp[i];
			g[i] += lambda * 2 * tt;
		}

		//double cnt = 0;
		//for (int i = 0; i < r; i++)
		//{
		//	cnt +=  X(i);
		//	//g[i] += lambda * 2 * X(i);
		//}
		//for (int i = 0; i < r; i++)
		//{
		//	//cnt += X(i);
		//	g[i] += lambda * 2 * cnt;
		//}
		//func += lambda * cnt * cnt;

		return func;
	};


	
	BGAL::_LBFGS::_Parameter param = BGAL::_LBFGS::_Parameter();
	param.epsilon = 5e-4;
	param.is_show = true;
	BGAL::_LBFGS lbfgs(param);


	Eigen::VectorXd iterX(r*3);
	for (int i = 0; i < r; i++)
	{
		iterX(i) = 0;
	}
	for (int i = 0; i < r; i++)
	{
		auto Q = V3toV2(Nall_new[i]);
		iterX(r+i*2) = Q.first;
		iterX(r+i*2+1) = Q.second;
	}
	int n = lbfgs.minimize(fg, iterX);
	//int a = 53;
	//int n = lbfgs.test(a);
	//std::cout << iterX << std::endl;
	std::cout << "n: " << n << std::endl;


	for (int i = 0; i < r; i++)
	{
		double u = iterX(r+i*2);
		double v = iterX(r+i*2+1);
		Nall_new[i].x() = sin(u) * cos(v);
		Nall_new[i].y() = sin(u) * sin(v);
		Nall_new[i].z() = cos(u);
		Vall_new[i] = Vall[i] + iterX(i) * Nall_new[i];
	}



	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Running Time: " << endtime << endl;
	cout << "Running Time: " << endtime * 1000 << " ms " << endl;


	ofstream fout(outputPath+ outputFile+"\\"+model+"\\DenoisePoints.xyz");
	
	for (size_t i = 0; i < r; i++)
	{
		int k = neighboor[i].size();
		if (k < 5)
		{
			continue;
		}
		fout << Vall_new[i].transpose() << " " << Nall_new[i].transpose() << endl;
	}
	fout.close();



}

void Poisson(string model)
{
	clock_t start, end;
	start = clock();

	string outputFile = "Timing";
	string outputPath = "E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\";

	//ifstream in("E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\famous_pts_normal\\timing\\" + model + ".xyz");

	std::vector<Pwn> points;
	if (!CGAL::IO::read_points(CGAL::data_file_path("E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\famous_pts_normal\\timing\\" + model + ".xyz"), std::back_inserter(points),
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>())
		.normal_map(CGAL::Second_of_pair_property_map<Pwn>())))
	{
		std::cerr << "Error: cannot read input file!" << std::endl;
		return;
	}
	Polyhedron11 output_mesh;
	double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>
		(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()));
	if (CGAL::poisson_surface_reconstruction_delaunay
	(points.begin(), points.end(),
		CGAL::First_of_pair_property_map<Pwn>(),
		CGAL::Second_of_pair_property_map<Pwn>(),
		output_mesh, average_spacing))
	{
		std::ofstream out(outputPath + outputFile + "\\" + model + "\\model_poisson.off");
		out << output_mesh;
	}


	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Poisson Running Time: " << endtime << endl;


}


Eigen::MatrixXd V, U;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L;
igl::opengl::glfw::Viewer viewer;

void LaplaceTest()
{
	using namespace Eigen;
	using namespace std;



	// Load a mesh in OFF format
	igl::readOFF("..\\..\\data\\cow.off", V, F);
	cout << V.rows() << endl;
	// Compute Laplace-Beltrami operator: #V by #V
	igl::cotmatrix(V, F, L);

	// Alternative construction of same Laplacian
	SparseMatrix<double> G, K;
	// Gradient/Divergence
	igl::grad(V, F, G);
	// Diagonal per-triangle "mass matrix"
	VectorXd dblA;
	igl::doublearea(V, F, dblA);
	// Place areas along diagonal #dim times
	const auto& T = 1. * (dblA.replicate(3, 1) * 0.5).asDiagonal();
	// Laplacian K built as discrete divergence of gradient or equivalently
	// discrete Dirichelet energy Hessian
	K = -G.transpose() * T * G;
	std::cout << "|K-L|: " << (K - L).norm() << endl;

	const auto& key_down = [](igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod)->bool
	{
		switch (key)
		{
		case 'r':
		case 'R':
			U = V;
			break;
		case ' ':
		{
			// Recompute just mass matrix on each step
			SparseMatrix<double> M;
			igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
			// Solve (M-delta*L) U = M*U
			const auto& S = (M - 0.001 * L);
			Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
			assert(solver.info() == Eigen::Success);
			U = solver.solve(M * U).eval();
			// Compute centroid and subtract (also important for numerics)
			VectorXd dblA;
			igl::doublearea(U, F, dblA);
			double area = 0.5 * dblA.sum();
			MatrixXd BC;
			igl::barycenter(U, F, BC);
			RowVector3d centroid(0, 0, 0);
			for (int i = 0; i < BC.rows(); i++)
			{
				centroid += 0.5 * dblA(i) / area * BC.row(i);
			}
			U.rowwise() -= centroid;
			// Normalize to unit surface area (important for numerics)
			U.array() /= sqrt(area);
			break;
		}
		default:
			return false;
		}
		// Send new positions, update normals, recenter
		viewer.data().set_vertices(U);
		viewer.data().compute_normals();
		viewer.core().align_camera_center(U, F);
		return true;
	};


	// Use original normals as pseudo-colors
	MatrixXd N;
	igl::per_vertex_normals(V, F, N);
	MatrixXd C = N.rowwise().normalized().array() * 0.5 + 0.5;

	// Initialize smoothing with base mesh
	U = V;
	viewer.data().set_mesh(U, F);
	viewer.data().set_colors(C);
	viewer.callback_key_down = key_down;

	std::cout << "Press [space] to smooth." << endl;
	std::cout << "Press [r] to reset." << endl;
	//return viewer.launch();
}




/*************************************

Expect:
====================BOCSignTest
1
0
====================LinearSystemTest
2
2
2
====================ALGLIBTest
[-3.00,3.00]
false
false
false
====================LBFGSTest
0       0       0.001   172.8   7465
1       5       0.002   140.8   4956.19
2       6       0.003   3.17764e-14     2.52435e-28
reach the gradient tolerance
1
1
n: 2
====================BaseShapeTest
yes!
20
====================Tessellation2DTest
====================CVTLBFGSTest
0       0       0.007   0.00829834      0.00377827
1       2       0.014   0.00350175      0.0024125
2       3       0.022   0.00190224      0.00207575
3       4       0.03    0.00138204      0.00189172
4       5       0.037   0.001058        0.00179544
5       6       0.044   0.000630967     0.00174395
6       7       0.052   0.000453071     0.00171594
7       8       0.059   0.000407016     0.00169794
8       9       0.067   0.000382601     0.00168672
9       10      0.074   0.000236295     0.00167822
10      11      0.083   0.000188259     0.00167365
11      12      0.089   0.00018715      0.00166988
12      13      0.095   0.000191645     0.00166741
13      14      0.102   0.0001226       0.00166534
14      15      0.11    0.000109166     0.00166371
15      16      0.118   0.00012134      0.00166249
16      17      0.127   0.000142921     0.00166056
17      18      0.135   0.000219643     0.00165917
18      19      0.147   0.00015968      0.00165643
19      20      0.155   0.000126286     0.0016543
20      21      0.161   0.000124011     0.00165273
21      22      0.171   0.000113692     0.00165165
22      23      0.18    8.62573e-05     0.00165076
reach the gradient tolerance
====================DrawTest
====================IntegralTest
0.5
====================ModelTest
V number: 642
F number: 1280
====================Tessellation3DTest
====================KDTreeTest
61 0.0051334350000000004283 -0.95469340000000002533 0.18068470000000000364
0.16776960319175260317    0.16776960319175260317
my:  61 0.16776960319175260317
====================ICPTest
    0.92387953251128940302     0.38268343236508994831  6.7307270867900115263e-16     -0.8156403124820496009
   -0.38268343236509022587     0.92387953251128607235 -5.2041704279304212832e-17      0.1213708393898148552
 1.3444106938820254982e-15  1.0451708942760262744e-16      1.0000000000000013323       0.149999999999999023
                         0                          0                          0                          1
     1.0000000000000026645  4.4408920985006261617e-16   6.417535974601941462e-16 -2.2204460492503130808e-15
 5.5511151231257827021e-16     0.99999999999999933387  2.0949350896789406863e-16 -4.7184478546569152968e-16
 1.3444106938820254982e-15  1.0451708942760262744e-16      1.0000000000000013323 -9.7144514654701197287e-16
                         0                          0                          0                          1
====================MarchingTetrahedraTest
====================GeodesicDijkstraTest
====================CPDTest
0       0       2.8559999999999998721   0.97077838761131618472  1.5745422097489745195
1       1       3.7879999999999998117   0.50654419345871326552  0.83706428321869918996
2       2       4.3559999999999998721   0.10341526695700922756  0.54823767814940860266
3       3       4.9279999999999999361   0.064348966727981085634 0.53241666390643571649
4       4       5.4930000000000003268   0.055978077579559790133 0.52303719647387347802
5       5       6.0590000000000001634   0.027350443137352131728 0.51869437610611468514
6       6       6.6159999999999996589   0.025028562315720939702 0.51602315877088045237
7       7       7.1310000000000002274   0.041603689229046926512 0.51276448733047774731
8       8       7.6490000000000000213   0.028852734948012288135 0.51009318178425633317
9       9       8.1630000000000002558   0.018929838601576352147 0.50782103206131068429
10      10      8.6669999999999998153   0.016154404661631139445 0.50680290391269833261
11      11      9.1940000000000008384   0.01314021086265272989  0.50611520612709437472
12      12      9.7010000000000005116   0.012379146785242668358 0.50538449212462099869
13      13      10.180999999999999162   0.011617886072246796231 0.50490602085507330088
14      14      10.675000000000000711   0.0083309046909829185396        0.50453526747561872057
15      15      11.137999999999999901   0.0057183422454947386432        0.50432714568149383805
16      16      11.617000000000000881   0.0040848813100232556073        0.50421835653346636086
17      17      12.096000000000000085   0.0036973204459675658613        0.50412838219085553959
18      18      12.564999999999999503   0.0053717282361031475427        0.50404070920698462732
19      19      13.041999999999999815   0.0050809776275175227295        0.50392744977097758685
20      21      14.038000000000000256   0.0061808731807658189028        0.50381777140314776275
21      22      14.516000000000000014   0.0080891448900077683737        0.50370077706668570094
22      23      15.026999999999999247   0.0058684635027048412739        0.50356297222700674432
23      24      15.489000000000000767   0.0032095819943958178375        0.50348377130270893787
24      25      15.942999999999999616   0.0038988783924685118353        0.50342742924118755177
25      26      16.40899999999999892    0.0045883746557730644214        0.50337278805180996066
26      27      16.885999999999999233   0.0039180574454319282846        0.50331414136301411144
27      28      17.349000000000000199   0.0027244578038674569821        0.50326446135064273335
28      29      17.821000000000001506   0.0019379409723720758454        0.5032354190101470115
29      30      18.329000000000000625   0.0018087757414846634789        0.50321983470384168413
30      31      18.812000000000001165   0.0013028630258690508496        0.50320715636633683854
31      32      19.321000000000001506   0.0011783374032515757986        0.50319915304635931541
32      33      19.818999999999999062   0.0014676454032900041   0.50319086790881939475
33      34      20.353999999999999204   0.0015583810724837368535        0.50318094734710294702
34      35      20.833999999999999631   0.002362075227291897915 0.50316470947673608283
35      36      21.309999999999998721   0.0032787689329946251814        0.50314831577231877713
36      37      21.803999999999998494   0.0026986173532612236017        0.50312304385060069301
37      38      22.262000000000000455   0.0014219395958781934613        0.50310648530927826183
38      39      22.739000000000000767   0.001615130064985079595 0.50309781280288112804
39      40      23.193000000000001393   0.0010293988295718848724        0.50309054561560839769
40      41      23.653999999999999915   0.001150556612973379798 0.50308201595055312971
41      42      24.176999999999999602   0.0018894256762530183425        0.50306894262166490517
42      43      24.685999999999999943   0.0032389060252479831212        0.50303698963606058303
43      46      26.155999999999998806   0.0041089762752991101924        0.50301811622742298447
44      47      26.615999999999999659   0.0052529712037364022573        0.50298995415369451845
45      48      27.082000000000000739   0.0052228035541843978451        0.50291050212833288136
46      49      27.588000000000000966   0.0037558136639685708695        0.50283448424049492775
47      50      28.059000000000001052   0.0049192653062895708854        0.50274012469509332668
48      51      28.556000000000000938   0.010434469543566506078 0.5025991741619602049
49      52      29.042999999999999261   0.0094345597461962023983        0.50242554455932364466
50      53      29.55099999999999838    0.0054597914390715086147        0.50221842072561184711
51      54      30.047999999999998266   0.0059490895457608421876        0.50209167430241052887
52      55      30.553999999999998494   0.0063564344407932042366        0.50200891702364192071
53      56      31.030000000000001137   0.0043077053720921012689        0.50192435069084972987
54      57      31.545000000000001705   0.0027823923960697792557        0.50187037839039727594
55      58      32.029000000000003467   0.0019360024677122155638        0.50183171328020392821
56      59      32.529000000000003467   0.0033217704188521347645        0.50182773657329549089
57      60      33.045999999999999375   0.0010214169540488850663        0.50181491924833965257
58      61      33.567000000000000171   0.00064919895087481641321       0.50181214372791038691
59      62      34.08100000000000307    0.00055139179941391021095       0.50180937461070618255
60      63      34.624000000000002331   0.00078619052963909937578       0.50180825059119238407
61      64      35.082999999999998408   0.00027829333736625617426       0.5018074726296616328
62      65      35.61500000000000199    0.00016581815728628954663       0.50180731065158645787
63      66      36.06799999999999784    0.00012529229361561805794       0.50180715577438583797
64      67      36.533000000000001251   0.0001881051793069788672        0.50180708027468379218
65      68      37      7.1675672068046648237e-05       0.50180702933455167969
reach the gradient tolerance
mass - capacity
0.25012830819059872489  0.25012984804989535359  -1.5398592966286983597e-06
0.25013019810614184335  0.25012984804989535359  3.5005624648976052526e-07
0.25013058054167031097  0.25012984804989535359  7.3249177495737782806e-07
0.25012839111854828777  0.25012984804989535359  -1.4569313470658151743e-06
0.25012853615062691226  0.25012984804989535359  -1.3118992684413299799e-06
0.25013060255317565161  0.25012984804989535359  7.5450328029802449237e-07
0.25013039914674778386  0.25012984804989535359  5.5109685243026618195e-07
0.25012847467613846808  0.25012984804989535359  -1.3733737568855097777e-06
0.25013025518694881333  0.25012984804989535359  4.0713705345973849603e-07
0.25013061992165341874  0.25012984804989535359  7.7187175806514574106e-07
0.25013014735038452407  0.25012984804989535359  2.9930048917048424073e-07
0.25013043153240493988  0.25012984804989535359  5.8348250958628611329e-07
0.2501283195836258022   0.25012984804989535359  -1.5284662695513873132e-06
0.25012961148541623668  0.25012984804989535359  -2.3656447911690747787e-07
0.25013030405732905592  0.25012984804989535359  4.5600743370233232099e-07
0.25013015638657509765  0.25012984804989535359  3.0833667974405898349e-07
0.25012830901375759929  0.25012984804989535359  -1.5390361377543015919e-06
0.25013049676849052894  0.25012984804989535359  6.4871859517534602446e-07
0.25012961153793605851  0.25012984804989535359  -2.3651195929508261884e-07
0.25013055535693423659  0.25012984804989535359  7.0730703888299828463e-07
0.25013036070870031669  0.25012984804989535359  5.1265880496309534919e-07
0.25012834730090183211  0.25012984804989535359  -1.5007489935214834986e-06
0.25012812541727147408  0.25012984804989535359  -1.7226326238795053314e-06
0.25013048852504093933  0.25012984804989535359  6.4047514558573936938e-07
0.25013062244000516809  0.25012984804989535359  7.7439010981450451254e-07
0.25013009594294205451  0.25012984804989535359  2.4789304670091993898e-07
0.25013029708295431153  0.25012984804989535359  4.4903305895793721447e-07
0.25013016985467523279  0.25012984804989535359  3.2180477987919786642e-07
0.25012833494101649467  0.25012984804989535359  -1.5131088788589153182e-06
0.25013059658552866393  0.25012984804989535359  7.4853563331034322914e-07
0.25012839803272507444  0.25012984804989535359  -1.4500171702791497808e-06
0.25013017163012218891  0.25012984804989535359  3.2358022683531828534e-07
0.25013017696256534261  0.25012984804989535359  3.2891266998902324303e-07
0.25013038689742878029  0.25012984804989535359  5.3884753342670066445e-07
0.25013020554906495452  0.25012984804989535359  3.5749916960092775753e-07
0.25013012258386424502  0.25012984804989535359  2.7453396889143277804e-07
0.25013061224202304267  0.25012984804989535359  7.6419212768907840427e-07
0.25013031715688155421  0.25012984804989535359  4.6910698620061808128e-07
0.25013038684554850244  0.25012984804989535359  5.3879565314884914073e-07
0.2501304064087886414   0.25012984804989535359  5.5835889328781362906e-07
0.25013013896463942576  0.25012984804989535359  2.9091474407216821874e-07
0.25012840413401904449  0.25012984804989535359  -1.4439158763090986781e-06
0.25013006787565189581  0.25012984804989535359  2.1982575654222458184e-07
0.25012825838974489523  0.25012984804989535359  -1.5896601504583607323e-06
0.25013037638930035733  0.25012984804989535359  5.2833940500374154681e-07
0.25013031479326036655  0.25012984804989535359  4.6674336501295599078e-07
0.25013019806972219827  0.25012984804989535359  3.5001982684468302409e-07
0.25013060746784665511  0.25012984804989535359  7.5941795130152200954e-07
0.25013048499741385999  0.25012984804989535359  6.3694751850640329849e-07
0.25013061963995819603  0.25012984804989535359  7.7159006284244213703e-07
successful!

*******************************************/




int alltest()
{
	//std::cout << "====================GraphCutsTest" << std::endl;
	//GraphCutsTest();
	std::cout << "====================BOCSignTest" << std::endl;
	BOCSignTest();	
	std::cout << "====================LinearSystemTest" << std::endl;
	LinearSystemTest();
	std::cout << "====================ALGLIBTest" << std::endl;
	ALGLIBTest();
	std::cout << "====================LBFGSTest" << std::endl;
	LBFGSTest();
	std::cout << "====================BaseShapeTest" << std::endl;
	BaseShapeTest();
	std::cout << "====================Tessellation2DTest" << std::endl;
	Tessellation2DTest();
	std::cout << "====================CVTLBFGSTest" << std::endl;
	CVTLBFGSTest();
	std::cout << "====================DrawTest" << std::endl;
	DrawTest();
	std::cout << "====================IntegralTest" << std::endl;
	IntegralTest();
	std::cout << "====================ModelTest" << std::endl;
	ModelTest();
	std::cout << "====================Tessellation3DTest" << std::endl;
	Tessellation3DTest();
	//std::cout << "====================ReadFileTest" << std::endl;
	//ReadFileTest();
	std::cout << "====================KDTreeTest" << std::endl;
	KDTreeTest();
	std::cout << "====================ICPTest" << std::endl;
	ICPTest();
	std::cout << "====================MarchingTetrahedraTest" << std::endl;
	MarchingTetrahedraTest();
	std::cout << "====================GeodesicDijkstraTest" << std::endl;
	GeodesicDijkstraTest();
	std::cout << "====================CPDTest" << std::endl;
	CPDTest();
	std::cout << "====================CVT3DTest" << std::endl;
	CVT3DTest();
	std::cout << "successful!" << std::endl;
	return 0;
}

int main()
{
	//std::cout << "====================DenoiseTest" << std::endl;
	//string model = "82-block";
	//DenoiseTest(model);
	/*RFEPSTest(model);
	Comput_rnn(model);
	Comput_RPD(model);*/
	MyCapCVT2DTest();
	//CVTLBFGSTest();
	//std::cout << "====================CVT3DTest" << std::endl;
	//CVT3DTest();
	//MyCapCVT3DTest();
	//CapVT3DTest();


	
	//std::cout << "====================LaplaceTest" << std::endl;
	//MyLaplaceTest();
	//std::cout << "successful!" << std::endl;

//	rnnnum = 60;
//
//	string ss;
//	ifstream infilename("E:\\Dropbox\\MyProjects\\SIG-2022-Feature-preserving-recon\\data\\modelnames.txt");
//	while (infilename >> ss)
//	{
//		files.push_back(ss);
//	}
//	double omp_cnt = 0;
//	omp_set_num_threads(8);
////#pragma omp parallel for schedule(dynamic, 1)
//	for (int filenum = 105; filenum < files.size(); filenum++)
//	{
//		cout << filenum << "  " << files[filenum] << endl;
//		
//		omp_cnt++;
//		cout << endl<< omp_cnt << endl<< endl;
//		files[filenum] = "0.0025_00040345_dd3e90fda9884594a7ea4f63_trimesh_001";
//
//		
//		string model = files[filenum];
//
//		
//		//Poisson(model);
//		
//		//DenoiseTest(model);
//		RFEPSTest(model);
//		Comput_rnn(model);
//		Comput_RPD(model);
//		
//		break;
//
//	}
//	








	
	return 0;
}