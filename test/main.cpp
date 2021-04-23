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
	std::ofstream out("data\\TessellationTest.ps");
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
void CVTLBFGSTest()
{
	BGAL::_Polygon boundary;
	boundary.start_();
	boundary.insert_(BGAL::_Point2(0, 0));
	boundary.insert_(BGAL::_Point2(1, 0));
	boundary.insert_(BGAL::_Point2(1, 1));
	boundary.insert_(BGAL::_Point2(0, 1));
	boundary.end_();
	std::vector<BGAL::_Point2> sites;
	int num = 100;
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
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fg
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
		sites[i] = BGAL::_Point2(iterX(i * 2), iterX(i * 2 + 1));
	}
	voronoi.calculate_(boundary, sites);
	cells = voronoi.get_cell_polygons_();
	std::ofstream out("data\\CVTLBFGSTest.ps");
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
	std::ofstream out("data\\DrawTest.ps");
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
	BGAL::_ManifoldModel model("data\\sphere.obj");
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
	BGAL::_ManifoldModel model("data\\sphere.obj");
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
	std::ofstream out("data\\Tessellation3DTest.obj");
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
//	std::string path("data");
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
	std::ifstream ip("data\\KDTreeTest.txt");
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
	std::ifstream ip("data\\ICPTest.txt");
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
	model.save_obj_file_("data\\MarchingTetrahedraTest.obj");
}
//***********************************

void GeodesicDijkstraTest()
{
	BGAL::_ManifoldModel model("data\\sphere.obj");
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
	model.save_scalar_field_obj_file_("data\\GeodesicDijkstraTest.obj", distance);
}
//************************************

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
0       0       0       172.8   7465
1       5       0.001   140.8   4956.19
2       6       0.001   3.17764e-14     2.52435e-28
reach the gradient tolerance
1
1
n: 2
====================BaseShapeTest
yes!
20
====================Tessellation2DTest
====================CVTLBFGSTest
0       0       0.037   0.000670698     0.000314948
1       4       0.075   0.000393595     0.000250546
2       5       0.113   0.000173118     0.000199999
3       6       0.149   0.000117813     0.000186701
4       7       0.187   9.79624e-05     0.000178419
5       8       0.224   6.49128e-05     0.000173976
6       9       0.261   4.76155e-05     0.000171355
7       10      0.3     4.05314e-05     0.000169645
8       11      0.339   3.9684e-05      0.000168307
9       12      0.376   3.00581e-05     0.000167344
10      13      0.417   2.37168e-05     0.000166656
11      14      0.459   2.21291e-05     0.000166138
12      15      0.498   2.2534e-05      0.000165658
13      16      0.539   1.97825e-05     0.000165254
14      17      0.582   1.64508e-05     0.000164907
15      18      0.624   1.53663e-05     0.000164663
16      19      0.666   1.30847e-05     0.000164454
17      20      0.711   1.24741e-05     0.000164276
18      21      0.749   1.32204e-05     0.000164119
19      22      0.787   1.16706e-05     0.00016397
20      23      0.824   1.02782e-05     0.000163829
21      24      0.862   1.15896e-05     0.000163711
22      25      0.898   9.8418e-06      0.000163606
23      26      0.936   7.94319e-06     0.000163519
24      27      0.974   8.32563e-06     0.000163435
25      28      1.013   9.86239e-06     0.000163359
26      29      1.051   8.6653e-06      0.000163278
27      30      1.089   8.1746e-06      0.000163208
28      31      1.126   5.78644e-06     0.000163162
29      32      1.165   5.5085e-06      0.000163125
30      33      1.204   7.93797e-06     0.000163075
31      34      1.243   8.42584e-06     0.000163026
32      35      1.281   6.72924e-06     0.000162965
33      36      1.319   7.05505e-06     0.000162918
34      37      1.357   5.55057e-06     0.000162887
35      38      1.393   4.81077e-06     0.00016286
36      39      1.431   5.00659e-06     0.000162833
37      40      1.469   4.9121e-06      0.000162814
38      41      1.508   4.20857e-06     0.000162791
39      42      1.545   5.68848e-06     0.000162764
40      43      1.583   5.68351e-06     0.000162743
41      44      1.627   3.75176e-06     0.000162722
42      45      1.666   3.82435e-06     0.000162701
43      46      1.704   4.5564e-06      0.000162686
44      47      1.741   5.52616e-06     0.000162662
45      48      1.791   7.89004e-06     0.000162644
46      49      1.83    4.8662e-06      0.000162605
47      50      1.868   5.53777e-06     0.000162572
48      51      1.914   5.67505e-06     0.000162551
49      52      1.956   4.99743e-06     0.000162534
50      53      2.032   2.38233e-06     0.000162523
51      54      2.069   1.93189e-06     0.000162517
52      55      2.107   2.01117e-06     0.000162512
53      56      2.144   3.11163e-06     0.000162508
54      57      2.18    1.35862e-06     0.000162504
55      58      2.217   1.26264e-06     0.000162502
56      59      2.256   1.7025e-06      0.000162499
57      60      2.293   3.56916e-06     0.000162492
58      61      2.329   3.46262e-06     0.000162486
59      62      2.365   2.64079e-06     0.000162479
60      64      2.437   2.91038e-06     0.00016247
61      65      2.474   3.55324e-06     0.000162462
62      66      2.511   3.50128e-06     0.000162454
63      67      2.549   2.67009e-06     0.000162443
64      68      2.585   2.51626e-06     0.000162438
65      69      2.622   2.11462e-06     0.000162433
66      70      2.66    2.31716e-06     0.000162429
67      71      2.697   2.944e-06       0.000162422
68      73      2.769   3.87759e-06     0.000162416
69      74      2.806   3.87643e-06     0.000162407
70      75      2.842   2.91555e-06     0.000162394
71      76      2.88    2.77798e-06     0.000162385
72      77      2.916   2.99546e-06     0.000162382
73      78      2.952   1.96549e-06     0.000162378
74      79      2.989   1.88072e-06     0.000162376
75      80      3.026   1.12348e-06     0.000162375
76      81      3.063   1.01447e-06     0.000162373
77      82      3.1     1.36782e-06     0.000162372
78      83      3.136   1.87648e-06     0.00016237
79      84      3.172   1.3626e-06      0.000162368
80      85      3.209   1.49772e-06     0.000162365
81      86      3.246   3.59291e-06     0.00016236
82      87      3.284   4.47717e-06     0.000162357
83      88      3.323   3.99455e-06     0.000162346
84      89      3.36    2.6632e-06      0.000162336
85      90      3.396   1.84074e-06     0.000162332
86      91      3.433   1.33321e-06     0.000162329
87      92      3.469   1.5166e-06      0.000162326
88      93      3.507   1.1788e-06      0.000162324
89      94      3.544   7.97761e-07     0.000162323
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
	0.92387953251128684951     0.38268343236509028138 -3.9464959078472361398e-17    -0.81564031248204793556
   -0.38268343236509005933     0.92387953251129040222   3.647256108241236916e-16     0.12137083938981409192
-4.4669129506402782681e-16  2.9013250135712098654e-16      1.0000000000000013323     0.15000000000000049405
						 0                          0                          0                          1
	  1.000000000000000222 -8.8817841970012523234e-16 -1.7603531656562569079e-16 -3.3306690738754696213e-16
-2.7755575615628913511e-16      1.0000000000000035527  3.2185994082478728629e-16 -4.9960036108132044319e-16
-4.4669129506402782681e-16  2.9013250135712098654e-16      1.0000000000000013323  4.9960036108132044319e-16
						 0                          0                          0                          1
====================MarchingTetrahedraTest
====================GeodesicDijkstraTest
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
	std::cout << "successful!" << std::endl;
	return 0;
}

int main()
{
	alltest();
	return 0;
}