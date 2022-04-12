#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/Integral/Integral.h"
#include "BGAL/Optimization/LinearSystem/LinearSystem.h"

namespace BGAL
{
	_CVT3D::_CVT3D(const _ManifoldModel& model) : _model(model), _RVD(model), _para()
	{
		_rho = [](BGAL::_Point3& p)
		{
			return 1;
		};
		_para.is_show = true;
		_para.epsilon = 5e-5;
	}
	_CVT3D::_CVT3D(const _ManifoldModel& model, std::function<double(_Point3& p)>& rho, _LBFGS::_Parameter para) : _model(model), _RVD(model), _rho(rho), _para(para)
	{
		
	}
	void _CVT3D::calculate_(int num_sites)
	{		
		int num = num_sites;
		_sites.resize(num);
		for (int i = 0; i < num; ++i)
		{
			int fid = rand() % _model.number_faces_();
			double l0, l1, l2, sum;
			l0 = _BOC::rand_();
			l1 = _BOC::rand_();
			l2 = _BOC::rand_();
			sum = l0 + l1 + l2;
			l0 /= sum;
			l1 /= sum;
			l2 /= sum;
			_sites[i] = _model.face_(fid).point(0) * l0 + _model.face_(fid).point(1) * l1 + _model.face_(fid).point(2) * l2;
		}
		_RVD.calculate_(_sites);
		std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fg
			= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
		{
			for (int i = 0; i < num; ++i)
			{
				BGAL::_Point3 p(X(i * 3), X(i * 3 + 1), X(i * 3 + 2));
				// 这里是不是需要将点投影到mesh表面有待考虑
				_sites[i] = p;
			}
			_RVD.calculate_(_sites);
			const std::vector<std::vector<std::tuple<int, int, int>>>& cells = _RVD.get_cells_();
			double energy = 0;
			g.setZero();
			for (int i = 0; i < num; ++i)
			{
				for (int j = 0; j < cells[i].size(); ++j)
				{
					Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
						[&](BGAL::_Point3 p)
						{
							Eigen::VectorXd r(5);
							r(0) = _rho(p);
							r(1) = _rho(p) * ((_sites[i] - p).sqlength_());
							r(2) = 2 * _rho(p) * (_sites[i].x() - p.x());
							r(3) = 2 * _rho(p) * (_sites[i].y() - p.y());
							r(4) = 2 * _rho(p) * (_sites[i].z() - p.z());
							return r;
						}, _RVD.vertex_(std::get<0>(cells[i][j])), _RVD.vertex_(std::get<1>(cells[i][j])), _RVD.vertex_(std::get<2>(cells[i][j]))
							);
					energy += inte(1);
					g(i * 3) += inte(2);
					g(i * 3 + 1) += inte(3);
					g(i * 3 + 2) += inte(4);
				}
			}
			return energy;
		};
		BGAL::_LBFGS lbfgs(_para);
		Eigen::VectorXd iterX(num * 3);
		for (int i = 0; i < num; ++i)
		{
			iterX(i * 3) = _sites[i].x();
			iterX(i * 3 + 1) = _sites[i].y();
			iterX(i * 3 + 2) = _sites[i].z();
		}
		lbfgs.minimize(fg, iterX);
		for (int i = 0; i < num; ++i)
		{
			_sites[i] = BGAL::_Point3(iterX(i * 3), iterX(i * 3 + 1), iterX(i * 3 + 2));
		}
		_RVD.calculate_(_sites);
	}
} // namespace BGAL
