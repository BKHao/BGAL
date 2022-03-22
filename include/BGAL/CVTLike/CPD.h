#pragma once
#include "BGAL/BaseShape/Point.h"
#include "BGAL/BaseShape/Polygon.h"
#include "BGAL/BaseShape/Triangle.h"
#include "BGAL/BaseShape/Line.h"
#include "BGAL/Model/ManifoldModel.h"
#include "BGAL/Model/Model_Iterator.h"
#include "BGAL/Tessellation3D/Tessellation3D.h"
#include "BGAL/Optimization/LBFGS/LBFGS.h"

namespace BGAL
{
	class _CPD3D
	{
	public:
		_CPD3D(const _ManifoldModel& model);
		_CPD3D(const _ManifoldModel& model, std::function<double(_Point3 p)>& rho, _LBFGS::_Parameter para);
		void calculate_(const std::vector<double>& capacity);
		const std::vector<_Point3>& get_sites() const
		{
			return _sites;
		}
		const std::vector<double>& get_weights() const
		{
			return _weights;
		}
		const _Restricted_Tessellation3D& get_RPD() const
		{
			return _RPD;
		}
	protected:
		const _ManifoldModel& _model;
		_Restricted_Tessellation3D _RPD;
		std::vector<double> _capacity{};
		std::vector<_Point3> _sites{};
		std::vector<double> _weights{};
		std::function<double(_Point3 p)> _rho;
		_LBFGS::_Parameter _para;
	};
} // namespace BGAL
