#pragma once
#include "MyBaseModel.hpp"
#include <cmath>
#include <queue>
class MyHalfEdgeModel :	public MyBaseModel
{
public:
	struct MyMeshEdge
	{
		int leftVert;
		int rightVert;
		int indexOfFrontFace;
		int indexOfOppositeVertex;
		int indexOfPreviousEdge;   //pre
		int indexOfNextEdge;  //next
		int indexOfReverseEdge;   //twin
		double leftAngle;        //����������н�
		double rightAngle;
		double length;
	};
protected:
	vector<MyMeshEdge> m_edges;
	map<pair<int, int>, int> fromEdge2ID;
	vector<int> fromVert2Edge;
	vector<vector<int>> fromVert2Face;
public:
	void ReadObjFile(const char* filename);
	vector<int> GetNeighboringVertices(int v) const;
	void CreateMeshEdges();
	MyHalfEdgeModel()
	{

	}
	MyHalfEdgeModel(vector<Eigen::Vector3d> verts, vector<Eigen::Vector3i> faces):MyBaseModel(verts,faces)
	{

	}
	vector<int> GetFacesByPoint(int id)
	{
		return fromVert2Face[id];
	}
	vector<MyMeshEdge> GetEdges()
	{
		return m_edges;
	}
	MyMeshEdge* GetEdgeByID(int ID) //���ݱߵ��±��ұ�
	{
		return &m_edges[ID];
	}
	MyMeshEdge* GetEdgeByPoint(int IDOfVer)   //���ݵ���±��ұ�
	{
		return GetEdgeByID(fromVert2Edge[IDOfVer]);
	}
	vector<MyMeshEdge*> GetEdgeByFace(int IDOfFace) //��������±��ұ�(����һ��)
	{
		vector<MyMeshEdge*> set;
		set.push_back(GetEdgeByID(IDOfFace * 3));
		set.push_back(GetEdgeByID(IDOfFace * 3 + 1));
		set.push_back(GetEdgeByID(IDOfFace * 3 + 2));
		return set;
	}
	int GetNumberOfComponents() ;
	vector<MyBaseModel> Decompose() ;
	//int GetNumberOfOpenBoundaries() const;
	int GetNumberOfGenii() const;
};

void MyHalfEdgeModel::CreateMeshEdges()
{
	m_edges.clear();
	fromVert2Edge.clear();
	m_edges.resize(3 * m_faces.size());
	fromVert2Edge.resize(m_verts.size());
	fromEdge2ID.clear();
	for (int i = 0; i < m_verts.size(); ++i)
	{
		fromVert2Face.push_back(vector<int>());
	}


	for (int i = 0; i < m_faces.size(); ++i)
	{
		//MyMeshEdge temp;
		fromVert2Face[m_faces[i][0]].push_back(i);
		fromVert2Face[m_faces[i][1]].push_back(i);
		fromVert2Face[m_faces[i][2]].push_back(i);


		for (int j = 0; j < 3; j++)
		{
			//3*i+j
			m_edges[3 * i + j].indexOfFrontFace = i;

			m_edges[3 * i + j].leftVert = m_faces[i][j];
			fromVert2Edge[m_edges[3 * i + j].leftVert] = 3 * i + j;

			m_edges[3 * i + j].rightVert = m_faces[i][(j + 1) % 3];
			m_edges[3 * i + j].indexOfOppositeVertex = m_faces[i][(j + 2) % 3];
			m_edges[3 * i + j].indexOfPreviousEdge = 3 * i + (j + 3 - 1) % 3;
			m_edges[3 * i + j].indexOfNextEdge = 3 * i + (j + 3 + 1) % 3;
			fromEdge2ID[make_pair(m_edges[3 * i + j].leftVert, m_edges[3 * i + j].rightVert)] = 3 * i + j;
			//��߳�
			m_edges[3 * i + j].length = (m_verts[m_edges[3 * i + j].leftVert] - m_verts[m_edges[3 * i + j].rightVert]).norm();

			Eigen::Vector3d v1, v2;
			v1 = m_verts[m_edges[3 * i + j].indexOfOppositeVertex] - m_verts[m_edges[3 * i + j].leftVert];
			v2 = m_verts[m_edges[3 * i + j].rightVert] - m_verts[m_edges[3 * i + j].leftVert];

			double cosValNew = v1.dot(v2) / (v1.norm() * v2.norm()); //�Ƕ�cosֵ
			m_edges[3 * i + j].leftAngle = acos(cosValNew);// *180 / EIGEN_PI;

			v1 = m_verts[m_edges[3 * i + j].indexOfOppositeVertex] - m_verts[m_edges[3 * i + j].rightVert];
			v2 = m_verts[m_edges[3 * i + j].leftVert] - m_verts[m_edges[3 * i + j].rightVert];
			cosValNew = v1.dot(v2) / (v1.norm() * v2.norm()); //�Ƕ�cosֵ
			m_edges[3 * i + j].rightAngle = acos(cosValNew);// *180 / EIGEN_PI;


		}
	}
	for (int i = 0; i < m_edges.size(); i++)
	{
		if (fromEdge2ID.find(make_pair(m_edges[i].rightVert, m_edges[i].leftVert)) != fromEdge2ID.end())
		{
			m_edges[i].indexOfReverseEdge = fromEdge2ID[make_pair(m_edges[i].rightVert, m_edges[i].leftVert)];
		}
		else
		{
			m_edges[i].indexOfReverseEdge = -1;
		}
		
	}	
}

int MyHalfEdgeModel::GetNumberOfComponents()
{
	
	set<int> unVisistedVertices;
	for (int i = 0; i < m_verts.size(); ++i)
	{
		unVisistedVertices.insert(i);
		VertsId2Component.push_back(0);
	}

	int numberOfComponents = 0;
	while (!unVisistedVertices.empty())
	{
		
		++numberOfComponents; //��ͨ��֧++
		auto firstVertex = *unVisistedVertices.begin();//������ǰ��ĵ�
		
		queue<int> tobedeleted; //Ҫ��ɾ�ĵ�
		tobedeleted.push(firstVertex); //װ��

		VertsId2Component[firstVertex] = numberOfComponents;

		unVisistedVertices.erase(firstVertex); //�ڼ�����ɾ��

		while (!tobedeleted.empty())  
		{	
			auto top = tobedeleted.front();//
			
			tobedeleted.pop();
			auto firstEdge = fromVert2Edge[top]; //@chengjialong �ҵ��ı�Ϊf�����һ����topΪ���ı�
			
			auto nxtEdge = firstEdge; //	
			do
			{
				
				if (unVisistedVertices.find(m_edges[nxtEdge].rightVert) != unVisistedVertices.end()) //����v2e�ҵ��ıߵ��Ҷ˵�δ�����ʹ�
				{	
					tobedeleted.push(m_edges[nxtEdge].rightVert);
					VertsId2Component[m_edges[nxtEdge].rightVert] = numberOfComponents;
					
					unVisistedVertices.erase(m_edges[nxtEdge].rightVert);
					nxtEdge = m_edges[m_edges[nxtEdge].indexOfPreviousEdge].indexOfReverseEdge; //�ҵ������ߵ�ǰ�ߵķ���
					
				}
				else
				{
					nxtEdge = m_edges[m_edges[nxtEdge].indexOfPreviousEdge].indexOfReverseEdge; //�ҵ������ߵ�ǰ�ߵķ���
				}
				if (nxtEdge == -1)
				{
					break;
				}
							
			} while (nxtEdge != firstEdge);
		}

	}
	return numberOfComponents;
}

vector<MyBaseModel> MyHalfEdgeModel::Decompose()
{
	int numberOfComponents = GetNumberOfComponents();
	
	vector<vector<Eigen::Vector3d>> verts_multi_component;
	verts_multi_component.clear();
	verts_multi_component.resize(numberOfComponents + 1);
	vector<vector<Eigen::Vector3i>> faces_multi_component;
	faces_multi_component.clear();
	faces_multi_component.resize(numberOfComponents + 1);
	
	vector<MyBaseModel> resModels;
	vector<int> StartCounter;
	StartCounter.push_back(0);
	for (int i = 0; i < m_verts.size(); i++)
	{
		verts_multi_component[VertsId2Component[i]-1].push_back(m_verts[i]);
		if (i > 0 && VertsId2Component[i] > VertsId2Component[i - 1])
		{
			StartCounter.push_back(i);
		}
	}

	for (int i = 0; i < m_faces.size(); i++) 
	{
		int index = VertsId2Component[m_faces[i].x()]-1;//�����ĸ���ͨ��֧��
		Eigen::Vector3i new_face;
		new_face.x() = m_faces[i].x() - StartCounter[index];
		new_face.y() = m_faces[i].y() - StartCounter[index];
		new_face.z() = m_faces[i].z() - StartCounter[index];
		faces_multi_component[index].push_back(new_face);
    }

	for (int i = 0; i < numberOfComponents; i++)
	{
		MyBaseModel MBM(verts_multi_component[i], faces_multi_component[i]);
		resModels.push_back(MBM);
	}
	return resModels;
}

int MyHalfEdgeModel::GetNumberOfGenii() const
{
	return 1 - (m_verts.size() + m_faces.size() - m_edges.size() / 2) / 2;
}

vector<int> MyHalfEdgeModel::GetNeighboringVertices(int v) const
{
	vector<int> neighbors;
	int firstEdge = fromVert2Edge[v];
	int nxtEdge = firstEdge;
	do
	{
		neighbors.push_back(m_edges[nxtEdge].rightVert);
		nxtEdge = m_edges[m_edges[nxtEdge].indexOfPreviousEdge].indexOfReverseEdge;
		if (nxtEdge == -1)
		{
			break;
		}
	} while (nxtEdge != firstEdge);
	return neighbors;
}

void MyHalfEdgeModel::ReadObjFile(const char* filename)
{
	MyBaseModel::ReadObjFile(filename);
	CreateMeshEdges();
}