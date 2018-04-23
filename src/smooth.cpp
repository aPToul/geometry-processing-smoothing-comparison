#include "smooth.h"
#include <iostream>
#include <igl/edge_lengths.h>
#include "massmatrix.h"
#include <Eigen/SparseCholesky>
#include <igl/edges.h>
#include "cotmatrix.h"

using namespace std;

void graph_Laplacian(
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double> & L)
{
  typedef Eigen::Triplet<double> T;

  std::vector<T> tripletList;
  // For each edge, two elements of L are filled in with + 1
  // We will add the diagonal elements after
  tripletList.reserve(E.rows() * 2);

  for(int edge_number = 0; edge_number < E.rows(); edge_number++)
  {
    auto start_node_index = E(edge_number, 0);
    auto end_node_index = E(edge_number, 1);
    
    tripletList.push_back(T(start_node_index, end_node_index, 1.0));
    tripletList.push_back(T(end_node_index, start_node_index, 1.0));
  }
  L.setFromTriplets(tripletList.begin(), tripletList.end());
  
  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}

void edge_weighted_Laplacian(
  const Eigen::MatrixXd & l,
  const Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double> & L)
{
  // Unknown effect of added distance to all points
  double epsilon = 0.00; // 0.01 

  // Unknown effect of lower bounding the edge lengths
  double edge_threshold = 0.0000; // 0.0001

  // For debugging
  double all_edge_differences = 0;
  int edges_added = 0;
  // Slightly redundant since logic is now edge-wise
  // Will simply overwrite each half-edge double-visited
  for (int faceIndex = 0; faceIndex < F.rows(); faceIndex++){
    auto vertices = F.row(faceIndex);
    // Indices
    int v1 = vertices[0];
    int v2 = vertices[1];
    int v3 = vertices[2];

    // Lengths of [1,2],[2,0],[0,1]
    // Lengths of [v2, v3], [v3, v1], [v1 , v2]
    auto lengths = l.row(faceIndex);
    // Side lengths
    double s1 = lengths[0];
    double s2 = lengths[1];
    double s3 = lengths[2];

    if (s1 < 0 or s2 < 0 or s3 < 0)
    {
      cout << "Assumption of positive side length violated!" << endl;
    }

    if (L.coeffRef(v1, v2) == 0)
    {
      if (s3 > edge_threshold)
      {
        // Side 1 and 2
        L.coeffRef(v1, v2) = 1.0 / (s3 + epsilon);
        L.coeffRef(v2, v1) = 1.0 / (s3 + epsilon);
        all_edge_differences += s3;
        edges_added += 1;        
      }
    }

    if (L.coeffRef(v2, v3) == 0)
    {
      if (s1 > edge_threshold) 
      {
        // Side 2 and 3
        L.coeffRef(v2, v3) = 1.0 / (s1 + epsilon);
        L.coeffRef(v3, v2) = 1.0 / (s1 + epsilon);
        all_edge_differences += s1;
        edges_added += 1;
      }
    }

    if (L.coeffRef(v1, v3) == 0)
    {
      if (s2 > edge_threshold) 
      {
        // Side 3 and 1
        L.coeffRef(v1, v3) = 1.0 / (s2 + epsilon);
        L.coeffRef(v3, v1) = 1.0 / (s2 + epsilon);
        all_edge_differences += s2;
        edges_added += 1;
      }
    }
  }

  if (edge_threshold > 0){
    cout << "Edges added: " << edges_added << endl;    
  }

  if (epsilon > 0){
    cout << "Average edge length: " << all_edge_differences / edges_added << endl;
  }

  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    // for some reason cannot actually edit the .diagonal()
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}

void smooth(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXd & G,
    double lambda,
    Eigen::MatrixXd & U,
    int mode)
{
  int number_of_vertices = V.rows();

  Eigen::SparseMatrix<double>Laplacian(number_of_vertices, number_of_vertices);
  Eigen::DiagonalMatrix<double,Eigen::Dynamic> Mass(number_of_vertices);

  if (mode == 0){
    cout << "Graph" << endl;
    Eigen::MatrixXi E;
    igl::edges(F, E);
    graph_Laplacian(E, Laplacian);
    Mass.setIdentity(number_of_vertices);
  }
  else if (mode == 1){
    cout << "Edge" << endl;
    Eigen::MatrixXd edge_lengths;
    igl::edge_lengths(V, F, edge_lengths);
    edge_weighted_Laplacian(edge_lengths, F, Laplacian);
    Mass.setIdentity(number_of_vertices);
  }
  else if (mode == 2){
    cout << "Cotangent" << endl;
    Eigen::MatrixXd edge_lengths;
    igl::edge_lengths(V, F, edge_lengths);
    cotmatrix(edge_lengths, F, Laplacian);
    massmatrix(edge_lengths, F, Mass);
  }

  Eigen::SparseMatrix<double> healthyA;
  // Multiply by the step-size lambda
  healthyA = - lambda * Laplacian;
  // Add in mass, which preserves symmetry
  for (int diagIndex = 0; diagIndex < Mass.rows(); diagIndex++){
    healthyA.coeffRef(diagIndex, diagIndex) += Mass.diagonal()[diagIndex];
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(healthyA);
  U = solver.solve(Mass * G);
}