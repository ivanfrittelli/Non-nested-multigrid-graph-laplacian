#pragma once

#include <deal.II/base/point.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/lac/vector.h>

#include<map>

using namespace dealii;

struct BigCell{
  int node1, node2;
  int cell_start, cell_end;
  int n_of_cells;
};

class Graph {
  public:
    Graph() {}

    using RestrictionMap = std::map<int, std::vector<std::pair<int,double>>>;

    void add_point(double x, double y, double z);
    void add_point(Point<3> point);

    void add_cell(int node1, int node2);
    void add_big_cell(int node1, int node2, int cell_start, int cell_end, int n_of_cells);

    int get_number_of_points();
    int get_number_of_cells();

    bool is_simple();

    Triangulation<1,3> create_graph_triangulation();

    Graph get_coarser_graph(
      RestrictionMap & vertex_prolongation_map,
      double length_treshold
    );

    Graph get_classic_coarser_graph(
      std::map<int, std::vector<int>> & coarse_to_fine_vertex_map, 
      std::map<int, int> & coarse_to_fine_cell_map, 
      double length_treshold);

    Graph get_coarser_graph_lort(RestrictionMap & vertex_prolongation_map);

    void coarse_big_cell(Graph & result, const BigCell & big_cell, 
      RestrictionMap & vertex_prolongation_map, std::map<int, int> & fine_to_coarse_vertex_map);

    Graph get_finer_graph(int n_of_points_per_segment);

    std::vector<Point<3>> points;
    std::vector<CellData<1>> cells;

    std::vector<BigCell> big_cells;
    
    std::vector<std::vector<int>> adiacency;    

    std::map<int,int> small_to_big_cell_map;

    std::vector<int> dirichlet_big_cells;
    std::vector<int> neumann_big_cells;
};
