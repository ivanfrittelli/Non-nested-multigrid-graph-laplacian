#pragma once

#include <deal.II/base/point.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

class Graph {
  public:
    Graph() {}

    void add_point(double x, double y, double z);
    void add_point(Point<3> point);

    void add_cell(int node1, int node2);

    int get_number_of_points();
    int get_number_of_cells();

    Triangulation<1,3> create_graph_triangulation();

    Graph get_coarser_graph(
      std::map<int, std::vector<int>> & coarse_to_fine_vertex_map, 
      std::map<int, int> & coarse_to_fine_cell_map, 
      double coarsening_percentage,
      const std::vector<double> & resistances);
    Graph get_finer_graph(int n_of_points_per_segment);

    std::vector<Point<3>> points;
    std::vector<CellData<1>> cells;

    std::vector<std::vector<int>> adiacency;    
};
