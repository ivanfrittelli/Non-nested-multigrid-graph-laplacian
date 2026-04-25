#pragma once

#include "graph.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


using namespace dealii;

struct TimeInfo {
  double dt;
  double theta;
  bool is_time_dependent;
};


class MG_Level {

  public:
    MG_Level(Graph & graph, int id): 
      dof_handler(triangulation),
      fe(1),
      graph(graph),
      id(id)
    {}
      
    void 
    make_grid();

    void
    setup_boundary_conditions(const std::set<int> & dirichlet_cells, const std::set<int> & neumann_cells);

    void
    setup_system(const Function<3> & boundary_conditions);

    void
    assemble_system(const TimeInfo & time_info);

    void
    assemble_rhs(const FunctionParser<3> & rhs_function, Vector<double> & rhs, 
      const FunctionParser<3> & neumann_function,
      const Vector<double> & old_solution, 
      const TimeInfo & time_info);

    std::map<int, int> 
    get_dof_to_vertex_map();

    std::map<int, int>
    get_vertex_to_dof_map();

    Graph graph;

    Triangulation<1,3> triangulation;
    FE_Q<1,3>          fe;
    DoFHandler<1,3>    dof_handler;

    int inlet_dof;
    Vector<double> resistances;

    AffineConstraints<double> constraints;
    
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    int id;
};