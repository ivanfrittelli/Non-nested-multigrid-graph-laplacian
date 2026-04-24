#pragma once

#include <string>


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

#include <deal.II/grid/grid_out.h>

#include <deal.II/base/timer.h>

#include "MG_Level.hpp"
#include "multigridpreconditioner.hpp"

using namespace dealii;

struct PoissonParameters
{
  PoissonParameters() : transport_field(3)
  {
    prm.enter_subsection("Poisson parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Number of cycles", n_cycles);
      prm.add_parameter("Exact solution expression", exact_solution_expression);
      prm.add_parameter("Initial solution", initial_solution_expression);

      prm.add_parameter("Inlet pressure", inlet_pressure);
      prm.add_parameter("Inlet dof", inlet_dof);
      prm.add_parameter("Outlet pressure", outlet_pressure);

      prm.add_parameter("Right hand side expression", rhs_expression);
      prm.add_parameter("Neumann boundary expression", neumann_expression);
      prm.add_parameter("Neumann boundary ids", neumann_boundary_ids);
      prm.add_parameter("Number of eigenvalues", n_of_eigenvalues);
      prm.add_parameter("Number of V-cycles", n_v_cycles);
      prm.add_parameter("Coarse CG tollerance", coarse_cg_tollerance);
      prm.add_parameter("Omega", omega);
      prm.add_parameter("Number of Pre-smoothing", n_pre_smoothing);
      prm.add_parameter("Number of Post-smoothing", n_post_smoothing);
      prm.add_parameter("Coarseing percentage", coarsening_percentage);
      prm.add_parameter("Coarseing percentage", coarsening_percentage);
      prm.add_parameter("Graph refining", graph_refining);
      prm.add_parameter("Initial coarsening", initial_coarsening);
      prm.add_parameter("Only One", only_one);
      prm.add_parameter("Multigrid it", n_multigrid_it);

      prm.add_parameter("Compute no mg solution", compute_no_mg_solution);

      prm.add_parameter("Time steps", time_steps);
      prm.add_parameter("Time step length", time_step_length);
      prm.add_parameter("Theta", theta);

      prm.add_parameter("Resistance constant", resistance_constant);
    }
    prm.leave_subsection();

    prm.enter_subsection("Convergence table");
    convergence_table.add_parameters(prm);
    prm.leave_subsection();

    try
      {
        prm.parse_input("poisson.prm");
      }
    catch (std::exception &exc)
      {
        prm.print_parameters("poisson.prm");
        prm.parse_input("poisson.prm");
      }
    std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;

    exact_solution.initialize(FunctionParser<3>::default_variable_names(),
                              {exact_solution_expression},
                              constants);
    rhs_function.initialize(FunctionParser<3>::default_variable_names(),
                            {rhs_expression},
                            constants);
    neumann_function.initialize(FunctionParser<3>::default_variable_names(),
                                {neumann_expression},
                                constants);
    initial_solution.initialize(FunctionParser<3>::default_variable_names(),
                                {initial_solution_expression},
                                constants);

  }
  unsigned int fe_degree                 = 1;
  unsigned int initial_refinement        = 3;
  unsigned int n_cycles                  = 1;

  unsigned int n_v_cycles                = 1;
  unsigned int n_pre_smoothing           = 2;
  unsigned int n_post_smoothing          = 2;
  double omega                           = 0.7;
  double coarse_cg_tollerance            = 1e-12;
  double coarsening_percentage           = 0.5;
  double resistance_constant = 0;
  bool only_one = true;

  unsigned int inlet_dof = 0;
  double inlet_pressure = 2;

  int outlet_pressure = 1;
  int n_multigrid_it = 1;
  int initial_coarsening = 1;

  bool compute_no_mg_solution = true;

  //Time
  int time_steps = -1;
  double time_step_length = 0.1;
  double theta = 0;

  int graph_refining                     = 0;

  unsigned int n_of_eigenvalues          = 1;

  std::string  exact_solution_expression = "cos(pi*x)*cos(pi*y)";
  std::string  rhs_expression            = "2*pi*pi*cos(pi*x)*cos(pi*y)";
  std::string  neumann_expression        = "cos(2*pi*x)";
  std::string initial_solution_expression = "0";
  std::set<types::boundary_id> neumann_boundary_ids = {};

  FunctionParser<3> transport_field;
  FunctionParser<3> exact_solution;
  FunctionParser<3> initial_solution;
  FunctionParser<3> rhs_function;
  FunctionParser<3> neumann_function;

  mutable ParsedConvergenceTable convergence_table;

  ParameterHandler prm;
};

class Poisson
{
public:
  Poisson(const PoissonParameters &parameters);
  void
  run();

private:
  void
  make_grid();
  void
  mark();
  void
  refine();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results(const unsigned int cycle);

  const PoissonParameters &par;

  Vector<double> solution;
  Vector<double> no_mg_solution;
  Vector<double> system_rhs;

  std::vector<std::shared_ptr<MG_Level>> mg_levels;
  std::vector<std::map<int, int>> coarse_to_fine_dof_maps;

  //l'i-esimo va da livello[i+1] a livello[i]
  std::vector<std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>>> not_trivial_prolongations_dof;

  std::vector<DynamicSparsityPattern> prolongation_dynamic_patterns;
  std::vector<SparsityPattern> prolongation_sparsity_patterns;
  std::vector<SparseMatrix<double>> prolongations;
};
