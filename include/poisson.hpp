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
  PoissonParameters()
  {
    prm.enter_subsection("Time");
    {
      prm.add_parameter("Time dependent", is_time_dependent);
      prm.add_parameter("Number of time steps", time_steps);
      prm.add_parameter("Time step length", time_step_length);
      prm.add_parameter("Theta", theta);
    }
    prm.leave_subsection();

    prm.enter_subsection("Mesh");
    {
      prm.add_parameter("Mesh file name", mesh_file_name);
      prm.add_parameter("Initial coarsening", initial_coarsening);
    }
    prm.leave_subsection();

    prm.enter_subsection("Multigrid");
    {
      prm.add_parameter("Compute solution without multigrid", compute_no_mg_solution);
      prm.add_parameter("Number of levels", n_v_cycles);
      prm.add_parameter("Number of pre-smoothing steps", n_pre_smoothing);
      prm.add_parameter("Number of post-smoothing steps", n_post_smoothing);
      prm.add_parameter("SSOR smoother damping", omega);
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions and rhs");
    {
      prm.add_parameter("Time zero solution", initial_solution_expression);
      prm.add_parameter("Arterioles neumann boundary condition", neumann_expression);
      prm.add_parameter("Venules dirichlet boundary condition", exact_solution_expression);

      prm.add_parameter("Right hand side expression", rhs_expression);
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


    std::string variables = FunctionParser<3>::default_variable_names();

    if (is_time_dependent) 
      variables = variables + ", t";

    rhs_function.initialize(variables,
                            {rhs_expression},
                            constants, is_time_dependent);

    neumann_function.initialize(variables,
                                {neumann_expression},
                                constants, is_time_dependent);

    //(Per ora) non è time dependent
    //Mi sembra realistico che a cambiare sia il flusso di sangue all'entrata
    //Mentre la pressione in uscita sia sempre la stessa
    //Potrebbe cambiare in futuro
    exact_solution.initialize(FunctionParser<3>::default_variable_names(),
                              {exact_solution_expression},
                              constants, false);

    //Non è time depdendent perché è la soluzione al tempo 0
    initial_solution.initialize(FunctionParser<3>::default_variable_names(),
                                {initial_solution_expression},
                                constants, false);

  }
  //Time
  int time_steps = 10;
  double time_step_length = 0.1;
  double theta = 1;
  bool is_time_dependent = false;

  //Mesh
  std::string mesh_file_name = "Network1.vtk";
  int initial_coarsening = 1;
  bool only_one = false;

  //Multigrid
  bool compute_no_mg_solution = false;
  unsigned int n_v_cycles                = 1;
  unsigned int n_pre_smoothing           = 2;
  unsigned int n_post_smoothing          = 2;
  double omega                           = 0.7;

  //Boundary conditions and rhs
  std::string initial_solution_expression = "0";
  mutable FunctionParser<3> initial_solution;
  std::string  neumann_expression        = "10";
  mutable FunctionParser<3> neumann_function;
  std::string  exact_solution_expression = "0";
  mutable FunctionParser<3> exact_solution;
  std::string  rhs_expression            = "1";
  mutable FunctionParser<3> rhs_function;

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

  Vector<double> radii;

  std::vector<std::shared_ptr<MG_Level>> mg_levels;
  std::vector<std::map<int, int>> coarse_to_fine_dof_maps;

  std::vector<DynamicSparsityPattern> restriction_dynamic_patterns;
  std::vector<SparsityPattern> restriction_sparsity_patterns;
  std::vector<SparseMatrix<double>> restrictions;
};
