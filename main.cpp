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

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/arpack_solver.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#  include <deal.II/grid/grid_in.h>
#  include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#  include <deal.II/grid/grid_tools.h>
#  include <deal.II/grid/tria.h>
#  include <deal.II/grid/tria_accessor.h>
#  include <deal.II/grid/tria_description.h>
#  include <deal.II/grid/tria_iterator.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace dealii;

struct PoissonParameters
{
  PoissonParameters()
  {
    prm.enter_subsection("Poisson parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Number of cycles", n_cycles);
      prm.add_parameter("Exact solution expression", exact_solution_expression);
      prm.add_parameter("Right hand side expression", rhs_expression);
      prm.add_parameter("Neumann boundary expression", neumann_expression);
      prm.add_parameter("Neumann boundary ids", neumann_boundary_ids);
      prm.add_parameter("Number of eigenvalues", n_of_eigenvalues);
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
  }
  unsigned int fe_degree                 = 1;
  unsigned int initial_refinement        = 3;
  unsigned int n_cycles                  = 1;

  unsigned int n_of_eigenvalues          = 1;

  std::string  exact_solution_expression = "cos(pi*x)*cos(pi*y)";
  std::string  rhs_expression            = "2*pi*pi*cos(pi*x)*cos(pi*y)";
  std::string  neumann_expression        = "cos(2*pi*x)";
  std::set<types::boundary_id> neumann_boundary_ids = {};

  FunctionParser<3> exact_solution;
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
  output_results(const unsigned int cycle) const;

  const PoissonParameters &par;

  Triangulation<1,3> triangulation;
  FE_Q<1,3>          fe;
  DoFHandler<1,3>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};




Poisson::Poisson(const PoissonParameters &par)
  : par(par)
  , fe(par.fe_degree)
  , dof_handler(triangulation)
{}


void
Poisson::make_grid()
{
  std::ifstream in("mesh.vtk");

  //Contiene ciò che devo leggere dal file
  std::string buffer;

  //Le prime 4 righe non contengono nulla
  for (int i = 0; i < 4; i++)
    getline(in, buffer);

  //La parola POINTS, non ci serve
  in >> buffer;

  //Numero di punti
  in >> buffer;
  int n_points = std::stoi(buffer);
  std::vector<Point<3>> points(n_points);

  //Il tipo della variabile, non ci serve
  in >> buffer;

  //Variabili temporanee
  double x,y,z;

  for (int i = 0; i < n_points; i++) {
    //Le tre coordinate
    in >> buffer;
    x = std::stod(buffer);

    in >> buffer;
    y = std::stod(buffer);

    in >> buffer;
    z = std::stod(buffer);

    points[i] = Point<3>(x,y,z);
  }

  //Per ora ho cancellato dalla mesh l'offset, poi vedo

  //La parola CELLS, non ci serve
  in >> buffer;

  //Il numero di celle
  in >> buffer;
  int n_cells = std::stoi(buffer);


  //Parole che non ci servono
  for (int i = 0; i < 3; i++)
    in >> buffer;

  std::vector<std::array<int,GeometryInfo<1>::vertices_per_cell>>
    cell_vertices;

  int node1, node2;
  for (int i = 0; i < 2 * (n_cells-1); i++) {
    in >> buffer;
    node1 = std::stoi(buffer);

    i++;

    in >> buffer;
    node2 = std::stoi(buffer);

    cell_vertices.push_back({node1, node2});
  }
  

   std::vector<CellData<1>> cells(cell_vertices.size(), CellData<1>());
  for (unsigned int i=0; i<cell_vertices.size(); ++i)
    for (unsigned int j=0; j<GeometryInfo<1>::vertices_per_cell; ++j)
      cells[i].vertices[j] = cell_vertices[i][j];

  triangulation.create_triangulation (points, cells, SubCellData());

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


void
Poisson::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();
  auto all_boundary_ids = triangulation.get_boundary_ids();
  std::set<types::boundary_id> dirichlet_boundary_ids;
  for (const auto &id : all_boundary_ids)
    if (par.neumann_boundary_ids.find(id) == par.neumann_boundary_ids.end())
      dirichlet_boundary_ids.insert(id);

  for (const auto &id : dirichlet_boundary_ids)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             id,
                                             par.exact_solution,
                                             constraints);

  // Create hanging node constraints
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}




void
Poisson::assemble_system()
{
  QGauss<1>     quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            par.rhs_function.value(x_q) *       // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

}


void
Poisson::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}




void
Poisson::output_results(const unsigned int cycle) const
{
  DataOut<1,3> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  auto fname =
    "solution_" + std::to_string(cycle) + ".vtu";

  std::ofstream output(fname);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back({cycle, fname});

  std::ofstream pvd_output("solution.pvd");

  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}




void
Poisson::run()
{
  for (unsigned int cycle = 0; cycle < par.n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid();
      else
        {
          triangulation.refine_global();
        }
      setup_system();
      assemble_system();
      solve();
      output_results(cycle);
      par.convergence_table.error_from_exact(dof_handler,
                                             solution,
                                             par.exact_solution);
    }
  par.convergence_table.output_table(std::cout);
}



int
main()
{
  {
    PoissonParameters par;
    Poisson           laplace_problem_1d(par);
    laplace_problem_1d.run();
  }

  return 0;
}