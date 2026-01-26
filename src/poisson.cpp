#include "../include/poisson.hpp"


#include <deal.II/base/function.h>
#include <deal.II/lac/precondition.h>
#include <iostream>
#include <fstream>

Poisson::Poisson(const PoissonParameters &par)
  : par(par)
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

  Graph fine_graph;

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

    fine_graph.add_point(x, y, z);
  }

  //Per ora ho cancellato dalla mesh l'offset, poi vedo

  //La parola CELLS, non ci serve
  in >> buffer;

  //Il numero di celle
  in >> buffer;
  unsigned int n_cells = std::stoi(buffer) - 1;

  //Parole che non ci servono
  for (int i = 0; i < 3; i++)
    in >> buffer;

  int node1, node2;
  for (unsigned int i = 0; i < 2 * n_cells; i++) {
    in >> buffer;
    node1 = std::stoi(buffer);

    i++;

    in >> buffer;
    node2 = std::stoi(buffer);

    fine_graph.add_cell(node1, node2);
  }

  if (par.graph_refining > 0) {
    fine_graph = fine_graph.get_finer_graph(par.graph_refining);
  }

  mg_levels.push_back(std::make_shared<MG_Level>(fine_graph, par.fe_degree, 0));
  mg_levels[0] -> make_grid();
  
  std::vector<std::map<int, std::vector<int>>> coarse_to_fine_vertex_maps;

  for (int i = 1; i < par.n_v_cycles; i++) {
    std::map<int, std::vector<int>> coarse_to_fine_vertex_map;

    Graph coarse_graph = mg_levels[i-1] -> graph.get_coarser_graph(coarse_to_fine_vertex_map, par.coarsening_percentage);

    mg_levels.push_back(std::make_shared<MG_Level>(coarse_graph, par.fe_degree, i));
    mg_levels[i] -> make_grid();

    coarse_to_fine_vertex_maps.push_back(coarse_to_fine_vertex_map);
  }

  GridOut g_out;

  for (int i = 0; i < mg_levels.size(); i++) {
    std::ofstream out("tria" + std::to_string(i) + ".vtk");

    g_out.write_vtk(mg_levels[i] -> triangulation, out);

    out.close();
  }

  //triangulation = fine_graph.create_graph_triangulation();

  //coarse_triangulation = coarse_graph.create_graph_triangulation();

  /*
  int current_cell = 0;

  std::map<int, int> dof_to_vertex_coarse_map;
  for(const auto & cell : coarse_dof_handler.active_cell_iterators()) {

    for (int i = 0; i < 2; i++) {
      int dof = cell -> vertex_dof_index(i, 0);
      int vertex = coarse_graph.cells[current_cell].vertices[i];

      if (dof_to_vertex_coarse_map.count(dof) == 0)
        dof_to_vertex_coarse_map[dof] = vertex;
    }

    current_cell++;
  }

  current_cell = 0;

  std::map<int, int> vertex_to_dof_fine_map;
  for(const auto & cell : dof_handler.active_cell_iterators()) {
    for (int i = 0; i < 2; i++) {
      int dof = cell -> vertex_dof_index(i, 0);
      int vertex = fine_graph.cells[current_cell].vertices[i];

      if (vertex_to_dof_fine_map.count(vertex) == 0)
        vertex_to_dof_fine_map[vertex] = dof;
    }
    current_cell++;
  }
  */


  for (int i = 0; i < par.n_v_cycles - 1; i++) {
    std::map<int, std::vector<int>> coarse_to_fine_dof_map;

    std::map<int, int> vertex_to_dof_fine_map = mg_levels[i] ->get_vertex_to_dof_map();
    std::map<int, int> dof_to_vertex_coarse_map = mg_levels[i+1] -> get_dof_to_vertex_map();

    //coarse_to_fine_dof_map = dof_to_vertex_coarse_map \circ coarse_to_fine_vertex_map \circ vertex_to_dof_fine_map
    for (const auto & [c_dof, c_vert] : dof_to_vertex_coarse_map) {
      for (const auto & f_vert : coarse_to_fine_vertex_maps[i][c_vert]) {
        coarse_to_fine_dof_map[c_dof].push_back(vertex_to_dof_fine_map[f_vert]);
      }
    }

    coarse_to_fine_dof_maps.push_back(coarse_to_fine_dof_map);
  }

  /*
  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl
            << "   Total number of points: " << triangulation.n_faces()
            << std::endl;
  */
}

void
Poisson::setup_system()
{
  mg_levels[0] -> setup_system(par.exact_solution);

  for (int i = 1; i < mg_levels.size(); i++) {
    mg_levels[i] -> setup_system(Functions::ZeroFunction<3>());
  }

  solution.reinit(mg_levels[0] -> dof_handler.n_dofs());
  no_mg_solution.reinit(mg_levels[0] -> dof_handler.n_dofs());

  system_rhs.reinit(mg_levels[0] -> dof_handler.n_dofs());
}

void
Poisson::assemble_system()
{
  mg_levels[0] -> assemble_system(par.rhs_function, system_rhs);
  
  //Qui non si assembla l'rhs
  for (int i = 1; i < par.n_v_cycles; i++) {
    mg_levels[i] -> assemble_system();
  }
}

void
Poisson::solve()
{
  Timer tim;
  
  tim.start();
  
  MultigridPreconditioner<Vector<double>> my_preconditioner(mg_levels, coarse_to_fine_dof_maps, 
    par.omega, par.coarse_cg_tollerance,
    par.n_pre_smoothing, par.n_post_smoothing);

  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(mg_levels[0] -> system_matrix, solution, system_rhs, my_preconditioner);
  mg_levels[0] -> constraints.distribute(solution);
      
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;  

  std::cout<<" Tempo impiegato multigrid: " << tim.stop() << " secondi.\n";

  tim.restart();

  PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(mg_levels[0] -> system_matrix);

  SolverControl            solver_control_2(1000, 1e-12);
  SolverCG<Vector<double>> solver_2(solver_control_2);

  solver_2.solve(mg_levels[0] -> system_matrix, no_mg_solution, system_rhs, preconditioner);
  mg_levels[0] -> constraints.distribute(no_mg_solution);

  std::cout << "   " << solver_control_2.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;

  std::cout<<" Tempo impiegato no multigrid: " << tim.stop() << " secondi.\n";

}


void
Poisson::output_results(const unsigned int cycle)
{
  data_out.attach_dof_handler(mg_levels[0] -> dof_handler);

  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(no_mg_solution, "no_mg_solution");

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
      std::cout<<"Creazione griglia.\n";
      if (cycle == 0)
        make_grid();
      /*
      else
        {
          triangulation.refine_global();
        }
      */
      std::cout<<"Inizializzazione sistema. \n";

      setup_system();
      std::cout<<"Assemblaggio sistema. \n";

      assemble_system();
      std::cout<<"Risoluzione sistema. \n";

      solve();
      std::cout<<"Scrittura output in corso. \n";

      output_results(cycle);

      par.convergence_table.difference(	mg_levels[0] -> dof_handler, solution, no_mg_solution);

    }
  par.convergence_table.output_table(std::cout);
}