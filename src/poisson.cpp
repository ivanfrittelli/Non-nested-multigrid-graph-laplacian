#include "../include/poisson.hpp"
#include "../include/vtk_utils.h"

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
  GridOut g_out;

  Graph fine_graph;

  VTKUtils::read_vtk_graph(par.mesh_file_name, fine_graph, par.only_one);
  VTKUtils::read_cell_label_graph(par.mesh_file_name, fine_graph, "labels");

  std::map<int, int> coarse_to_fine_vertex_mapl;
  std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>> not_trivial_prolongationl;

  for (int i = 0; i < par.initial_coarsening; i++)
    fine_graph = fine_graph.get_coarser_graph_lort(coarse_to_fine_vertex_mapl, not_trivial_prolongationl);

  mg_levels.push_back(std::make_shared<MG_Level>(fine_graph, 0));
  mg_levels[0] -> make_grid();

  std::vector<std::map<int, int>> coarse_to_fine_vertex_maps;
  std::vector<std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>>> not_trivial_prolongations;

  //Creazione altri livelli
  for (unsigned int i = 1; i < par.n_v_cycles; i++) {
    std::map<int, int> coarse_to_fine_vertex_map_lort;
    std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>> not_trivial_prolongation_lort;

    Timer tim;
    tim.start();

    Graph my_coarse_graph = mg_levels[i-1] -> graph.get_coarser_graph_lort(coarse_to_fine_vertex_map_lort, not_trivial_prolongation_lort);

    std::cout<<tim.stop() <<"\n";

    mg_levels.push_back(std::make_shared<MG_Level>(my_coarse_graph, i));
    mg_levels[i] -> make_grid();

    coarse_to_fine_vertex_maps.push_back(coarse_to_fine_vertex_map_lort);
    not_trivial_prolongations.push_back(not_trivial_prolongation_lort);
  }

  //Impostazioni per mappa restriction e prolongation
  for (unsigned int i = 0; i < par.n_v_cycles - 1; i++) {
    std::ofstream oout("AAA.txt");

    //Imposto la coarse to fine dof map
    std::map<int, int> coarse_to_fine_dof_map;
    std::map<int, int> vertex_to_dof_fine_map = mg_levels[i] ->get_vertex_to_dof_map();
    std::map<int, int> dof_to_vertex_coarse_map = mg_levels[i+1] -> get_dof_to_vertex_map();

    //mg_levels[i] -> set_inlet_dof(i == 0 ? vertex_to_dof_fine_map[0] : -1);

    //coarse_to_fine_dof_map = dof_to_vertex_coarse_map \circ coarse_to_fine_vertex_map \circ vertex_to_dof_fine_map
    for (const auto & [c_dof, c_vert] : dof_to_vertex_coarse_map) {
      int f_vert = (coarse_to_fine_vertex_maps[i])[c_vert];
      coarse_to_fine_dof_map[c_dof] = vertex_to_dof_fine_map[f_vert];

      oout << c_dof << " " << vertex_to_dof_fine_map[f_vert] << "\n";
    }

    coarse_to_fine_dof_maps.push_back(coarse_to_fine_dof_map);

    //Imposto le prolongation non banali
    std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>> not_trivial_prolongation_dof;

    std::map<int, int> vertex_to_dof_coarse_map = mg_levels[i+1] ->get_vertex_to_dof_map();

    for (const auto & [key, pair] : not_trivial_prolongations[i]) {
      std::pair<int, double> left_dof_pair;
      left_dof_pair.first = vertex_to_dof_coarse_map[pair.first.first];
      left_dof_pair.second = pair.first.second;

      std::pair<int, double> right_dof_pair;
      right_dof_pair.first = vertex_to_dof_coarse_map[pair.second.first];
      right_dof_pair.second = pair.second.second;

      not_trivial_prolongation_dof[vertex_to_dof_fine_map[key]] = std::make_pair(left_dof_pair, right_dof_pair);
    }

    not_trivial_prolongations_dof.push_back(not_trivial_prolongation_dof);
  }

  //Boundary conditions
  for (int i = 0; i < par.n_v_cycles; i++) {
    std::set<int> dirichlet_cells, neumann_cells;

    Graph * graph = &mg_levels[i] -> graph;

    for (unsigned int j = 0; j < graph ->dirichlet_big_cells.size(); j++) {
      auto big_cell = graph ->big_cells[graph ->dirichlet_big_cells[j]];
      if (graph ->adiacency[big_cell.node1].size() == 1) {
        dirichlet_cells.insert(graph ->adiacency[big_cell.node1][0]);
      }
      if (graph ->adiacency[big_cell.node2].size() == 1) {
        dirichlet_cells.insert(graph ->adiacency[big_cell.node2][0]);
      }
    }

    for (unsigned int j = 0; j < graph ->neumann_big_cells.size(); j++) {
      auto big_cell = graph ->big_cells[graph ->neumann_big_cells[j]];
      if (graph ->adiacency[big_cell.node1].size() == 1) {
        neumann_cells.insert(graph ->adiacency[big_cell.node1][0]);
      }
      if (graph ->adiacency[big_cell.node2].size() == 1) {
        neumann_cells.insert(graph ->adiacency[big_cell.node2][0]);
      }
    }
    
    mg_levels[i] -> setup_boundary_conditions(dirichlet_cells, neumann_cells);
  }
  

  prolongation_dynamic_patterns.resize(par.n_v_cycles - 1);
  prolongation_sparsity_patterns.resize(par.n_v_cycles - 1);
  prolongations.resize(par.n_v_cycles - 1);

  for (unsigned int i = 0; i < par.n_v_cycles - 1; i++) {
    prolongation_dynamic_patterns[i].reinit(mg_levels[i] -> dof_handler.n_dofs(), mg_levels[i+1] -> dof_handler.n_dofs());

    for (const auto [key, value] : coarse_to_fine_dof_maps[i]) {
      prolongation_dynamic_patterns[i].add(value, key);
    }

    for (const auto & [key, pair] : not_trivial_prolongations[i]) {
      prolongation_dynamic_patterns[i].add(key, pair.first.first);
      prolongation_dynamic_patterns[i].add(key, pair.second.first);
    }

    prolongation_sparsity_patterns[i].copy_from(prolongation_dynamic_patterns[i]);
    prolongations[i].reinit(prolongation_sparsity_patterns[i]);

    for (const auto [key, value] : coarse_to_fine_dof_maps[i]) {
      prolongations[i].set(value, key, 1);
    }

    for (const auto & [key, pair] : not_trivial_prolongations[i]) {
      prolongations[i].set(key, pair.first.first, pair.first.second);
      prolongations[i].set(key, pair.second.first, pair.second.second);
    }
  }


  //std::map<int, int> vertex_to_dof_fine_map = mg_levels[par.n_v_cycles-1] ->get_vertex_to_dof_map();
  //mg_levels[par.n_v_cycles-1] -> set_inlet_dof(mg_levels.size() == 1 ? vertex_to_dof_fine_map[0] : -1);

  for (unsigned int i = 0; i < mg_levels.size(); i++) {
    std::ofstream out("tria" + std::to_string(i) + ".vtk");

    g_out.write_vtk(mg_levels[i] -> triangulation, out);

    out.close();
  }
}

void
Poisson::setup_system()
{
  mg_levels[0] -> setup_system(par.exact_solution);

  for (unsigned int i = 1; i < mg_levels.size(); i++) {
    mg_levels[i] -> setup_system(Functions::ZeroFunction<3>());
  }

  solution.reinit(mg_levels[0] -> dof_handler.n_dofs());
  no_mg_solution.reinit(mg_levels[0] -> dof_handler.n_dofs());

  VectorTools::interpolate(mg_levels[0] -> dof_handler, 
                        par.initial_solution, 
                        solution);

  system_rhs.reinit(mg_levels[0] -> dof_handler.n_dofs());
}

void
Poisson::assemble_system()
{
  TimeInfo time_info;
  time_info.dt = par.time_step_length;
  time_info.theta = par.theta;

  for (unsigned int i = 0; i < par.n_v_cycles; i++) {
    mg_levels[i] -> assemble_system(time_info);
  }
}

void
Poisson::solve()
{
  Timer tim;
    
  TimeInfo time_info;
  time_info.dt = par.time_step_length;
  time_info.theta = par.theta;

  //No mg solution compute only fist step (for time dependent problem).
  if (par.compute_no_mg_solution) {
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(mg_levels[0] -> system_matrix);

    SolverControl            solver_control_2(10000, 1e-12);
    SolverCG<Vector<double>> solver_2(solver_control_2);

    //std::cout<<system_rhs.linfty_norm() <<"\n";

    //mg_levels[0] -> constraints.print(std::cout);

    mg_levels[0] -> assemble_rhs(par.rhs_function, system_rhs, par.neumann_function, solution, time_info);

    solver_2.solve(mg_levels[0] -> system_matrix, no_mg_solution, system_rhs, preconditioner);
    mg_levels[0] -> constraints.distribute(no_mg_solution);

    std::cout << "   " << solver_control_2.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;

    std::cout<<" Tempo impiegato no multigrid: " << tim.stop() << " secondi.\n";

    //Altrimenti si stampa dopo insieme alla soluzione con mg
    if (par.n_v_cycles <= 1)
      output_results(0);
  }

  if (par.n_v_cycles > 1) { 
    tim.restart();

    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
  
    MultigridPreconditioner<Vector<double>> my_preconditioner(mg_levels, coarse_to_fine_dof_maps, 
      not_trivial_prolongations_dof,
      par.omega,
      par.n_pre_smoothing, par.n_post_smoothing, prolongations);

      int i = 0;

      //Con il ciclo scritto così, se non è time dependent esegue solo uno step, come giusto che sia
      do {
        mg_levels[0] -> assemble_rhs(par.rhs_function, system_rhs, par.neumann_function,solution, time_info);

        solver.solve(mg_levels[0] -> system_matrix, solution, system_rhs, my_preconditioner);
        mg_levels[0] -> constraints.distribute(solution);

        output_results(i);

        i++;
      } while(i < par.time_steps && par.is_time_dependent);
    
    /*
    Vector<double> residue(mg_levels[0] -> dof_handler.n_dofs());
    Vector<double> correction(mg_levels[0] -> dof_handler.n_dofs());

    solution = 0;
    for (int i = 0; i < par.n_multigrid_it; i++) {
      mg_levels[0]->system_matrix.residual(residue, solution, system_rhs);

      double res = residue.l2_norm();

      std::cout<<"All'iterazione " << i << ", il residuo è: " << residue.l2_norm() << "\n";

      if (res < 1e-12) break;

      my_preconditioner.vmult(correction, residue);
      mg_levels[0] -> constraints.distribute(correction);

      solution +=correction;

      //par.convergence_table.difference(mg_levels[0] -> dof_handler, solution, no_mg_solution);
    }
      */

        
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;  

    std::cout<<" Tempo impiegato multigrid: " << tim.stop() << " secondi.\n";

  }  
}


void
Poisson::output_results(const unsigned int cycle)
{
  DataOut<1,3> data_out;

  data_out.attach_dof_handler(mg_levels[0] -> dof_handler);

  if (par.n_v_cycles > 1) data_out.add_data_vector(solution, "solution", DataOut_DoFData<1,1,3,3>::DataVectorType::type_dof_data);
  if (par.compute_no_mg_solution) data_out.add_data_vector(no_mg_solution, "no_mg_solution", DataOut_DoFData<1,1,3,3>::DataVectorType::type_dof_data);

  data_out.build_patches();

  auto fname =
    "solution_" + std::to_string(cycle) + ".vtu";

  std::ofstream output(fname);
  data_out.write_vtu(output);

}

void
Poisson::run()
{
  std::cout<<"Creazione griglia.\n";
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
    
  //par.convergence_table.output_table(std::cout);
}