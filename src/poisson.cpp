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

  radii.reinit(fine_graph.big_cells.size());

  VTKUtils::read_cell_data(par.mesh_file_name, "thickness", radii);

  std::map<int, std::vector<std::pair<int,double>>> tmp;
  //Coarsening iniziale, se non voglio risolvere il problema sul grafo originale ma su uno meno fine 
  for (int i = 0; i < par.initial_coarsening; i++)
    fine_graph = fine_graph.get_coarser_graph(tmp, 0.5);
    
  //Configurazione primo livello
  mg_levels.push_back(std::make_shared<MG_Level>(fine_graph, 0));
  mg_levels[0] -> make_grid();
  
  {
    //Boundary conditions
    std::set<int> dirichlet_cells, neumann_cells;

    for (unsigned int j = 0; j < fine_graph.dirichlet_big_cells.size(); j++) {
      auto big_cell = fine_graph.big_cells[fine_graph.dirichlet_big_cells[j]];
      if (fine_graph.adiacency[big_cell.node1].size() == 1) {
        dirichlet_cells.insert(fine_graph.adiacency[big_cell.node1][0]);
      }
      if (fine_graph.adiacency[big_cell.node2].size() == 1) {
        dirichlet_cells.insert(fine_graph.adiacency[big_cell.node2][0]);
      }
    }

    for (unsigned int j = 0; j < fine_graph.neumann_big_cells.size(); j++) {
      auto big_cell = fine_graph.big_cells[fine_graph.neumann_big_cells[j]];
      if (fine_graph.adiacency[big_cell.node1].size() == 1) {
        neumann_cells.insert(fine_graph.adiacency[big_cell.node1][0]);
      }
      if (fine_graph.adiacency[big_cell.node2].size() == 1) {
        neumann_cells.insert(fine_graph.adiacency[big_cell.node2][0]);
      }
    }

    mg_levels[0] -> setup_boundary_conditions(dirichlet_cells, neumann_cells);
  }
  

  double treshold = 10;

  restriction_dynamic_patterns.resize(par.n_v_cycles - 1);
  restriction_sparsity_patterns.resize(par.n_v_cycles - 1);
  restrictions.resize(par.n_v_cycles - 1);

  //Creazione altri livelli
  for (unsigned int i = 1; i < par.n_v_cycles; i++) {
    std::map<int, std::vector<std::pair<int,double>>> vertex_restriction_map;

    //Grafo coarse
    Graph my_coarse_graph = mg_levels[i-1] -> graph.get_coarser_graph(vertex_restriction_map, treshold);

    //Boundary conditions
    std::set<int> dirichlet_cells, neumann_cells;
    {
      for (unsigned int j = 0; j < my_coarse_graph.dirichlet_big_cells.size(); j++) {
        auto big_cell = my_coarse_graph.big_cells[my_coarse_graph.dirichlet_big_cells[j]];
        if (my_coarse_graph.adiacency[big_cell.node1].size() == 1) {
          dirichlet_cells.insert(my_coarse_graph.adiacency[big_cell.node1][0]);
        }
        if (my_coarse_graph.adiacency[big_cell.node2].size() == 1) {
          dirichlet_cells.insert(my_coarse_graph.adiacency[big_cell.node2][0]);
        }
      }

      for (unsigned int j = 0; j < my_coarse_graph.neumann_big_cells.size(); j++) {
        auto big_cell = my_coarse_graph.big_cells[my_coarse_graph.neumann_big_cells[j]];
        if (my_coarse_graph.adiacency[big_cell.node1].size() == 1) {
          neumann_cells.insert(my_coarse_graph.adiacency[big_cell.node1][0]);
        }
        if (my_coarse_graph.adiacency[big_cell.node2].size() == 1) {
          neumann_cells.insert(my_coarse_graph.adiacency[big_cell.node2][0]);
        }
      }
    }

    //Setup mg level
    mg_levels.push_back(std::make_shared<MG_Level>(my_coarse_graph, i));
    mg_levels[i] -> make_grid();
    mg_levels[i] -> setup_boundary_conditions(dirichlet_cells, neumann_cells);

    //Setup mappa di restriction
    std::map<int, int> vertex_to_dof_fine_map = mg_levels[i-1] ->get_vertex_to_dof_map();
    std::map<int, int> vertex_to_dof_coarse_map = mg_levels[i] -> get_vertex_to_dof_map();

    //Setup sparsity pattern dinamico della restriction
    restriction_dynamic_patterns[i-1].reinit(mg_levels[i] -> dof_handler.n_dofs(), mg_levels[i-1] -> dof_handler.n_dofs());
    for (const auto & [fine_dof, coarse_dofs_map] : vertex_restriction_map) {
      for (const auto & [coarse_dof, weight] : coarse_dofs_map) {
        restriction_dynamic_patterns[i-1].add(vertex_to_dof_coarse_map[coarse_dof], vertex_to_dof_fine_map[fine_dof]);
      }
    }
    restriction_sparsity_patterns[i-1].copy_from(restriction_dynamic_patterns[i-1]);

    //Scrivo la matrice restriction
    restrictions[i-1].reinit(restriction_sparsity_patterns[i-1]);
    for (const auto & [fine_dof, coarse_dofs_map] : vertex_restriction_map) {
      for (const auto & [coarse_dof, weight] : coarse_dofs_map) {
        restrictions[i-1].set(vertex_to_dof_coarse_map[coarse_dof], vertex_to_dof_fine_map[fine_dof], weight);
      }
    }

    /*
    std::ofstream out("re.txt");
    
    if (i > 1)
    for (int k = 0; k < mg_levels[i] -> dof_handler.n_dofs(); k++) {
      for (int j = 0; j < mg_levels[i-1] -> dof_handler.n_dofs(); j++) {
        out<<restrictions[i-1].el(k,j) << " ";
      }
      //out<<"||";
      for (int j = 0; j < 3; j++) {
        //out<<my_coarse_graph.points[k][j]<<" ";
      }
      out<<"\n";
    }
    out.close();
    */
  }
    

  //Faccio un vtk delle triangolazioni
  for (unsigned int i = 0; i < mg_levels.size(); i++) {
    std::ofstream out("tria" + std::to_string(i) + ".vtk");

    g_out.write_vtk(mg_levels[i] -> triangulation, out);

    out.close();
  }
 
  /*
  {
    std::map<int, std::vector<int>> classic_coarse_to_fine_vertex_map;
    std::map<int, int> classic_coarse_to_fine_cell_map;
    double classic_length_treshold = 0.4;

    Graph classic_coarse_graph = fine_graph.get_classic_coarser_graph(classic_coarse_to_fine_vertex_map, 
      classic_coarse_to_fine_cell_map,classic_length_treshold);

    auto classic_mg_level = std::make_shared<MG_Level>(classic_coarse_graph, -1);
    classic_mg_level -> make_grid();

    std::cout<<"Punti scartati: " << fine_graph.points.size() - classic_coarse_graph.points.size() <<"\n";

    std::ofstream out("my_tria.vtk");
    g_out.write_vtk(classic_mg_level -> triangulation, out);
    out.close();
  }
  */
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
Poisson::solve()
{
  Timer tim;
    
  TimeInfo time_info;
  time_info.dt = par.time_step_length;
  time_info.is_time_dependent = par.is_time_dependent;

  //No mg solution compute only fist step (for time dependent problem).
  if (par.compute_no_mg_solution) {
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(mg_levels[0] -> system_matrix);

    SolverControl            solver_control_2(10000, 1e-12);
    SolverCG<Vector<double>> solver_2(solver_control_2);

    //std::cout<<system_rhs.linfty_norm() <<"\n";

    //mg_levels[0] -> constraints.print(std::cout);

    mg_levels[0] -> assemble_system_and_rhs(par.reaction_term, par.rhs_function, system_rhs, par.neumann_function, solution, time_info, radii);

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
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
  
    MultigridPreconditioner<Vector<double>> my_preconditioner(mg_levels, coarse_to_fine_dof_maps, 
      par.omega,
      par.n_pre_smoothing, par.n_post_smoothing, restrictions);

      
      int i = 0;
      
      /*
      for (int j = 0; j < system_rhs.size(); j++) {
        for (int k = 0; k < system_rhs.size(); k++) {
          std::cout<<mg_levels[0] -> system_matrix.el(j,k) << " ";
        }
        std::cout<<"\n";
      }
      */
      
      //Con il ciclo scritto così, se non è time dependent, esegue solo uno step, come giusto che sia
      do {
        mg_levels[0] -> assemble_system_and_rhs(par.reaction_term, par.rhs_function, system_rhs, par.neumann_function, solution, time_info, radii);

        //Nei sottolivelli non devo assemblare l rhs
        for (unsigned int j = 1; j < par.n_v_cycles; j++) {
          mg_levels[j] -> assemble_system(par.reaction_term, time_info, radii);
        }
        my_preconditioner.setup_coarse_grid_solver();

        par.rhs_function.advance_time(time_info.dt);
        par.neumann_function.advance_time(time_info.dt);
        par.reaction_term.advance_time(time_info.dt);

        tim.restart();
        solver.solve(mg_levels[0] -> system_matrix, solution, system_rhs, my_preconditioner);
        mg_levels[0] -> constraints.distribute(solution);
        std::cout<<" Tempo impiegato risoluzione";
        if(time_info.is_time_dependent) 
          std::cout<< " time-step";

        std::cout <<":"<< tim.stop() << " secondi.\n";

        output_results(i);

        i++;
      } while(i < par.time_steps && par.is_time_dependent);

    /*
    Vector<double> residue(mg_levels[0] -> dof_handler.n_dofs());
    Vector<double> correction(mg_levels[0] -> dof_handler.n_dofs());

    solution = 0;
    mg_levels[0] -> assemble_rhs(par.rhs_function, system_rhs, par.neumann_function, solution, time_info, radii);
    for (int k = 0; k < 100; k++) {
      mg_levels[0]-> system_matrix.residual(residue, solution, system_rhs);


      double res = residue.l2_norm();

      std::cout<<"All'iterazione " << k << ", il residuo è: " << residue.l2_norm() << "\n";

      if (res < 1e-11) break;

      my_preconditioner.vmult(correction, residue);
      mg_levels[0] -> constraints.distribute(correction);

      solution +=correction;

      //par.convergence_table.difference(mg_levels[0] -> dof_handler, solution, no_mg_solution);
    }
     
    //output_results(0);
    */
    

    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;  

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

  std::cout<<"Risoluzione sistema. \n";

  solve();
    
  //par.convergence_table.output_table(std::cout);
}