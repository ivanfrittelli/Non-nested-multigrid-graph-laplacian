#include "../include/MG_Level.hpp"

#include <iostream>

void
MG_Level::set_inlet_dof(int i_dof) {
  inlet_dof = i_dof;
}

void
MG_Level::make_grid() {
  triangulation = graph.create_graph_triangulation(); 
  dof_handler.distribute_dofs(fe);
}

void
MG_Level::setup_system(const Function<3> & boundary_conditions, double inlet_pressure) {
  std::cout << "   Number of degrees of freedom in level " << id << " : " << dof_handler.n_dofs()
            << std::endl;


  constraints.clear();
      
  for (int i = 0; i < 2; i++)
    for (const auto &cell : triangulation.active_cell_iterators()) 
      if (cell-> face(i) -> at_boundary()) {
        cell -> face(i) -> set_boundary_id(0);
      }


  std::cout << "   Number of not boundary degrees of freedom in level " << id << " : " << dof_handler.n_dofs() - dof_handler.n_boundary_dofs({0})
            << std::endl;

  VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            boundary_conditions,
                                            constraints);

  constraints.add_line(inlet_dof);
  constraints.set_inhomogeneity(inlet_dof, inlet_pressure);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  
}

void
MG_Level::assemble_system() {
  QGauss<1>     quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  int current_cell = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (cell -> diameter() / resistances[current_cell]) *
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
          }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, local_dof_indices, system_matrix);

      current_cell++;
    }

    /*
        for (int i = 0; i < dof_handler.n_dofs(); i++) {
      for (int j = 0; j < dof_handler.n_dofs(); j++) {
        std::cout<<system_matrix.el(i,j) << " ";
      }
      std::cout<<"\n";
    }
      */
}

void
MG_Level::assemble_system(const FunctionParser<3> & rhs_function, Vector<double> & rhs) {
  QGauss<1>     quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  int current_cell = 0;
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
                (cell -> diameter() / resistances[current_cell]) *
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (i == inlet_dof ? 1:0)*
                            (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            rhs_function.value(x_q) *       // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, rhs);

      current_cell++;

    }
/*
    for (int i = 0; i < dof_handler.n_dofs(); i++) {
      for (int j = 0; j < dof_handler.n_dofs(); j++) {
        std::cout<<system_matrix.el(i,j) << " ";
      }
      std::cout<<"\n";
    }*/

}

std::map<int, int>
MG_Level::get_dof_to_vertex_map() {
  int current_cell = 0;

  std::map<int, int> dof_to_vertex_map;
  for(const auto & cell : dof_handler.active_cell_iterators()) {

    for (int i = 0; i < 2; i++) {

      int dof = cell -> vertex_dof_index(i, 0);
      int vertex = graph.cells[current_cell].vertices[i];

      if (dof_to_vertex_map.count(dof) == 0)
        dof_to_vertex_map[dof] = vertex;
    }

    current_cell++;
  }

  return dof_to_vertex_map;
}

std::map<int, int>
MG_Level::get_vertex_to_dof_map() {
  int current_cell = 0;

  std::map<int, int> vertex_to_dof_map;
  for(const auto & cell : dof_handler.active_cell_iterators()) {
    for (int i = 0; i < 2; i++) {
      int dof = cell -> vertex_dof_index(i, 0);
      int vertex = graph.cells[current_cell].vertices[i];

      if (vertex_to_dof_map.count(vertex) == 0)
        vertex_to_dof_map[vertex] = dof;
    }
    current_cell++;
  }
  return vertex_to_dof_map;

}