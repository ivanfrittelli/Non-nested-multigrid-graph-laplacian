#include "../include/MG_Level.hpp"

#include <deal.II/base/numbers.h>
#include <deal.II/fe/fe_values.h>
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
MG_Level::setup_system(const Function<3> & boundary_conditions) {
  std::cout << "   Number of degrees of freedom in level " << id << " : " << dof_handler.n_dofs()
            << std::endl;


  constraints.clear();
      
  for (int i = 0; i < 2; i++)
    for (const auto &cell : triangulation.active_cell_iterators()) {
      if (cell-> face(i) -> at_boundary()) {
        if (inlet_dof >= 0 && cell -> face(i) -> index() == inlet_dof) {
          cell -> face(i) -> set_boundary_id(1);
        }
        else
          cell -> face(i) -> set_boundary_id(0);
      }
    }
      


  std::cout << "   Number of not boundary degrees of freedom in level " << id << " : " << dof_handler.n_dofs() - dof_handler.n_boundary_dofs({0})
            << std::endl;

  VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            boundary_conditions,
                                            constraints);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  
}

void
MG_Level::assemble_system(const TimeInfo & time_info) {
  QGauss<1>     quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices()) {

              if (time_info.dt <= 0) {
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                  fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                  fe_values.JxW(q_index));           // dx
              } else {
                cell_matrix(i, j) += time_info.dt * time_info.theta * 
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                  fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                  fe_values.JxW(q_index));

                cell_matrix(i,j) += fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                  fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                  fe_values.JxW(q_index);
              }
                
            }
              
          }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, local_dof_indices, system_matrix);

    }

}

void 
MG_Level::assemble_rhs(const FunctionParser<3> & rhs_function,
  Vector<double> & rhs, const Vector<double> & old_solution,
  const TimeInfo & time_info,
  double inlet_flow) 
{
  QGauss<1>     quadrature_formula(fe.degree + 1);
  QGauss<0> face_quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<1,3> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);


  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> old_solution_values(fe_values.n_quadrature_points);
  std::vector<Tensor<1,3>> old_solution_gradients(fe_values.n_quadrature_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      fe_values.get_function_values(old_solution, old_solution_values);
      fe_values.get_function_gradients(old_solution, old_solution_gradients);

      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            const auto &x_q = fe_values.quadrature_point(q_index);

            if (time_info.dt <= 0) {
              cell_rhs(i) += //(i == inlet_dof ? 1:0)*
                            (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            rhs_function.value(x_q) *       // f(x_q)
                            fe_values.JxW(q_index));            // dx
              for (const auto &f : cell->face_indices())
                if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
                  {
                    fe_face_values.reinit(cell, f);

                    for (const unsigned int q_index :
                        fe_face_values.quadrature_point_indices())
                      for (const unsigned int i : fe_face_values.dof_indices())
                        cell_rhs(i) += fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                          inlet_flow * // g(x_q)
                          fe_face_values.JxW(q_index);                 // ds
                  }
            }
            else {
              cell_rhs(i) += fe_values.shape_value(i, q_index)
                * old_solution_values[q_index]
                * fe_values.JxW(q_index);

              cell_rhs(i) -= time_info.dt * (1 - time_info.theta)
                            * fe_values.shape_grad(i, q_index)
                            * old_solution_gradients[q_index]
                            * fe_values.JxW(q_index);
              
              cell_rhs(i) += 
                            time_info.dt * (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            rhs_function.value(x_q) *       // f(x_q)
                            fe_values.JxW(q_index));            // dx

              for (const auto &f : cell->face_indices())
                if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
                  {
                    fe_face_values.reinit(cell, f);

                    for (const unsigned int q_index :
                        fe_face_values.quadrature_point_indices())
                      for (const unsigned int i : fe_face_values.dof_indices())
                        cell_rhs(i) += time_info.dt * fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                          inlet_flow * // g(x_q)
                          fe_face_values.JxW(q_index);                 // ds
                  }
            }
            
          }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_rhs, local_dof_indices, rhs);
    }
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