#pragma once

#include "MG_Level.hpp"
#include <boost/serialization/smart_cast.hpp>
#include <cmath>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>

template <typename VectorType>
class MultigridPreconditioner {

  public:
    MultigridPreconditioner(std::vector<std::shared_ptr<MG_Level>> & mg_levels,
        std::vector<std::map<int, int>> & coarse_to_fine_dof_maps, 
        std::vector<std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>>> not_trivial_prolongations_dof,
        double omega, double coarse_cg_tollerance, int n_pre_smoothing, int n_post_smoothing, std::vector<SparseMatrix<double>> & prolongations
        ): 
      mg_levels(mg_levels),
      coarse_to_fine_dof_maps(coarse_to_fine_dof_maps),
      not_trivial_prolongations_dof(not_trivial_prolongations_dof),
      omega(omega), coarse_cg_tollerance(coarse_cg_tollerance),
      n_pre_smoothing(n_pre_smoothing), n_post_smoothing(n_post_smoothing), prolongations(prolongations),
      solver_control(1000, coarse_cg_tollerance), solver(solver_control)
    {
      for (unsigned int i = 0; i < mg_levels.size(); i++) {
        dofs.push_back(mg_levels[i] -> dof_handler.n_dofs());
      }

      preconditioners.resize(mg_levels.size() - 1);


      for (unsigned int i = 0; i < mg_levels.size() - 1; i++) {
        auto a_fine = linear_operator(mg_levels[i] -> system_matrix);
        auto id_fine = identity_operator(a_fine);

        preconditioners[i].initialize(mg_levels[i] -> system_matrix);
        smoothers.push_back(omega * linear_operator(a_fine, preconditioners[i]));
      }


      x.resize(mg_levels.size());
      residual.resize(mg_levels.size());
      rhs.resize(mg_levels.size());
      coarse_grid_corrections.resize(mg_levels.size());
      
      for (unsigned int i = 0; i < mg_levels.size(); i++) {
        x[i].reinit(dofs[i]);
        residual[i].reinit(dofs[i]);
        rhs[i].reinit(dofs[i]);
        coarse_grid_corrections[i].reinit(dofs[i]);
      }

      solver2.initialize(mg_levels[mg_levels.size()-1] -> system_matrix);
      
      
          /*  for (int k = 0; k < 2; k++)
      for (int i = 0; i < dofs[k]; i++) {
        for (int j = 0; j < dofs[k]; j++) {
          std::cout << mg_levels[k] -> system_matrix.el(i,j) << " ";
        }
        std::cout << "\n";
      }

      Vector<double> e1(dofs[0]);
      Vector<double> e2(dofs[1]);

      for (int i = 0; i < dofs[0]; i++) {
        if (i > 0) e1[i-1] = 0;
        e1[i] = 1;

        restrictions[0].vmult(e2, e1);
        std::cout<<e2 << "\n";
      }*/
    }

    void vmult(VectorType &my_dst,
              const VectorType &my_src) const
    {
      my_dst = 0;

      for (unsigned int i = 0; i < mg_levels.size(); i++) {
        x[i] = 0;
        residual[i ]= 0;
        rhs[i] = 0;
        coarse_grid_corrections[i] = 0;
      }

      unsigned int n_of_levels = mg_levels.size();

      rhs[0] = my_src;


      for (unsigned int k = 0; k < n_of_levels - 1; k++) {

        //Pre smoothing
        for (unsigned int i = 0; i < n_pre_smoothing * (n_of_levels-1-k)  ; i++)
        {
          mg_levels[k] -> system_matrix.residual(residual[k], x[k], rhs[k]);

          smoothers[k].vmult_add(x[k], residual[k]);
        }

        mg_levels[k] -> system_matrix.residual(residual[k], x[k], rhs[k]);
        prolongations[k].Tvmult(rhs[k+1], residual[k]);
      }

      //PreconditionSSOR<SparseMatrix<double>> preconditioner;
      //preconditioner.initialize(mg_levels[n_of_levels-1] -> system_matrix);

      //solver.solve(mg_levels[n_of_levels-1] -> system_matrix, x[n_of_levels-1], rhs[n_of_levels-1], preconditioner);

      solver2.vmult(x[n_of_levels-1], rhs[n_of_levels-1]);
      mg_levels[n_of_levels-1] -> constraints.distribute(x[n_of_levels-1]);

      for (int k = n_of_levels - 2; k >= 0; k--) {

        prolongations[k].vmult(coarse_grid_corrections[k], x[k+1]);

        x[k] += coarse_grid_corrections[k];

        //Post smoothing
        for (unsigned int i = 0; i < n_post_smoothing * (n_of_levels-1-k) ; i++)
        {
          // Start with the residual
          mg_levels[k] -> system_matrix.residual(residual[k], x[k], rhs[k]);

          smoothers[k].vmult_add(x[k], residual[k]);
        }
        
      }

      my_dst = x[0];
    }

    private:
      std::vector<std::shared_ptr<MG_Level>> & mg_levels;

      std::vector<std::map<int, int>> & coarse_to_fine_dof_maps;
      
      double omega;
      double coarse_cg_tollerance;

      unsigned int n_pre_smoothing;
      unsigned int n_post_smoothing;

      SolverControl solver_control;
      mutable SolverCG<Vector<double>> solver;

      mutable SparseDirectUMFPACK solver2;

      std::vector<LinearOperator<Vector<double>>> restrictions;

      std::vector<PreconditionSSOR<SparseMatrix<double>>> preconditioners;
      std::vector<LinearOperator<Vector<double>>> smoothers;

      std::vector<std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>>> not_trivial_prolongations_dof;

      mutable std::vector<Vector<double>> x, rhs, residual, coarse_grid_corrections;

      std::vector<unsigned int> dofs;

      std::vector<SparseMatrix<double>> & prolongations;
};