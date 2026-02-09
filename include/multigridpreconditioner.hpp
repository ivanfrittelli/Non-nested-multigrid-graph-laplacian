#pragma once

/*
template <typename VectorType>
class MultigridPreconditioner {

  public:
    MultigridPreconditioner(DoFHandler<1, 3> & coarse_dof_handler, DoFHandler<1, 3> & dof_handler,
        SparseMatrix<double> & system_matrix, SparseMatrix<double> & coarse_system_matrix,
        std::map<int, std::vector<int>> & coarse_to_fine_dof_map,
        double omega, double coarse_cg_tollerance, int n_pre_smoothing, int n_post_smoothing, DataOut<1,3> * data_out, DataOut<1,3> * coarse_data_out, AffineConstraints<double> * coarse_constraints
        ): 
      coarse_dof_handler(coarse_dof_handler),
      dof_handler(dof_handler),
      system_matrix(system_matrix),
      coarse_system_matrix(coarse_system_matrix),
      coarse_to_fine_dof_map(coarse_to_fine_dof_map),
      omega(omega), coarse_cg_tollerance(coarse_cg_tollerance),
      n_pre_smoothing(n_pre_smoothing), n_post_smoothing(n_post_smoothing),
      solver_control(1000, coarse_cg_tollerance), solver(solver_control), data_out(data_out), coarse_data_out(coarse_data_out), coarse_constraints(coarse_constraints)
    {
      preconditioner.initialize(system_matrix);
      preconditioner_coarse.initialize(coarse_system_matrix);

      a_fine = linear_operator(system_matrix);
      auto id_fine = identity_operator(a_fine);

      //Restriction
      restriction = id_fine;
      restriction.vmult = [&](Vector<double> &dst, const Vector<double> &src) {
        dst.reinit(coarse_dof_handler.n_dofs());

        for (const auto& [key, value] : coarse_to_fine_dof_map) {
          //La media del valore dei vertici che sono stati collassati
          double result = 0;
          for (unsigned int i = 0; i < value.size(); i++) {
            result += src[value[i]];
          }
          result /= double(value.size());

          dst[key] = result;
        }
      };
      restriction.reinit_range_vector = [&](Vector<double> &v, bool) {
        v.reinit(coarse_dof_handler.n_dofs());
      };
      restriction.reinit_domain_vector = [&](Vector<double> &v, bool) {
        v.reinit(dof_handler.n_dofs());
      };
      
      //Prolongation
      a_coarse = linear_operator(coarse_system_matrix);
      auto id_coarse = identity_operator(a_coarse);

      prolongation = id_coarse;
      prolongation.vmult = [&](Vector<double> &dst, const Vector<double> &src) {
        dst.reinit(dof_handler.n_dofs());
        dst = 0;
      
        //A ogni vertice collassato associo il valore del vertice nuovo
        for (const auto& [c_dof, f_dof] : coarse_to_fine_dof_map) {
          for (unsigned int i = 0; i < f_dof.size(); i++) {
            dst[f_dof[i]] = src[c_dof]/f_dof.size(); //Trasposta
          }
        }
      };

      prolongation.reinit_range_vector = [&](Vector<double> &v, bool) {
        v.reinit(dof_handler.n_dofs());
      };
      prolongation.reinit_domain_vector = [&](Vector<double> &v, bool) {
        v.reinit(coarse_dof_handler.n_dofs());
      };

      //Smoothing
      auto J = linear_operator(a_fine, preconditioner);

      Pinv = omega * J;

      //a_coarse = restriction*a_fine*prolongation;
      
      residual.reinit(dof_handler.n_dofs());
      coarse_residual.reinit(coarse_dof_handler.n_dofs());
      coarse_error.reinit(coarse_dof_handler.n_dofs());
      coarse_grid_correction.reinit(dof_handler.n_dofs());
    }

    void vmult(VectorType &my_dst,
              const VectorType &my_src) const
    {
      my_dst = 0;

      //Pre smoothing
      for (unsigned int i = 0; i < n_pre_smoothing; i++)
      {
        system_matrix.residual(residual, my_dst, my_src);

        Pinv.vmult_add(my_dst, residual);
      }

      system_matrix.residual(residual, my_dst, my_src);

      //Coarse grid correction
      restriction.vmult(coarse_residual, residual);

      solver.solve(a_coarse, coarse_error, coarse_residual, PreconditionIdentity());
      coarse_constraints -> distribute(coarse_error);

      prolongation.vmult(coarse_grid_correction, coarse_error);

      my_dst += coarse_grid_correction;

      //Post smoothing
      for (unsigned int i = 0; i < n_post_smoothing; i++)
      {
        // Start with the residual
        system_matrix.residual(residual, my_dst, my_src);

        Pinv.vmult_add(my_dst, residual);
      }

    }

    private:
      DoFHandler<1, 3> & coarse_dof_handler;
      DoFHandler<1, 3> & dof_handler;

      SparseMatrix<double> & system_matrix;
      SparseMatrix<double> & coarse_system_matrix;

      std::map<int, std::vector<int>> & coarse_to_fine_dof_map;
      
      double omega;
      double coarse_cg_tollerance;

      unsigned int n_pre_smoothing;
      unsigned int n_post_smoothing;

      SolverControl solver_control;
      mutable SolverCG<Vector<double>> solver;

      PreconditionSSOR<SparseMatrix<double>> preconditioner;

      PreconditionJacobi<SparseMatrix<double>> preconditioner_coarse;

      LinearOperator<Vector<double>> a_fine, a_coarse, prolongation, restriction, Pinv;

      mutable Vector<double> residual, coarse_residual, coarse_error, coarse_grid_correction;

      DataOut<1,3> * data_out;
      DataOut<1,3> * coarse_data_out;

      AffineConstraints<double> * coarse_constraints;

};
*/

#include "MG_Level.hpp"
#include <boost/serialization/smart_cast.hpp>
#include <cmath>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>

template <typename VectorType>
class MultigridPreconditioner {

  public:
    MultigridPreconditioner(std::vector<std::shared_ptr<MG_Level>> & mg_levels,
        std::vector<std::map<int, std::vector<int>>> & coarse_to_fine_dof_maps,
        double omega, double coarse_cg_tollerance, int n_pre_smoothing, int n_post_smoothing
        ): 
      mg_levels(mg_levels),
      coarse_to_fine_dof_maps(coarse_to_fine_dof_maps),
      omega(omega), coarse_cg_tollerance(coarse_cg_tollerance),
      n_pre_smoothing(n_pre_smoothing), n_post_smoothing(n_post_smoothing),
      solver_control(1000, coarse_cg_tollerance), solver(solver_control)
    {

      for (unsigned int i = 0; i < mg_levels.size(); i++) {
        dofs.push_back(mg_levels[i] -> dof_handler.n_dofs());
      }

      preconditioners.resize(mg_levels.size() - 1);

      a_coarses.push_back(linear_operator(mg_levels[0] -> system_matrix));

      for (unsigned int i = 0; i < mg_levels.size() - 1; i++) {
        auto a_fine = linear_operator(mg_levels[i] -> system_matrix);
        auto id_fine = identity_operator(a_fine);

        preconditioners[i].initialize(mg_levels[i] -> system_matrix);

        smoothers.push_back(omega * linear_operator(a_fine, preconditioners[i]));

        //Restriction
        auto restriction = id_fine;
        restriction.vmult = [&, i](Vector<double> &dst, const Vector<double> &src) {
          dst.reinit(dofs[i+1]);
          dst = 0;

          int max = 0;
          
          for (const auto& [key, value] : coarse_to_fine_dof_maps[i]) {
            //La media del valore dei vertici che sono stati collassati
            
            double result = 0;
           
            for (unsigned int j = 0; j < value.size(); j++) {
              result += src[value[j]];
            }

            if (value.size() > max) max = value.size();

            dst[key] = result;    
          }
        };
        restriction.reinit_range_vector = [&](Vector<double> &v, bool) {
          v.reinit(dofs[i+1]);
        };
        restriction.reinit_domain_vector = [&](Vector<double> &v, bool) {
          v.reinit(dofs[i]);
        };
        //Prolongation
        auto a_coarse = linear_operator(mg_levels[i + 1] -> system_matrix);
        auto id_coarse = identity_operator(a_coarse);

        auto prolongation = id_coarse;
        prolongation.vmult = [&, i](Vector<double> &dst, const Vector<double> &src) {
          dst.reinit(dofs[i]);
          dst = 0;
               
          //A ogni vertice collassato associo il valore del vertice nuovo
          for (const auto& [c_dof, f_dof] : coarse_to_fine_dof_maps[i]) {
            
            for (unsigned int j = 0; j < f_dof.size(); j++) {
              dst[f_dof[j]] = src[c_dof]; //Trasposta
            }
          }

        
        };
        prolongation.reinit_range_vector = [&](Vector<double> &v, bool) {
          v.reinit(dofs[i]);
        };
        prolongation.reinit_domain_vector = [&](Vector<double> &v, bool) {
          v.reinit(dofs[i+1]);
        };
        prolongations.push_back(prolongation);
        restrictions.push_back(restriction);

        //a_coarses.push_back(restrictions[i] * a_coarses[i] * prolongations[i]);
      
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
      
      /*
            for (int k = 0; k < 2; k++)
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

      //std::cout<<rhs[0]<<"\n";

      for (unsigned int k = 0; k < n_of_levels - 1; k++) {

        //Pre smoothing
        for (unsigned int i = 0; i < n_pre_smoothing * (n_of_levels-1-k)  ; i++)
        {
          //residual[k] =  rhs[k] - a_coarses[k]*x[k];

          mg_levels[k] -> system_matrix.residual(residual[k], x[k], rhs[k]);

          smoothers[k].vmult_add(x[k], residual[k]);
        }
        //residual[k] =  rhs[k] - a_coarses[k]*x[k];

        mg_levels[k] -> system_matrix.residual(residual[k], x[k], rhs[k]);

        //std::cout<< residual[k] << "\n";

        //Restriction (Il residuo del livello k diventa l'rhs del livello k+1)
        restrictions[k].vmult(rhs[k+1], residual[k]);
        //std::cout<< rhs[k+1] << "\n";
      }

      solver.solve(mg_levels[n_of_levels-1] -> system_matrix, x[n_of_levels-1], rhs[n_of_levels-1], PreconditionIdentity());
      mg_levels[n_of_levels-1] -> constraints.distribute(x[n_of_levels-1]);
      //std::cout<< x[n_of_levels-1] << "\n";
      //std::cout<< rhs[n_of_levels-1] << "\n";


      for (int k = n_of_levels - 2; k >= 0; k--) {

        prolongations[k].vmult(coarse_grid_corrections[k], x[k+1]);

        //std::cout<< coarse_grid_corrections[k] << "\n";

        //mg_levels[k] -> constraints.distribute(x[k]);

        x[k] += coarse_grid_corrections[k];

        //Post smoothing
        for (unsigned int i = 0; i < n_post_smoothing * (n_of_levels-1-k) ; i++)
        {
          // Start with the residual
          //residual[k] =  rhs[k]radius - a_coarses[k]*x[k];
          mg_levels[k] -> system_matrix.residual(residual[k], x[k], rhs[k]);

          smoothers[k].vmult_add(x[k], residual[k]);
        }
        
      }

      my_dst = x[0];
    }

    private:
      std::vector<std::shared_ptr<MG_Level>> & mg_levels;

      std::vector<std::map<int, std::vector<int>>> & coarse_to_fine_dof_maps;
      
      double omega;
      double coarse_cg_tollerance;

      unsigned int n_pre_smoothing;
      unsigned int n_post_smoothing;

      SolverControl solver_control;
      mutable SolverCG<Vector<double>> solver;

      mutable SparseDirectUMFPACK solver2;

      std::vector<LinearOperator<Vector<double>>> prolongations;
      std::vector<LinearOperator<Vector<double>>> restrictions;
      std::vector<LinearOperator<Vector<double>>> a_coarses;

      std::vector<PreconditionJacobi<SparseMatrix<double>>> preconditioners;
      std::vector<LinearOperator<Vector<double>>> smoothers;

      mutable std::vector<Vector<double>> x, rhs, residual, coarse_grid_corrections;

      std::vector<unsigned int> dofs;
};