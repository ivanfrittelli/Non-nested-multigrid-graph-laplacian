#include "include/poisson.hpp"

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