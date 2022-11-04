# include <RcppArmadillo.h>
# include "projections.hpp"
# include <limits.h>
# include <math.h>
using namespace Rcpp;
using namespace arma;

// [[ Rcpp :: depends ( RcppArmadillo )]]


// [[Rcpp::export]]
arma::colvec soft(arma::colvec x, double T){
  //return sign(x);
  return sign(x)%clamp(abs(x)-T,0.0,LONG_MAX);
}

// [[Rcpp::export]]
arma::colvec proj_simplex_l0(arma::colvec x, int s, arma::colvec D){
  int p = x.n_elem;
  uvec order_D = sort_index(D);
  uvec nonzero_index = order_D(span(0,s-1));
  colvec D_active_inv = 1/D(nonzero_index);
  colvec x_proj(p, fill::zeros);
  x_proj(nonzero_index) = D_active_inv/sum(D_active_inv);
  return x_proj;
}

// [[Rcpp::export]]
arma::colvec proj_simplex_l1(arma::colvec x, double lambda, arma::colvec D){
  int p = x.n_elem;
  double alpha_max = 1000;
  double lower = 0, upper = alpha_max;
  double alpha;
  double diff = LONG_MAX, diff_upper;
  while(upper-lower>=1e-7){
    alpha = (lower+upper)/2;
    diff = sum(soft(alpha/D,lambda))-2;
    if(diff==0){
      break;
    }else{
      diff_upper = sum(soft(upper/D,lambda))-2;
      if(diff_upper*diff<0){
        lower = alpha;
      }else{
        upper = alpha;
      }
    }
  } 
  return 0.5*soft(alpha/D,lambda);
}



// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
soft(c(1,-2),1.5)
set.seed(1)
proj_simplex_l0(rnorm(10),3,rgamma(10,1))
set.seed(1)
proj_simplex_l1(rnorm(10),0.15,rgamma(10,1))
*/
