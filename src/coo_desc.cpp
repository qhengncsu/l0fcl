# include <RcppArmadillo.h>
# include "projections.hpp"
# include <string>
# include <iostream>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List coo_desc(arma::mat X, arma::mat M, arma::colvec w, double gamma, int s=10, double lambda=0.15,
                   int k=5, int max_iter=100, int update_phi_every = 1, int w_sparse=0){
  int n = X.n_rows, p = X.n_cols;
  colvec D(p,fill::zeros),phi_ij,phi_ji,M_jl;
  mat dist(n,n,fill::zeros);
  Mat<unsigned int> nn(k,n,fill::zeros);
  mat phi(n,n,fill::zeros);
  uvec index, i_vec, l_vec;
  double fit_ssq;
  for(int iter=0;iter<100;iter++){
    if(iter%update_phi_every==0){
      dist.zeros();
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(w_sparse==0){
            dist(i,j) = sum(pow(X.row(i)-X.row(j),2).t()%square(w));
          }else{
            dist(i,j) = sum(pow(X.row(i)-X.row(j),2).t()%(square(w)+lambda*w));
          }
        }
      }
      nn.zeros();
      for(int i=0;i<n;i++){
        index = sort_index(dist.col(i));
        nn.col(i) = index(span(1,k));
      }
      phi.zeros();
      for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
          phi(i,nn(j,i)) = exp(-0.5*dist(i,nn(j,i))/p);
        }
      }
    }
    for(int i=0;i<n;i++){
      index = nn.col(i);
      i_vec = i;
      phi_ij = phi(i_vec,index).as_col();
      phi_ji = phi(index,i_vec).as_col();
      for(int l=0;l<p;l++){
        l_vec = l;
        M_jl = M(index,l_vec).as_col();
        if(w_sparse==0){
          M(i,l) = (gamma*(sum(phi_ij%M_jl)+sum(phi_ji%M_jl))+pow(w(l),2)*X(i,l))/((sum(phi_ij)+sum(phi_ji))*gamma+pow(w(l),2));
        }else{
          M(i,l) = (gamma*(sum(phi_ij%M_jl)+sum(phi_ji%M_jl))+(pow(w(l),2)+lambda*w(l))*X(i,l))/((sum(phi_ij)+sum(phi_ji))*gamma+pow(w(l),2)+lambda*w(l));
        }
      }
    }
    for(int l=0;l<p;l++){
      D(l) = sum(square(X.col(l)-M.col(l)));
      if(D(l)==0){
        throw std::runtime_error("D_l = 0!");
      }
    }
    if(w_sparse==0){
      w = proj_simplex_l0(w,s,D);
    }else{
      w = proj_simplex_l1(w,lambda,D);
    }
  }
  if(w_sparse==0){
    fit_ssq = sum(D%square(w));
  }else{
    fit_ssq = sum(D%(square(w))+lambda*w);
  }
  //for(int i=0;i<n;i++){
    //for(int j=0;j<n;j++){
      //f_optimal += gamma*phi(i,j)*sum(square(M.row(i)-M.row(j)));
    //}
  //}
  List L = List::create(M, w, fit_ssq);
  return L;
}


