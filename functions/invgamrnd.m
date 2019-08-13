%This function takes draws from the inverted gamma density.
%The inputs are the two parameters alpha, and beta, and 
%the function returns a n times m matrix. 
%The inverted gamma density function is
%f(x) propto x^-(alpha + 1) exp( -1/(beta*x) );

function [igdraws] = invgamrnd(alpha,beta,n,m)

tempp = gamm_rnd(n,m,alpha,beta);
igdraws = 1./tempp;
