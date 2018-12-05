#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>

/* returns the number pi */
double pi() {
    return 3.14159265358979311599796346854;
}

/* errorfunction */
double erf(double x) {
    double t = 1/(1+0.3275911*fabs(x));
    return x/fabs(x)* 
            (1-(0.254829592*t - 0.284496736*t*t + 1.421413741*t*t*t -
            1.453152027*t*t*t*t + 1.061405429*t*t*t*t*t)*exp(-x*x));
}

/* errorfunction with a different approximation */
double erf_v2(double x) {
    double t = 1/(1+0.5*fabs(x));
    return x/fabs(x)*(1-t*exp(-x*x-1.26551223+1.00002368*t+0.37409196*t*t+0.09678418*t*t*t-
        0.18628806*t*t*t*t+0.27886807*t*t*t*t*t-1.13520398*t*t*t*t*t*t+1.48851587*t*t*t*t*t*t*t-
        0.82215223*t*t*t*t*t*t*t*t+0.17087277*t*t*t*t*t*t*t*t*t));
}

/* returns the standard deviation for a step in sigma, which depends on the value for sigma itself */
double sigmaSigma(double x, double sigma_sigma) {
    return x > sigma_sigma ? sigma_sigma : x;
}

/* this function makes shure that log(p^N) = N*log(p) gives the correct value if N = p = 0 according to the statictics (should be 0) */
double prx_step(double p, double N) {
    double val = N*log(p);
    
    if(mxIsFinite(val)) {
        return val;
    }
    return 0;
}

/* For Gamma error:
   computes the lambda moments for gamma fluctuations */
void compLambdaN(double lambda0, double xi, double xe, double sigma, double *lambda_n) {
    double c = -(1-(1-xi)*(1-xe));
    double fac = lambda0*lambda0/(sigma*sigma);
    
    if (sigma < 2e-9) {
        lambda_n[0] = exp(c*lambda0);
        lambda_n[1] = lambda0*lambda_n[0];
        lambda_n[2] = lambda0*lambda_n[1];
        lambda_n[3] = lambda0*lambda_n[2];
        return;
    }
     
    /* approximation of value if "fac" gets too big and numerical errors would occur */
    if (fac < 1e9) {
        lambda_n[0] = exp(-lambda0*lambda0/(sigma*sigma)*log(1-c*sigma*sigma/lambda0));
    } else {
        lambda_n[0] = exp(c*lambda0);
    }
    lambda_n[1] = lambda_n[0]*(lambda0*lambda0/(sigma*sigma))/(lambda0/(sigma*sigma)-c);
    lambda_n[2] = lambda_n[1]*(lambda0*lambda0/(sigma*sigma)+1)/(lambda0/(sigma*sigma)-c);
    lambda_n[3] = lambda_n[2]*(lambda0*lambda0/(sigma*sigma)+2)/(lambda0/(sigma*sigma)-c);
    
    
}

/* For Gaussian Error
   computes the gamma moments if Gaussian fluctuations are present */
/*
void compLambdaN(double lambda0, double xi, double xe, double sigma, double *lambda_n) {
    double Z;
    double c = -(1-(1-xi)*(1-xe));
    double x = (1+c*lambda0*sigma*sigma)/(sqrt(2)*sigma);
    double t = 1/(1+0.5*fabs(x));
    
    /* compute the norm of the distribution */
    /*Z =0.5*(1 + erf_v2(1/(sqrt(2)*sigma)));

    /* <lamda^0> is rewritten for numerical stability for high sigma! */
    /*lambda_n[0] = 0.5*exp(lambda0*c) * ((x/fabs(x) > 0 ? 2*exp(0.5*c*c*lambda0*lambda0*sigma*sigma) : 0) - 
            x/fabs(x)*exp(log(t)-x*x-1.26551223+1.00002368*t+0.37409196*t*t+0.09678418*t*t*t-
        0.18628806*t*t*t*t+0.27886807*t*t*t*t*t-1.13520398*t*t*t*t*t*t+1.48851587*t*t*t*t*t*t*t-
        0.82215223*t*t*t*t*t*t*t*t+0.17087277*t*t*t*t*t*t*t*t*t+0.5*c*c*lambda0*lambda0*sigma*sigma))/Z;
    
    lambda_n[1] = lambda0*(1+lambda0*sigma*sigma*c)*lambda_n[0] + 
            lambda0*sigma/(Z*sqrt(2*pi()))*exp(-1/(2*sigma*sigma));
    lambda_n[2] = lambda0*(1+lambda0*sigma*sigma*c)*lambda_n[1] +   sigma*sigma*lambda0*lambda0*lambda_n[0];
    lambda_n[3] = lambda0*(1+lambda0*sigma*sigma*c)*lambda_n[2] + 2*sigma*sigma*lambda0*lambda0*lambda_n[1];
}
*/


/* this function computes the <lambda_beta> moments */
void compLambdaN_beta(double *lambda_n_beta, double *lambda_n_1, double *lambda_n_2) {
    int i=0, j=0, k=0;
    double binom;
    
    for(i = 0; i<4; i++) {
        lambda_n_beta[i] = 0;
        for(j=0; j <= i; j++) {
            binom = 1;
            for(k = 0; k < j; k++) {
                binom = binom*(i-k)/(k+1);
            }
            lambda_n_beta[i] = lambda_n_beta[i] + binom*lambda_n_1[j]*lambda_n_2[i-j];
        }
    }
}

/* compute the probability for this set of parameters (pi) given the data*/
double prx(double *lambda_n, double xi, double xe, double Nr, double N00, double N01, double N02, double N10, double N11, double N20, double N03, double N12, double N21, double N30) {
   double p00 = lambda_n[0];
   double p01 = lambda_n[1]*xi*(1-xe);
   double p10 = lambda_n[1]*(1-xi)*xe;
   double p11 = (lambda_n[1]+lambda_n[2]*(1-xi)*(1-xe))*xi*xe;
   double p02 = lambda_n[2]*xi*(1-xe)*xi*(1-xe)/2;
   double p20 = lambda_n[2]*(1-xi)*xe*(1-xi)*xe/2;
   double p03 = lambda_n[3]*xi*xi*xi*(1-xe)*(1-xe)*(1-xe)/6;
   double p30 = lambda_n[3]*xe*xe*xe*(1-xi)*(1-xi)*(1-xi)/6;
   double p12 = (lambda_n[3]*(1-xe)*(1-xi)+2*lambda_n[2])*xi*xi*(1-xe)*xe/2;
   double p21 = (lambda_n[3]*(1-xi)*(1-xe)+2*lambda_n[2])*xe*xe*(1-xi)*xi/2;
   double val;
   
   val = prx_step(p00, N00) + 
         prx_step(p01, N01) + 
         prx_step(p02, N02) + 
         prx_step(p10, N10) + 
         prx_step(p11, N11) + 
         prx_step(p20, N20) + 
         prx_step(p03, N03) +
         prx_step(p12, N12) + 
         prx_step(p21, N21) +
         prx_step(p30, N30) + 
         prx_step(1-(p00+p01+p02+p10+p11+p20+p03+p12+p21+p30), (Nr-(N00+N01+N02+N10+N11+N20+N03+N12+N21+N30)));
   
   if(!mxIsFinite(val)) {
       mexPrintf("prx: non finite value!\n");
   }
}

/* compute a Gaussian random number for steps in the pi or q space */
double randn (double mu, double sigma) {
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}

/* rounds a number */
double round(double x) {
    return floor(x+0.5);
}

/* main probram */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* initilize variables */
    double *n_alpha, *n_beta, *q1_ret, *q2_ret, *q1_0, *q2_0, *para, *param, *pacc,
            lambda_1_n[4] = {0},  lambda_1_n_new[4] = {0}, lambda_2_n[4] = {0}, 
            lambda_2_n_new[4] = {0}, lambda_n_beta[4] = {0}, lambda_n_beta_new[4] = {0};
    int countp[8]={0}, count[8]={0}, dims[3];
    int Nrun, Nsweep, L, M, logover, i, j, l, k, n, x, chi, nu, mu, eta, teta;
    double Nr1,Nr2,N00_1, N01_1, N02_1, N10_1, N11_1, N20_1, N00_2, N01_2, N02_2, N10_2, N11_2, N20_2, N03_1, N12_1, N21_1, N30_1, N03_2, N12_2, N21_2, N30_2;
    double xi, xe, lambda_1_u, lambda_2_u, sigma_1, sigma_2, kappa_alpha, kappa_beta, sigma_qa, sigma_qb,
            sigma_sigma_1, sigma_sigma_2, sigma_lambda_1_u, sigma_lambda_2_u,
            sigma_xi, sigma_xe, lambda_1_u_new, lambda_2_u_new, xi_new, xe_new, 
            sigma_1_new, sigma_2_new, step, kappa_alpha_new, kappa_beta_new, p, qa_tilde_mn, qb_tilde_mn, 
            qa_tilde_mn_new, qb_tilde_mn_new, log_qa_tilde, log_qb_tilde, 
            log_qa_tilde_new, log_qb_tilde_new, log_para_update, 
            log_para_update_new, norm, tmp, tmp2, p1, p1_new;
    double gamma_dotdot, gamma_dotdot_new, omega, omega_new;
    double *q1, *q2, *q1_mu, *q1_nu, *q2_mu, *q2_nu;
    double detailedBalance, sigma_sigma_tmp;
    
    /* check number of input parameters */
    if(!(nrhs == 5 || nrhs == 7)) {
        mexErrMsgTxt("Function takes 5 or 7 input variables!");
    }
    
    /* get input data */
    n_beta  = mxGetData(prhs[0]); /* measured pump probe spectrum */
    n_alpha = mxGetData(prhs[1]); /* measured pump only spectrum (background) */
    Nrun    = (int) mxGetScalar(prhs[2]);
    Nsweep  = (int) mxGetScalar(prhs[3]);
    para    = mxGetData(prhs[4]); /* container for variables */
    
    L = (int) mxGetM(prhs[0]);
    M = (int) mxGetN(prhs[0]);
    dims[0] = L;
    dims[1] = M;
    dims[2] = Nsweep;
    
    /* check if data has correct dimensions */
    if(L < 1 || M < 1) {
        mexErrMsgTxt("Data error: n_alpha dimensions must be greater than 0!");
    }
    
    if(L != (int) mxGetM(prhs[1]) || M != (int) mxGetN(prhs[1])) {
        mexErrMsgTxt("Data error: n_beta must have the same dimensions than n_alpha!");
    }
    
    if(Nrun < 1) {
        mexErrMsgTxt("Input error: Nrun must be greater than 1!");
    }
    
    if(Nsweep < 1) {
        mexErrMsgTxt("Input error: Nsweep must be greater than 1!");
    }
    
    if(nrhs == 7) {
        q1_0 = mxGetData(prhs[5]); /* start vector for qa */
        q2_0 = mxGetData(prhs[6]); /* start vector for q2 */
        if(L != (int) mxGetM(prhs[5]) || M != (int) mxGetN(prhs[5])) {
            mexErrMsgTxt("Data error: startvector of qa must have the same dimensions than n_alpha!");
        }
        if(L != (int) mxGetM(prhs[6]) || M != (int) mxGetN(prhs[6])) {
            mexErrMsgTxt("Data error: startvector of q2 must have the same dimensions than n_alpha!");
        }
    }
    
    if(nlhs != 4) {
        mexErrMsgTxt("Output error: not enough output arguments!");
    }
    
    /* allocate memory for q matrixes */
    q1     = mxMalloc(M*L * sizeof(double));
    q2     = mxMalloc(M*L * sizeof(double));
    q1_mu  = mxMalloc(L * sizeof(double));
    q1_nu  = mxMalloc(M * sizeof(double));
    q2_mu  = mxMalloc(L * sizeof(double));
    q2_nu  = mxMalloc(M * sizeof(double));
    
    /* prepare output datapointer */
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(6, Nsweep, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(8, 1, mxREAL);
    q1_ret  = mxGetPr(plhs[0]); /* calculated spectrum timeseries (q_alpha) */
    q2_ret  = mxGetPr(plhs[1]); /* calculated spectrum timeseries (q_2) */
    param   = mxGetPr(plhs[2]); /* calculated parameter timeseries (lambda_alpha, lambda_betha, xi, xe) */
    pacc    = mxGetPr(plhs[3]); /* acceptence probability (lambda_alpha, lambda_betha, xi, xe, q) */
    
    /* load important parameters */
    lambda_1_u = para[0];
    lambda_2_u = para[1];
    sigma_1    = para[2];
    sigma_2    = para[3];
    xi         = para[4];
    xe         = para[5];
    sigma_qa = fabs(para[6]);
    sigma_qb = fabs(para[7]);
    sigma_lambda_1_u  = fabs(para[8]);
    sigma_lambda_2_u  = fabs(para[9]);
    sigma_sigma_1 = fabs(para[10]);
    sigma_sigma_2 = fabs(para[11]);
    sigma_xi = fabs(para[12]);
    sigma_xe = fabs(para[13]);
    Nr2   = para[14];
    N00_2 = para[15];
    N01_2 = para[16];
    N02_2 = para[17];
    N03_2 = para[18];
    N10_2 = para[19];
    N11_2 = para[20];
    N12_2 = para[21];
    N20_2 = para[22];
    N21_2 = para[23];
    N30_2 = para[24];
    Nr1   = para[25];
    N00_1 = para[26];
    N01_1 = para[27];
    N02_1 = para[28];
    N03_1 = para[29];
    N10_1 = para[30];
    N11_1 = para[31];
    N12_1 = para[32];
    N20_1 = para[33];
    N21_1 = para[34];
    N30_1 = para[35];
    
    /* check values of the parameters (is start point valid?) */
    if(lambda_1_u < 0) {
        mexErrMsgTxt("Parameter error: lambda_1_underline must be greater than 0!");
    }
    if(lambda_2_u < 0) {
        mexErrMsgTxt("Parameter error: lambda_2_underline must be greater than 0!");
    }
    if((xi < 0) || (xi > 1)) {
        mexErrMsgTxt("Parameter error: xi_i must be between 0 and 1!");
    }
    if((xe < 0) || (xe > 1)) {
        mexErrMsgTxt("Parameter error: xi_e must be between 0 and 1!");
    }
    if((Nr2 < 0) || (N00_2 < 0) || (N01_2 < 0) || (N02_2 < 0) 
        || (N03_2 < 0) || (N10_2 < 0) || (N11_2 < 0) || (N12_2 < 0)
        || (N20_2 < 0) || (N21_2 < 0) || (N30_2 < 0) || (Nr1 < 0)
        || (N00_1 < 0) || (N01_1 < 0) || (N02_1 < 0) || (N03_1 < 0) 
        || (N10_1 < 0) || (N11_1 < 0) || (N12_1 < 0) || (N20_1 < 0)
        || (N21_1 < 0) || (N30_1 < 0)) {
        mexErrMsgTxt("Parameter error: N_xy must be positive!");
    }
    
    if((sigma_qa <= 0) || (sigma_qb <= 0) || (sigma_1 <= 0) || (sigma_2 <= 0) 
        || (sigma_sigma_1 <= 0) || (sigma_sigma_2 <= 0) || (sigma_xi <= 0) || (sigma_xe <= 0)) {
        mexErrMsgTxt("Parameter error: sigma's must be greater than 0!");
    }
    
    for(i = 0; i < (L*M); i++ ) {
        if(n_alpha[i] < 0) {
            mexErrMsgTxt("Data error: n_alpha must be greater or equal to 0!");
        }
        if(n_beta[i] < 0) {
            mexErrMsgTxt("Data error: n_beta must be greater or equal to 0!");
        }
    }
    
    for(i = 0; i < L; i++) {
        q1_mu[i] = 0;
        q2_mu[i] = 0;
    }
    for(j = 0; j < M; j++) {
        q1_nu[j] = 0;
        q2_nu[j] = 0;
    }
    
    /* compute important values of starting point */
    compLambdaN(lambda_1_u, xi, xe, sigma_1, lambda_1_n);
    compLambdaN(lambda_2_u, xi, xe, sigma_2, lambda_2_n);
    compLambdaN_beta(lambda_n_beta, lambda_1_n, lambda_2_n);
    
    kappa_alpha = lambda_1_n[2]/lambda_1_n[1]*(1-xi)*(1-xe);
    kappa_beta  = lambda_2_n[2]/lambda_2_n[1]*(1-xi)*(1-xe);
    omega = 1+lambda_1_n[1]*lambda_2_n[1]/(lambda_1_n[0]*lambda_2_n[2]);
    gamma_dotdot = lambda_1_n[1]*lambda_2_n[0]/(lambda_1_n[0]*lambda_2_n[1]) * (1+kappa_alpha);
    
    /* start-vector */
    if(nrhs == 7) {
        tmp = 0;
        tmp2 = 0;
        for(i = 0; i < L; i++) {
            for(j = 0; j < M; j++) {
                q1[i+L*j]     = q1_0[i+L*j];
                q1_ret[i+L*j] = q1_0[i+L*j];
                q2[i+L*j]     = q2_0[i+L*j];
                q2_ret[i+L*j] = q2_0[i+L*j];
                q1_mu[i]      += q1[i+L*j];
                q2_mu[i]      += q2[i+L*j];
                q1_nu[j]      += q1[i+L*j];
                q2_nu[j]      += q2[i+L*j];
                tmp  += q1[i+L*j];
                tmp2 += q2[i+L*j];
                if((q1[i+L*j] < 0) || (q1[i+L*j] > 1)) {
                    mexErrMsgTxt("Start point error: q1 must be between 0 and 1!");
                }
                if((q2[i+L*j] < 0) || (q2[i+L*j] > 1)) {
                    mexErrMsgTxt("Start point error: q2 must be between 0 and 1!");
                }
            }
        }
        
        /* is start vector normalized? */
        if(fabs(1-tmp) > 0.00001) {
            mexErrMsgTxt("Start point error: q1 must be normalized!");
        }
        if(fabs(1-tmp2) > 0.00001) {
            mexErrMsgTxt("Start point error: q2 must be normalized!");
        }
    } else {
        /* start with flat distribution */
        for(i = 0; i < L; i++) {
            for(j = 0; j < M; j++) {
                q1[i+L*j]      = 1/((double)(L*M));
                q1_ret[i+L*j]  = q1[i+L*j];
                q2[i+L*j]      = 1/((double)(L*M));
                q2_ret[i+L*j]  = q2[i+L*j];
                q1_mu[i]      += q1[i+L*j];
                q2_mu[i]      += q2[i+L*j];
                q1_nu[j]      += q1[i+L*j];
                q2_nu[j]      += q2[i+L*j];
            }
        }
    }
    
    /* compute important probabilities for start point */
    log_qa_tilde = 0;
    log_qb_tilde = 0;  
    for(n = 0; n<M; n++){
        for(k = 0; k<L; k++){
            qa_tilde_mn = (q1[k+n*L] + kappa_alpha*q1_mu[k]*q1_nu[n])/(1+kappa_alpha); 
            qb_tilde_mn = (q2[k+n*L] + kappa_beta*(q2_mu[k]*q2_nu[n]+(omega-1)*(q1_mu[k]*q2_nu[n]+q2_mu[k]*q1_nu[n])) + qa_tilde_mn*gamma_dotdot)/(1+kappa_beta*(2*omega-1)+gamma_dotdot);
            
            log_qa_tilde += n_alpha[k+n*L]*log(qa_tilde_mn);
            log_qb_tilde += n_beta[k+n*L] *log(qb_tilde_mn);
        }
     }
    
    log_para_update = prx(lambda_1_n,    xi,xe,Nr1,N00_1,N01_1,N02_1,N10_1,N11_1,N20_1,N03_1, N12_1, N21_1, N30_1) + 
                      prx(lambda_n_beta, xi,xe,Nr2,N00_2,N01_2,N02_2,N10_2,N11_2,N20_2,N03_2, N12_2, N21_2, N30_2);
    
    /* write start point into output */
    param[0] = lambda_1_u;
    param[1] = lambda_2_u;
    param[2] = sigma_1;
    param[3] = sigma_2;
    param[4] = xi;
    param[5] = xe;
    
    /* here starts the computation and the inizializing ends */
    for(i = 1; i < Nsweep; i++) {
        /* make a sweep */
        for(j = 0; j < Nrun; j++) {             
            /* choose if parameters or spectrum will be updated */
            if (rand() % (2*L*M+6) < 6) {
                /* Parameter update */ 
                lambda_1_u_new = lambda_1_u;
                lambda_2_u_new = lambda_2_u;
                sigma_1_new = sigma_1;
                sigma_2_new = sigma_2;
                xi_new = xi;
                xe_new = xe;
                detailedBalance = 1;
                
                /* chosse a parameter to make a step */ 
                x = rand() % 6;
                
                count[x] += 1;
                switch (x) {
                    case 0:
                        /* lambda_1_u */
                        step             = (randn(0, sigma_lambda_1_u));
                        lambda_1_u_new = lambda_1_u + step;
                        logover = lambda_1_u_new > 0;
                        
                        break;
                    case 1:
                        /* lambda_2_u */
                        step            = (randn(0,sigma_lambda_2_u));
                        lambda_2_u_new = lambda_2_u + step;
                        logover = lambda_2_u_new > 0;
                        
                        break;
                    case 2:
                        /* sigma_1 */
                        /* compute not only the step but also consider the unsymmetric point suggestion of sigma (detailedBalance) */
                        sigma_sigma_tmp = sigmaSigma(sigma_1, sigma_sigma_1);
                        step            = randn(0,sigma_sigma_tmp);
                        detailedBalance = sigma_sigma_tmp/exp(-0.5*step*step/(sigma_sigma_tmp*sigma_sigma_tmp));
                        
                        sigma_1_new     = sigma_1 + step;
                        
                        sigma_sigma_tmp = sigmaSigma(sigma_1, sigma_sigma_1);
                        detailedBalance = detailedBalance/sigma_sigma_tmp*exp(-0.5*step*step/(sigma_sigma_tmp*sigma_sigma_tmp));
                        
                        /* lower limit for sigma_1 (numerical stability reasons) */
                        logover = sigma_1_new > 1e-8;
                        break;
                    case 3:
                        /* sigma_2 */
                        /* compute not only the step but also consider the unsymmetric point suggestion of sigma (detailedBalance) */
                        sigma_sigma_tmp = sigmaSigma(sigma_2, sigma_sigma_2);
                        step            = randn(0,sigma_sigma_tmp);
                        detailedBalance = sigma_sigma_tmp/exp(-0.5*step*step/(sigma_sigma_tmp*sigma_sigma_tmp));
                        
                        sigma_2_new     = sigma_2 + step;
                        
                        sigma_sigma_tmp = sigmaSigma(sigma_2, sigma_sigma_2);
                        detailedBalance = detailedBalance/sigma_sigma_tmp*exp(-0.5*step*step/(sigma_sigma_tmp*sigma_sigma_tmp));
                        
                        /* lower limit for sigma_2 (numerical stability reasons) */
                        logover = sigma_2_new > 1e-8;
                        
                        break;
                    case 4:
                        /* xi */
                        step    = (randn(0,sigma_xi));
                        xi_new  = xi + step;
                        logover = ((xi_new > 0)&&(xi_new < 1));
                        break;    
                    case 5:
                        /* xe */
                        step    = (randn(0,sigma_xe));
                        xe_new  = xe + step;
                        logover = ((xe_new > 0)&&(xe_new < 1));
                        break;    
                    default: mexErrMsgTxt("Error!"); break;
                }
                
                /* compute new important variables */
                compLambdaN(lambda_1_u_new, xi_new, xe_new, sigma_1_new, lambda_1_n_new);
                compLambdaN(lambda_2_u_new, xi_new, xe_new, sigma_2_new, lambda_2_n_new);
                compLambdaN_beta(lambda_n_beta_new, lambda_1_n_new, lambda_2_n_new);
                
                kappa_alpha_new = lambda_1_n_new[2]/lambda_1_n_new[1]*(1-xi_new)*(1-xe_new);
                kappa_beta_new  = lambda_2_n_new[2]/lambda_2_n_new[1]*(1-xi_new)*(1-xe_new);
    
                omega_new = 1 + lambda_1_n_new[1]*lambda_2_n_new[1]/(lambda_1_n_new[0]*lambda_2_n_new[2]);
                gamma_dotdot_new = lambda_1_n_new[1]*lambda_2_n_new[0]/(lambda_1_n_new[0]*lambda_2_n_new[1])*(1+kappa_alpha_new);
                
                /* computate step probability*/
                /* if logover == 0 -> invalid step (out of boundaries) */
                if(logover){
                    log_qa_tilde_new = 0;
                    log_qb_tilde_new = 0;  
                    for(n = 0; n<M; n++){
                        for(k = 0; k<L; k++){
                            qa_tilde_mn_new = (q1[k+n*L] + kappa_alpha_new*q1_mu[k]*q1_nu[n])/(1+kappa_alpha_new); 
                            qb_tilde_mn_new = (q2[k+n*L] + kappa_beta_new*(q2_mu[k]*q2_nu[n]+(omega_new-1)*(q1_mu[k]*q2_nu[n]+q2_mu[k]*q1_nu[n])) + qa_tilde_mn_new*gamma_dotdot_new)/(1+kappa_beta_new*(2*omega_new-1)+gamma_dotdot_new);

                            log_qa_tilde_new += n_alpha[k+n*L]*log(qa_tilde_mn_new);
                            log_qb_tilde_new += n_beta[k+n*L] *log(qb_tilde_mn_new);
                        }
                     }
                    
                    /* compute update prop for parameters */
                    log_para_update_new = prx(lambda_1_n_new,    xi_new, xe_new, Nr1, N00_1, N01_1, N02_1, N10_1, N11_1, N20_1, N03_1, N12_1, N21_1, N30_1) + 
                                          prx(lambda_n_beta_new, xi_new, xe_new, Nr2, N00_2, N01_2, N02_2, N10_2, N11_2, N20_2, N03_2, N12_2, N21_2, N30_2);   

                    p = lambda_1_u*lambda_2_u/(lambda_1_u_new*lambda_2_u_new) /* Jeffrey */
                        *exp((M-1)*(L-1)*(log(1+kappa_alpha)-log(1+kappa_alpha_new))) /* Jacobi (false coincidencies alpha) */
                        *exp(((-M*L+1)*log(1+kappa_beta_new*(2*omega_new-1)+gamma_dotdot_new) + (M+L-2)*log(1+omega_new*kappa_beta_new)) - /* Jacobi (false coincidencies beta new) */
                             ((-M*L+1)*log(1+kappa_beta*    (2*omega-1)    +gamma_dotdot)     + (M+L-2)*log(1+omega    *kappa_beta))) /* Jacobi (false coincidencies beta) */
                        *exp(log_qa_tilde_new - log_qa_tilde + log_qb_tilde_new - log_qb_tilde + log_para_update_new - log_para_update)  /* update */
                        *detailedBalance;
                    
                    
                    if (!mxIsFinite(p)) {
                        if (mxIsNaN(p)){
                            p = 0;
                            mexPrintf("p is nan! - some parameters/distributions are nan? - check!\n");
                        }
                    }
                } else {
                    p = 0;
                }
                
                
                if((double)rand() / (double)RAND_MAX < p) { 
                    /* accept step */
                    countp[x]       += 1;
                    lambda_1_u     = lambda_1_u_new;
                    lambda_2_u     = lambda_2_u_new;
                    sigma_1        = sigma_1_new;
                    sigma_2        = sigma_2_new;
                    memcpy(&lambda_1_n, &lambda_1_n_new, 4*sizeof(double));
                    memcpy(&lambda_2_n, &lambda_2_n_new, 4*sizeof(double));
                    memcpy(&lambda_n_beta, &lambda_n_beta_new, 4*sizeof(double));
                    xi              = xi_new;
                    xe              = xe_new;
                    kappa_alpha     = kappa_alpha_new;
                    kappa_beta      = kappa_beta_new;
                    log_qa_tilde    = log_qa_tilde_new;
                    log_qb_tilde    = log_qb_tilde_new;
                    log_para_update = log_para_update_new;
                    omega           = omega_new;
                    gamma_dotdot    = gamma_dotdot_new;
                }
            } else {
                /* spectrum update */
                chi  = (int) (rand() % (2)); /* random number between 0 and 1 */
                mu   = (int) (rand() % (L)); /* random number between 0 and L-1 */
                nu   = (int) (rand() % (M)); /* random number between 0 and M-1 */
                eta  = (int) (rand() % (L)); /* random number between 0 and L-1 */
                teta = (int) (rand() % (M)); /* random number between 0 and M-1 */
                
                if (chi == 0) {
                    /* update q_alpha */
                    count[6] += 1;
                    
                    step = fabs(randn(0,sigma_qa));
                    
                    /* make step here (return to start point if step is not accepted */
                    q1[mu+nu*L]    += step;
                    q1[eta+teta*L] -= step;
                    q1_mu[mu]      += step;
                    q1_nu[nu]      += step;
                    q1_mu[eta]     -= step;
                    q1_nu[teta]    -= step;
                    
                    /* check boundaries */
                    logover = (q1[mu+nu*L] < 1) && (q1[eta+teta*L] > 0);

                    /* if logover == 0 the value is out of boundaries */
                    if(logover){
                        /* compute q_alpha_tilde_mu_nu & q_beta_tilde_mu_nu */
                        log_qa_tilde_new = 0;
                        log_qb_tilde_new = 0;
                        for(n = 0; n<M; n++){
                            for(k = 0; k<L; k++){
                                qa_tilde_mn_new = (q1[k+n*L] + kappa_alpha*q1_mu[k]*q1_nu[n])/(1+kappa_alpha); 
                                qb_tilde_mn_new = (q2[k+n*L] + kappa_beta*(q2_mu[k]*q2_nu[n]+(omega-1)*(q1_mu[k]*q2_nu[n]+q2_mu[k]*q1_nu[n])) + qa_tilde_mn_new*gamma_dotdot)/(1+kappa_beta*(2*omega-1)+gamma_dotdot);
                                
                                log_qa_tilde_new += n_alpha[k+n*L]*log(qa_tilde_mn_new);
                                log_qb_tilde_new += n_beta[k+n*L] *log(qb_tilde_mn_new);
                            }
                         }
                         p = exp(log_qa_tilde_new - log_qa_tilde + log_qb_tilde_new - log_qb_tilde);
                    }else{
                        p = 0;
                    }

                    if((double)rand() / (double)RAND_MAX < p) { 
                        /* accept */
                        countp[6] += 1;
                        log_qa_tilde = log_qa_tilde_new;
                        log_qb_tilde = log_qb_tilde_new;
                    } else {
                        /* if step not accepted change it back */
                        q1[mu+nu*L]    -= step;
                        q1[eta+teta*L] += step;
                        q1_mu[mu]      -= step;
                        q1_nu[nu]      -= step;
                        q1_mu[eta]     += step;
                        q1_nu[teta]    += step;
                    }
                } else {
                    count[7] += 1;
                    
                    step = fabs(randn(0,sigma_qb));
                    
                    /* make step here (return to start point if step is not accepted */
                    q2[mu+nu*L]    += step;
                    q2[eta+teta*L] -= step;
                    q2_mu[mu]      += step;
                    q2_nu[nu]      += step;
                    q2_mu[eta]     -= step;
                    q2_nu[teta]    -= step;
                    
                    /* check boundaries */
                    logover = (q2[mu+nu*L]<1) && (q2[eta+teta*L] > 0);
                    
                    /* if logover == 0 q is out of boundaries */
                    if(logover){
                        /* compute q_beta_tilde_mu_nu */
                        log_qb_tilde_new = 0;
                        for(n = 0; n<M; n++){
                            for(k = 0; k<L; k++){
                                qa_tilde_mn_new = (q1[k+n*L] + kappa_alpha*q1_mu[k]*q1_nu[n])/(1+kappa_alpha); 
                                qb_tilde_mn_new = (q2[k+n*L] + kappa_beta*(q2_mu[k]*q2_nu[n]+(omega-1)*(q1_mu[k]*q2_nu[n]+q2_mu[k]*q1_nu[n])) + qa_tilde_mn_new*gamma_dotdot)/(1+kappa_beta*(2*omega-1)+gamma_dotdot);
                                
                                log_qb_tilde_new += n_beta[k+n*L] *log(qb_tilde_mn_new);
                            }
                         }
                         p = exp(log_qb_tilde_new - log_qb_tilde);
                    }else{
                        p = 0;
                    }
                    
                    if((double)rand() / (double)RAND_MAX < p) { 
                        /* accept */
                        countp[7] += 1;
                        log_qb_tilde = log_qb_tilde_new;
                    } else {
                        /* if step not accepted change it back*/
                        q2[mu+nu*L]    -= step;
                        q2[eta+teta*L] += step;
                        q2_mu[mu]      -= step;
                        q2_nu[nu]      -= step;
                        q2_mu[eta]     += step;
                        q2_nu[teta]    += step;
                    }
                }
            }
        }
        
        /* save current point into output vector */
        for(n = 0; n<M; n++){
            for(k = 0; k<L; k++){
                q1_ret[k+L*n+M*L*i] = q1[k+L*n];
                q2_ret[k+L*n+M*L*i] = q2[k+L*n];
            }
        }
        param[0 + 6*i] = lambda_1_u;
        param[1 + 6*i] = lambda_2_u;
        param[2 + 6*i] = sigma_1;
        param[3 + 6*i] = sigma_2;
        param[4 + 6*i] = xi;
        param[5 + 6*i] = xe;
    }
    
    /* compute accetpance probability */
    for(j = 0; j < 8; j++) {
        pacc[j] = ((double) countp[j])/((double) count[j]);
    }
    
    /* free memory */
    mxFree(q1);
    mxFree(q2);
    mxFree(q1_mu);
    mxFree(q1_nu);
    mxFree(q2_mu);
    mxFree(q2_nu);
}
