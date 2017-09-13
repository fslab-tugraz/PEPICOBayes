#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>

double prx(double lambda, double xi, double xe, double Nr, double N00, double N01, double N02, double N10, double N11, double N20, double N03, double N12, double N21, double N30) {
   double p11 = lambda*xi*xe*(1+lambda*(1-xi)*(1-xe))*exp(-lambda*(1-(1-xi)*(1-xe)));
   double p00 = exp(-lambda*(1-(1-xi)*(1-xe)));
   double p01 = lambda*xi*(1-xe)*exp(-lambda*(1-(1-xi)*(1-xe)));
   double p10 = lambda*(1-xi)*xe*exp(-lambda*(1-(1-xi)*(1-xe)));
   double p02 = lambda*xi*(1-xe)*lambda*xi*(1-xe)*exp(-lambda*(1-(1-xi)*(1-xe)))/2;
   double p20 = lambda*(1-xi)*xe*lambda*(1-xi)*xe*exp(-lambda*(1-(1-xi)*(1-xe)))/2;
   double p03 = lambda*lambda*lambda*xi*xi*xi*(1-xe)*(1-xe)*(1-xe)*exp(-lambda*(1-(1-xi)*(1-xe)))/6;
   double p30 = lambda*lambda*lambda*xe*xe*xe*(1-xi)*(1-xi)*(1-xi)*exp(-lambda*(1-(1-xi)*(1-xe)))/6;
   double p12 = lambda*lambda*xi*xi*(1-xe)*xe*((1-xe)*(1-xi)*lambda+2)*exp(-lambda*(1-(1-xi)*(1-xe)))/2;
   double p21 = lambda*lambda*xe*xe*(1-xi)*xi*((1-xi)*(1-xe)*lambda+2)*exp(-lambda*(1-(1-xi)*(1-xe)))/2;
   double val;
   
   val = N00*log(p00) + N01*log(p01) + N02*log(p02) + N10*log(p10) + N11*log(p11) + N20*log(p20) + N03*log(p03) + N12*log(p12) + N21*log(p21) + N30*log(p30)
          + (Nr-(N00+N01+N02+N10+N11+N20+N03+N12+N21+N30))*log(1-(p00+p01+p02+p10+p11+p20+p03+p12+p21+p30));
   
   return val;
}

/* create Gaussian distributed random number */
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

double round(double x) {
    return floor(x+0.5);
}

/* Monte Carlo code */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *n_alpha, *n_beta, *qa_ret, *q2_ret, *qa_0, *q2_0, *para, *pacc, *pi0, *pi;
    int countp[6]={0}, count[6]={0}, dims[3];
    int Nrun, Nsweep, N, L, M,
            logpara, logover, i, j, l, k, n, x, chi, nu, mu, eta, teta;
    double Nr1,Nr2,N00_1, N01_1, N02_1, N10_1, N11_1, N20_1, N00_2, N01_2, N02_2, N10_2, N11_2, N20_2, N03_1, N12_1, N21_1, N30_1, N03_2, N12_2, N21_2, N30_2;
    double xi, xe, lambda_alpha, lambda_beta, kappa_alpha, kappa_beta, sigma_qa, sigma_qb, sigma_la, sigma_lb, sigma_xi, sigma_xe,  
            lambda_alpha_new, lambda_beta_new, xi_new, xe_new, step, kappa_alpha_new, kappa_beta_new, p, 
            qa_tilde_mn, qb_tilde_mn, qa_tilde_mn_new, qb_tilde_mn_new, log_qa_tilde, log_qb_tilde, log_qa_tilde_new, log_qb_tilde_new,
            log_para_update, log_para_update_new, norm, tmp, tmp2;
    double *qa, *qb, *qa_mu, *qa_nu, *qb_mu, *qb_nu;
    
    if(!(nrhs == 6 || nrhs == 8)) {
        mexErrMsgTxt("Function takes 6 or 8 input variables!");
    }
    
    /* get input data */
    n_beta  = mxGetData(prhs[0]); /* measured pump probe spectrum */
    n_alpha = mxGetData(prhs[1]); /* measured pump only spectrum (background) */
    Nrun    = (int) mxGetScalar(prhs[2]);
    Nsweep  = (int) mxGetScalar(prhs[3]);
    para    = mxGetData(prhs[4]); /* container for variables */
    pi0    = mxGetData(prhs[5]); /* start values (lambda_1, lambda_2, xi_i, xi_e */
    
    L = (int) mxGetM(prhs[0]);
    M = (int) mxGetN(prhs[0]);
    dims[0] = L;
    dims[1] = M;
    dims[2] = Nsweep;
    
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
    
    if((int) (mxGetM(prhs[4])*mxGetN(prhs[4])) != 28) {
        mexErrMsgTxt("Input error: parameter must have 28 entries!");
    }
    
    if((int) (mxGetM(prhs[5])*mxGetN(prhs[5])) != 4) {
        mexErrMsgTxt("Input error: pi0 must have 4 entries!");
    }
    
    if(nrhs == 8) {
        qa_0 = mxGetData(prhs[6]); /* start vector for qa */
        q2_0 = mxGetData(prhs[7]); /* start vector for q2 */
        if(L != (int) mxGetM(prhs[6]) || M != (int) mxGetN(prhs[6])) {
            mexErrMsgTxt("Data error: startvector of qa must have the same dimensions than n_alpha!");
        }
        if(L != (int) mxGetM(prhs[7]) || M != (int) mxGetN(prhs[7])) {
            mexErrMsgTxt("Data error: startvector of q2 must have the same dimensions than n_alpha!");
        }
    }
    
    if(nlhs != 4) {
        mexErrMsgTxt("Output error: 4 output arguments needed!");
    }
    
    qa     = mxMalloc(M*L * sizeof(double));
    qb     = mxMalloc(M*L * sizeof(double));
    qa_mu  = mxMalloc(L * sizeof(double));
    qa_nu  = mxMalloc(M * sizeof(double));
    qb_mu  = mxMalloc(L * sizeof(double));
    qb_nu  = mxMalloc(M * sizeof(double));
    
    N = 2*L*M;
    
    /* prepare output datapointer */
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(4, Nsweep, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(6, 1, mxREAL);
    qa_ret  = mxGetPr(plhs[0]); /* calculated spectrum timeseries (q_alpha) */
    q2_ret  = mxGetPr(plhs[1]); /* calculated spectrum timeseries (q_2) */
    pi      = mxGetPr(plhs[2]); /* calculated parameter timeseries (lambda_alpha, lambda_betha, xi, xe) */
    pacc    = mxGetPr(plhs[3]); /* acceptence probability (lambda_alpha, lambda_betha, xi, xe, q) */
    
    /* Variablendefinitionen */
    lambda_alpha = pi0[0];
    lambda_beta  = pi0[1] + lambda_alpha;
    xi    = pi0[2];
    xe    = pi0[3];
    sigma_qa = fabs(para[0]);/*((double)(10*L*M));*/
    sigma_qb = fabs(para[1]);/*((double)(10*L*M));*/
    sigma_la = fabs(para[2]);/*4000*/
    sigma_lb = fabs(para[3]);/*4000*/
    sigma_xi = fabs(para[4]);/*1000*/
    sigma_xe = fabs(para[5]);/*1000*/
    Nr2   = para[6];
    N00_2 = para[7];
    N01_2 = para[8];
    N02_2 = para[9];
    N03_2 = para[10];
    N10_2 = para[11];
    N11_2 = para[12];
    N12_2 = para[13];
    N20_2 = para[14];
    N21_2 = para[15];
    N30_2 = para[16];
    Nr1   = para[17];
    N00_1 = para[18];
    N01_1 = para[19];
    N02_1 = para[20];
    N03_1 = para[21];
    N10_1 = para[22];
    N11_1 = para[23];
    N12_1 = para[24];
    N20_1 = para[25];
    N21_1 = para[26];
    N30_1 = para[27];
    
    if(lambda_alpha < 0) {
        mexErrMsgTxt("Parameter error: lambda_1 must be greater than 0!");
    }
    if(lambda_beta-lambda_alpha < 0) {
        mexErrMsgTxt("Parameter error: lambda_2 must be greater than 0!");
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
    
    for(i = 0; i < (L*M); i++ ) {
        if(n_alpha[i] < 0) {
            mexErrMsgTxt("Data error: n_alpha must be greater or equal to 0!");
        }
        if(n_beta[i] < 0) {
            mexErrMsgTxt("Data error: n_beta must be greater or equal to 0!");
        }
    }
    
    for(i = 0; i < L; i++) {
        qa_mu[i] = 0;
        qb_mu[i] = 0;
    }
    for(j = 0; j < M; j++) {
        qa_nu[j] = 0;
        qb_nu[j] = 0;
    }
    
    /* start-vector */
    if(nrhs == 8) {
        tmp = 0;
        tmp2 = 0;
        for(i = 0; i < L; i++) {
            for(j = 0; j < M; j++) {
                qa_ret[i+L*j] = qa_0[i+L*j];
                q2_ret[i+L*j] = q2_0[i+L*j];
                qa[i+L*j]     = qa_0[i+L*j];
                qb[i+L*j]     = (lambda_alpha*qa[i+L*j] + (lambda_beta-lambda_alpha)*q2_0[i+L*j])/lambda_beta;
                qa_mu[i]      += qa[i+L*j];
                qb_mu[i]      += qb[i+L*j];
                qa_nu[j]      += qa[i+L*j];
                qb_nu[j]      += qb[i+L*j];
                tmp  += qa_ret[i+L*j];
                tmp2 += q2_ret[i+L*j];
                if((qa[i+L*j] < 0) || (qa[i+L*j] > 1)) {
                    mexErrMsgTxt("Start point error: qa must be between 0 and 1!");
                }
                if((q2_0[i+L*j] < 0) || (q2_0[i+L*j] > 1)) {
                    mexErrMsgTxt("Start point error: q2 must be between 0 and 1!");
                }
                if((qb[i+L*j] < 0) || (qb[i+L*j] > 1)) {
                    mexErrMsgTxt("Start point error: Not a valid starting point (q_beta is not in correct range)!");
                }
            }
        }
        if(fabs(1-tmp) > 0.00001) {
            mexErrMsgTxt("Start point error: qa must be normalized!");
        }
        if(fabs(1-tmp2) > 0.00001) {
            mexErrMsgTxt("Start point error: q2 must be normalized!");
        }
    } else {
        /* start with flat distribution */
        for(i = 0; i < L; i++) {
            for(j = 0; j < M; j++) {
                qa[i+L*j]      = 1/((double)(L*M));
                q2_ret[i+L*j]  = 1/((double)(L*M));
                qa_ret[i+L*j]  = qa[i+L*j];
                qb[i+L*j]      = (lambda_alpha*qa[i+L*j] + (lambda_beta-lambda_alpha)*q2_ret[i+L*j])/lambda_beta;
                qa_mu[i]      += qa[i+L*j];
                qb_mu[i]      += qb[i+L*j];
                qa_nu[j]      += qa[i+L*j];
                qb_nu[j]      += qb[i+L*j];
            }
        }
    }
    
    pi[0] = lambda_alpha;
    pi[1] = lambda_beta - lambda_alpha;
    pi[2] = xi;
    pi[3] = xe;
    
    kappa_alpha = lambda_alpha*(1-xi)*(1-xe);
    kappa_beta  = lambda_beta*(1-xi)*(1-xe); 
    
    log_qa_tilde = 0;
    log_qb_tilde = 0;  
    for(n = 0; n<M; n++){
        for(k = 0; k<L; k++){
            qa_tilde_mn = (qa[k+n*L] + kappa_alpha*qa_mu[k]*qa_nu[n])/(1+kappa_alpha); 
            qb_tilde_mn = (qb[k+n*L] + kappa_beta *qb_mu[k]*qb_nu[n])/(1+kappa_beta);
            
            log_qa_tilde += n_alpha[k+n*L]*log(qa_tilde_mn);
            log_qb_tilde += n_beta[k+n*L] *log(qb_tilde_mn);      
        }
     }
    
    log_para_update = prx(lambda_alpha,xi,xe,Nr1,N00_1,N01_1,N02_1,N10_1,N11_1,N20_1,N03_1, N12_1, N21_1, N30_1) + 
                      prx(lambda_beta, xi,xe,Nr2,N00_2,N01_2,N02_2,N10_2,N11_2,N20_2,N03_2, N12_2, N21_2, N30_2);
    /* here starts the computation and the inizializing ends */
    
    for(i = 1; i < Nsweep; i++) {
        /* make a sweep */
        for(j = 0; j < Nrun; j++) {             
            /* choose if parameters or spectrum will be updated */
            if (rand() % (N+4) < 4) {
                /* Parameter update */ 
                lambda_alpha_new = lambda_alpha;
                lambda_beta_new  = lambda_beta;
                xi_new = xi;
                xe_new = xe;

                /* chosse a parameter */ 
                x = rand() % 4;
                count[x] += 1;
                switch (x) {
                    case 0:
                        /* lambda_alpha */
                        step             = (randn(0,sigma_la));
                        lambda_alpha_new = lambda_alpha + step;
                        logover = 0;
                        if((lambda_alpha_new > 0) && (lambda_alpha_new < lambda_beta)) {
                            logover = 1;
                            for(n = 0; ((n < L) && logover); n++) {
                                for(k = 0; ((k < M) && logover); k++) {
                                    tmp = (lambda_beta*qb[n+L*k] - lambda_alpha_new*qa[n+L*k])/(lambda_beta-lambda_alpha_new);
                                    logover = ((logover && (tmp>0)) && (tmp<1));
                                }
                            }
                        }
                        break;
                    case 1:
                        /* lambda_beta */
                        step            = (randn(0,sigma_lb));
                        lambda_beta_new = lambda_beta + step;
                        logover = 0;
                        if(lambda_beta_new > lambda_alpha) {
                            logover = 1;
                            for(n = 0; ((n < L) && logover); n++) {
                                for(k = 0; ((k < M) && logover); k++) {
                                    tmp = (lambda_beta_new*qb[n+L*k] - lambda_alpha*qa[n+L*k])/(lambda_beta_new-lambda_alpha);
                                    logover = ((logover && (tmp>0)) && (tmp<1));
                                }
                            }
                        }
                        break;
                    case 2:
                        /* xi */
                        step    = (randn(0,sigma_xi));
                        xi_new  = xi+ step;
                        logover = ((xi_new > 0)&&(xi_new < 1));
                        break;    
                    case 3:
                        /* xe */
                        step    = (randn(0,sigma_xe));
                        xe_new  = xe + step;
                        logover = ((xe_new > 0)&&(xe_new < 1));
                        break;    
                    default: mexErrMsgTxt("Error!"); break;
                }
                kappa_alpha_new = lambda_alpha_new*(1-xi_new)*(1-xe_new);
                kappa_beta_new  = lambda_beta_new *(1-xi_new)*(1-xe_new);
                
                /* computate step probability*/
                if(logover){
                    log_qa_tilde_new = 0;
                    log_qb_tilde_new = 0;
                    for(n = 0; n<M; n++){   
                        for(k = 0; k<L; k++){
                            /* compute every q_alpha_tild_mu_nu and q_beta_tild_mu_nu */
                            qa_tilde_mn_new = (qa[k+n*L] + kappa_alpha_new*qa_mu[k]*qa_nu[n])/(1+kappa_alpha_new);
                            qb_tilde_mn_new = (qb[k+n*L] + kappa_beta_new* qb_mu[k]*qb_nu[n])/(1+kappa_beta_new);

                            log_qa_tilde_new += n_alpha[k+n*L]*log(qa_tilde_mn_new);
                            log_qb_tilde_new += n_beta[k+n*L] *log(qb_tilde_mn_new);
                        }
                    }
                    
                    /* compute update prop for parameters */
                    log_para_update_new = prx(lambda_alpha_new,xi_new,xe_new,Nr1,N00_1,N01_1,N02_1,N10_1,N11_1,N20_1,N03_1, N12_1, N21_1, N30_1) + 
                                          prx(lambda_beta_new, xi_new,xe_new,Nr2,N00_2,N01_2,N02_2,N10_2,N11_2,N20_2,N03_2, N12_2, N21_2, N30_2);   

                    p = lambda_alpha*lambda_beta/(lambda_alpha_new*lambda_beta_new) /* Jeffrey */
                        *exp(M*L*log(1-lambda_alpha_new/lambda_beta_new) - M*L*log(1-lambda_alpha/lambda_beta)) /* Jacobi (subtraction) */
                        *exp((M-1)*(L-1)*(log(1+kappa_alpha)-log(1+kappa_alpha_new))) /* Jacobi (false coincidencies alpha) */
                        *exp((M-1)*(L-1)*(log(1+kappa_beta)-log(1+kappa_beta_new))) /* Jacobi (false coincidencies beta) */
                        *exp(log_qa_tilde_new - log_qa_tilde + log_qb_tilde_new - log_qb_tilde + log_para_update_new - log_para_update); 
                }else{
                    p = 0;
                }
                
                if((double)rand() / (double)RAND_MAX < p) { 
                    /* accept step */
                    countp[x]       += 1;
                    lambda_alpha    = lambda_alpha_new;
                    lambda_beta     = lambda_beta_new;
                    xi              = xi_new;
                    xe              = xe_new;
                    kappa_alpha     = kappa_alpha_new;
                    kappa_beta      = kappa_beta_new;
                    log_qa_tilde    = log_qa_tilde_new;
                    log_qb_tilde    = log_qb_tilde_new;
                    log_para_update = log_para_update_new;
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
                    count[4] += 1;
                    
                    step = fabs(randn(0,sigma_qa));
                    
                    /* make step here (return to start point if step is not accepted */
                    qa[mu+nu*L]    += step;
                    qa[eta+teta*L] -= step;
                    qa_mu[mu]      += step;
                    qa_nu[nu]      += step;
                    qa_mu[eta]     -= step;
                    qa_nu[teta]    -= step;
                    
                    /* is step valid? - check also q_2 ranges */
                    tmp =  (lambda_beta*qb[mu+nu*L]    - qa[mu+nu*L]*   lambda_alpha)/(lambda_beta-lambda_alpha);
                    tmp2 = (lambda_beta*qb[eta+teta*L] - qa[eta+teta*L]*lambda_alpha)/(lambda_beta-lambda_alpha);
                    logover = (0 < tmp) && (1 > tmp2) && (qa[mu+nu*L] < 1) && (qa[eta+teta*L] > 0);

                    if(logover){
                        /* compute q_alpha_tilde_mu_nu */
                        log_qa_tilde_new = 0;
                        for(n = 0; n<M; n++){
                            for(k = 0; k<L; k++){
                                qa_tilde_mn_new = (qa[k+n*L] + kappa_alpha*qa_mu[k]*qa_nu[n])/(1+kappa_alpha);
                                log_qa_tilde_new += n_alpha[k+n*L]*log(qa_tilde_mn_new);
                            }
                         }
                         p = exp(log_qa_tilde_new - log_qa_tilde);
                    }else{
                        p = 0;
                    }

                    if((double)rand() / (double)RAND_MAX < p) { 
                        /* accept */
                        countp[4] += 1;
                        log_qa_tilde = log_qa_tilde_new;
                    } else {
                        /* if step not accepted change it back*/
                        qa[mu+nu*L]    -= step;
                        qa[eta+teta*L] += step;
                        qa_mu[mu]      -= step;
                        qa_nu[nu]      -= step;
                        qa_mu[eta]     += step;
                        qa_nu[teta]    += step;
                    }
                } else {
                    count[5] += 1;
                    
                    step = fabs(randn(0,sigma_qb));
                    
                    /* make step here (return to start point if step is not accepted */
                    qb[mu+nu*L]    += step;
                    qb[eta+teta*L] -= step;
                    qb_mu[mu]      += step;
                    qb_nu[nu]      += step;
                    qb_mu[eta]     -= step;
                    qb_nu[teta]    -= step;

                    /* is step valid? - check also q_2 ranges */
                    tmp  = (lambda_beta*qb[mu+nu*L]    - qa[mu+nu*L]*   lambda_alpha)/(lambda_beta-lambda_alpha);
                    tmp2 = (lambda_beta*qb[eta+teta*L] - qa[eta+teta*L]*lambda_alpha)/(lambda_beta-lambda_alpha);
                    logover = (1 > tmp) && (0 < tmp2) && (qb[mu+nu*L]<1) && (qb[eta+teta*L] > 0);
                    if(logover){
                        /* compute q_beta_tilde_mu_nu */
                        log_qb_tilde_new = 0;
                        for(n = 0; n<M; n++){
                            for(k = 0; k<L; k++){
                                qb_tilde_mn_new = (qb[k+n*L] + kappa_beta*qb_mu[k]*qb_nu[n])/(1+kappa_beta);
                                log_qb_tilde_new += n_beta[k+n*L]*log(qb_tilde_mn_new);
                            }
                         }
                         p = exp(log_qb_tilde_new - log_qb_tilde);
                    }else{
                        p = 0;
                    }

                    if((double)rand() / (double)RAND_MAX < p) { 
                        /* accept */
                        countp[5] += 1;
                        log_qb_tilde = log_qb_tilde_new;
                    } else {
                        /* if step not accepted change it back*/
                        qb[mu+nu*L]    -= step;
                        qb[eta+teta*L] += step;
                        qb_mu[mu]      -= step;
                        qb_nu[nu]      -= step;
                        qb_mu[eta]     += step;
                        qb_nu[teta]    += step;
                    }
                }
            }
        }
        
        /* save current point into output vector */
        for(n = 0; n<M; n++){
            for(k = 0; k<L; k++){
                qa_ret[k+L*n+M*L*i] = qa[k+L*n];
                q2_ret[k+L*n+M*L*i] = (qb[k+L*n]*lambda_beta - qa[k+L*n]*lambda_alpha)/(lambda_beta-lambda_alpha);
            }
        }
        pi[0 + 4*i] = lambda_alpha;
        pi[1 + 4*i] = lambda_beta - lambda_alpha;
        pi[2 + 4*i] = xi;
        pi[3 + 4*i] = xe;
    }
    
    for(j = 0; j < 6; j++) {
        pacc[j] = ((double) countp[j])/((double) count[j]);
    }
    
    /* free memory */
    mxFree(qa);
    mxFree(qb);
    mxFree(qa_mu);
    mxFree(qa_nu);
    mxFree(qb_mu);
    mxFree(qb_nu);
}
