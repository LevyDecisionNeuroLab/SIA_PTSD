// Modeling robust regression with student's t instead of normal distribution

data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    
    real<lower=0> sigma;
    real nu;
}
model {
  alpha ~ normal(0,10);
  beta ~ normal(0,10);
  nu ~ gamma(2, 0.1);
  y ~ student_t(nu, alpha + beta * x , sigma);
}