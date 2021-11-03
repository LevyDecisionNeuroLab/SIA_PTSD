data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    vector[N] cov;
}
parameters {
    real alpha;
    real beta;
    real nu;
    real<lower=0> sigma;
}
model {
    alpha ~ normal(0, 1);
    beta ~ normal(0,1);
    nu ~ gamma(30, 0.1);
    y ~ student_t(nu, alpha + beta * x + cov, sigma);
}