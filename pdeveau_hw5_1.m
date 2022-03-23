%%%a
X = [1 1 0 2;3 7 5 5];
[d,n] = size(X);
mu_x = (1/n) * sum(X,2)
one_n = ones(n,1);
X_tilde = X - mu_x * one_n';
Sx = (1/n) .* ((X - mu_x) * (X - mu_x)');
Sx = (1/n) * X_tilde * X_tilde'
[V,D] = eig(Sx)
[coeff, score, latent, tsquared, explained] = pca(X)

%%%b
r = [-1 2 -1 0 2]'
s = [1 -1 -1 1 1]'
X = [3*r,r,-1*r,-3*r,3*s,2*s,s,-6*s]
[d,n] = size(X);
mu_x = (1/n) * sum(X,2)
one_n = ones(n,1);
X_tilde = X - mu_x * one_n'
Sx = (1/n) .* ((X - mu_x) * (X - mu_x)')
Sx = (1/n) * X_tilde * X_tilde'