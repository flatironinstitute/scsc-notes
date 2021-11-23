function [x rhist] = mygmres(A,b,tol,maxit)
% MYGMRES  demo a basic GMRES iterative solver for general square linear system
%
% [x rhist] = mygmres(A,b,tol,maxit) returns solution x and relative residual
%  norms for all iterations.
%
% Called without arguments does a self test demo, borrowing from
%  Trefethen & Bau, Numerical Linear Algebra (SIAM, 1997). Ch. 33 & 35.

% Barnett 10/26/21
if nargin==0, test_mygmres; return; end

nrmb = norm(b);
Q(:,1) = b/nrmb;
for n=1:maxit
    % do nth iter of Arnoldi...
    v = A*Q(:,n);
    for j=1:n
        H(j,n) = Q(:,j)'*v;
        v = v - H(j,n)*Q(:,j);
    end
    H(n+1,n) = norm(v);
    Q(:,n+1) = v/H(n+1,n);
    % dense solve of little LSQ prob for y in C^n...
    rhs = [nrmb; zeros(n,1)];
    y = H \ rhs;
    rhist(n) = norm(H*y - rhs) / nrmb;
    x = Q(:,1:n)*y;
    if rhist(n)<tol, return; end
end

%%%%%%%
function test_mygmres
N = 1e3;       % size
A = eye(N) + 0.3/sqrt(N)*randn(N,N);    % nice matrix: clustered away from 0
                                        % OR uncomment the following...
%A = diag(exp(5i*(1:N)/N)) + 0.3/sqrt(N)*randn(N,N); % nasty matrix: surrounds 0
% (note we could diag precond away this particular nastiness easily)

figure; if N<=3e3, subplot(1,2,1); plot(eig(A),'+'); title('spec(A)');
    xlabel('Re \lambda'); ylabel('Im \lambda'); axis equal; end

x = randn(N,1);  % true
b = A*x;
tol = 1e-6;
[x rhist] = mygmres(A,b,tol,N);
subplot(1,2,2); semilogy(rhist,'+-'); title('convergence');
xlabel('iters'); ylabel('relative residual norm'); hline(tol); axis tight;
hold on; ii=1:numel(rhist); plot(ii,0.3.^ii,'r-');  % nice geom rate bound

 