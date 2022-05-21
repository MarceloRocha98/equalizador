clc;clear;close all

q = 2.9;
N = 500;
M = 11; % ordem do equalizador 

Var_n = 0.001; % variância do ruído branco

% geracao de a (iid) com distribuicao de bernoulli
p = 0.5;
a = rand(1, N) > p;
a = double(a);
a(a==0) = -1;

n_ = randn(1,N)*sqrt(Var_n); % ruído branco


% item a)

q_ = [2.9 3.1 3.3 3.5];
for qs=1:length(q_)
    H_z = calc_H_z(q_(qs));
    a_filt = filter( [H_z(1), H_z(2), H_z(3)] , 1, a);
    u = n_ + a_filt; %u_n = n(n) + h(n);
    R = calc_cors(u, M);
    
    lambda = eig(R);
    lambda_max = max(lambda);
    lambda_min = min(lambda);
    lambda_max_min = lambda_max/lambda_min;
    
    sprintf('item a) PARA Q = %d, ru(0) = %d, ru(1) = %d, ru(2) = %d', q_(qs), R(1,1), R(2,1), R(3,1) )
    sprintf('lambda_max = %d', lambda_max)
    sprintf('lambda_min = %d', lambda_min)
    sprintf('lambda_max_min = %d', lambda_max_min)
    newline
    
end



% item b)

% Atraso de 1 amostra pelo canal H(z) (filtro FIR com 3 coeficientes tem atraso de 1 amostra)
% Atraso de 5 amostras pelo equalizador (11 Coeficientes tem atraso de 5 amostras)
% Total de atraso = 5 + 1 = 6 amostras
nd = 6;


% d = a(n)*z^-nd -> atraso de nd amostras (joga fora as primeiras nd amostras)
d = delayseq(a', -nd)'; % obs: ultimas nd amostras de d tao com 0
d = delayseq(d', nd); % torna as primeiras nd amostras de d com 0

H_z = calc_H_z(q);
a_filt = filter( [H_z(1), H_z(2), H_z(3)] , 1, a);

u = n_ + a_filt; %u_n = n(n) + h(n);




% LMS
mu = 0.075;
[e, W] = calc_LMS(M, N, u, d, mu);


[W, erro_medio] = calc_erros(M,N, mu, H_z, nd, Var_n, true);
figure(1);
plot([1:N],10*log10(abs(erro_medio)));
hold on;



% NLMS
[W, erro_medio] = calc_erros(M,N, mu, H_z, nd, Var_n, false);
plot([1:N],10*log10(abs(erro_medio)));

title('MSE (dB)')
legend('LMS','NLMS')




% funcoes


function [W, erro_medio] = calc_erros(M,N, mu, H_z, nd, Var_n, LMS) % Calcula o erro medio para LMS se LMS = true, ou NLMS, se LMS = false
erros = zeros(N,N);
erro_medio = zeros(N,1);

for n=1:N
    p = 0.5;
    a = rand(1, N) > p;
    a = double(a);
    a(a==0) = -1;
    
    n_ = randn(1,N)*sqrt(Var_n); % ruído branco

    a_filt = filter( [H_z(1), H_z(2), H_z(3)] , 1, a);
    u = n_ + a_filt; %u_n = n(n) + h(n);
    d = delayseq(a', -nd)'; % obs: ultimas nd amostras de d tao com 0
    d = delayseq(d', nd)'; % torna as primeiras nd amostras de d com 0 ("shift" em d para direita)
    
    if(LMS)
        [e, W] = calc_LMS(M, N, u, d, mu);
        erros(:,n) = transpose(e.^2); 
        
    else
        u_ = mu ; % u_ entre 0 e 2
        [e, W] = calc_NLMS(N, u, d, u_);
        erros(:,n) = transpose(e.^2); 
    end
    

end

for n=1:N
    erro_medio(n,:) = sum(erros(n,1:100))/100; % erro para as 100 primeiras iteracoes
end


end


function [e, W] = calc_LMS(M, N, u, d, mu)
X = zeros(M,1);
e = zeros(1,N);
y = zeros(1,N);
W = zeros(N+1,M);


for n=1:N
   X = [u(n);X(1:M-1)]; 
   y(n) = W(n,:)*X;
   e(n) = d(n)-y(n);
   W(n+1,:) = W(n,:) + mu*e(n)*X';
end

end


function [e, W] = calc_NLMS(N, u, d, u_)
sig = 10e-5;
e = zeros(1,N);
y = zeros(1,N);
W = zeros(N, 1);

for n=2:N
   y(n) = transpose(u(n))*W(n-1);
%    y(n) = W(n-1)'*u(n);
   e(n) = d(n)-y(n);
   W(n) = W(n-1) + (u_/(sig+abs(u(n))^2))*u(n)*e(n);
%   W(n) = W(n-1) + (u_/(sig+u(n)'*u(n)))*u(n)*e(n);
end

end




function H_z = calc_H_z(q)
    h = @(k,q) 0.5*(1+cos(2*pi*(k-2)/q));
    H_z = [h(1, q) h(2, q) h(3, q)];
end

function R = calc_cors(u, M)
    r=xcorr(u,M-1,'biased');
    ru=r(M:end);
    R=toeplitz(ru); % matriz de autocorrelacao
    
%     rdu=xcorr(d,u,M-1,'biased');
%     p=rdu(M:end); % vetor correlacao cruzada
end