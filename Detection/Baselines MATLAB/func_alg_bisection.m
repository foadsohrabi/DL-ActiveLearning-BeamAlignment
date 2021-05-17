function idx_hat = func_alg_bisection(S,w_D,P,alpha, h,noise_mat)

k = 1;
t = 1;
for ell=1:S
    w = w_D{ell}{2*k-1};
    y1 = sqrt(P)*alpha*w'*h + noise_mat(1,t);%w'*noise_mat(:,t);
    t = t+1;

    w = w_D{ell}{2*k};
    y2 = sqrt(P)*alpha*w'*h + noise_mat(1,t);%w'*noise_mat(:,t);
    t = t+1;

    if abs(y1)>abs(y2)
        k = 2*k-1;
    else
        k = 2*k;
    end
end
idx_hat = k;