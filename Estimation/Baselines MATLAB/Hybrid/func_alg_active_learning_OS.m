function  idx_hat = func_alg_active_learning_OS(control_plot,delta_inv,S,tau,alpha,alpha_hat,w_D,noise_mat,P,h,OS_rate,A_BS_OS)

if control_plot ==1
    figure;
end
delta = 1/delta_inv;
PI = delta*ones(delta_inv,1);
posteriors = func_compute_posteriors(PI,S,control_plot);

PI_OS = (1/(OS_rate*delta_inv))*ones(OS_rate*delta_inv,1);
for t = 1:tau
    %%% Codeword Selection
    [~,k] = max([posteriors{1}{:}]);
    for ell = 1:S
        if posteriors{ell}{k} >= 0.5 && ell~=S
            ell_star = ell;
            [~,k12]=max([posteriors{ell+1}{2*k-1:2*k}]);
            k = (k12-1)*(2*k)+(2-k12)*(2*k-1);           
        else
            choice1 = posteriors{ell_star}{ceil(k/2)};
            choice2 = posteriors{ell_star+1}{k};
            [~,choice_idx] = min([abs(choice1-0.5),abs(choice2-0.5)]);
            if choice_idx == 1
                ell_measure = ell_star;
                k_measure = ceil(k/2);
            else
                ell_measure = ell_star+1;
                k_measure = k;
            end
            break;
        end
    end
    %disp(128/2^ell_measure)
    w = w_D{ell_measure}{k_measure};
    y = sqrt(P)*alpha*w'*h + noise_mat(:,t);%w'*noise_mat(:,t);       
        
    [PI, PI_OS] = func_update_PI_OS(PI,PI_OS,y,alpha_hat,P,w,A_BS_OS,OS_rate);
    posteriors = func_compute_posteriors(PI,S,control_plot);
   
end
[~,idx_hat] = max(PI_OS);
