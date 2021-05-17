function [w_D,A_BS,A_BS_pinv,theta,A_BS_OS,theta_OS]=func_codedesign(delta_inv,theta_min,theta_max,N,S,control_plot,l_plot,delta_theta,OS_rate)

delta = 1/delta_inv;
i_set = 1:delta_inv;
theta = theta_min + (delta_theta/2) + (i_set-1)*delta*(theta_max-theta_min);
A_BS = exp(1j*pi*(0:N-1)'*sin(theta));

if delta_inv > N
    A_BS_pinv = (A_BS*A_BS'+10*eye(N))\A_BS;
else
    A_BS_pinv = pinv(A_BS');
end

w_D = {};
for l=1:S
    range_int = (theta_max-theta_min)/2^l;
    for k=1:2^l
        theta_range_min = theta_min + (k-1)*range_int;
        theta_range_max = theta_min + k*range_int;
        g = zeros(delta_inv,1);
        g(theta>=theta_range_min&theta<theta_range_max) = 1;
        wd_temp = A_BS_pinv*g;
        wd_temp = wd_temp*(sqrt(1)/norm(wd_temp));
        %%% Plot Section
        if control_plot ==1
            if l == l_plot
                rho = abs(A_BS'*wd_temp);
                polarplot(theta,rho);
                hold on;
            end
        end            
        w_D{l}{k} = wd_temp;
    end
end

delta_inv_OS = delta_inv*OS_rate;
i_set_OS = 1:delta_inv_OS;
delta_OS = 1/delta_inv_OS;
delta_theta_OS = (theta_max-theta_min)/delta_inv_OS;
theta_OS = theta_min + (delta_theta_OS/2) + (i_set_OS-1)*delta_OS*(theta_max-theta_min);
A_BS_OS =  exp(1j*pi*(0:N-1)'*sin(theta_OS));
