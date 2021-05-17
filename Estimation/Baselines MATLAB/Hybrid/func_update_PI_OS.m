function [PI_new,PI_OS_new] = func_update_PI_OS(PI,PI_OS,y,alpha_hat,P,w,A_BS_OS,OS_rate)
% Asumption sigma2 = 1;

cand_size = length(PI);

mean = transpose( alpha_hat*sqrt(P)*w'*A_BS_OS);
f = exp(-abs(y-mean).^2)+ 0.00000001; %the small number is added to avoid numerical errors
PI_OS_new = (f.*PI_OS)/(f'*PI_OS);


PI_temp = reshape(PI_OS_new,OS_rate,cand_size);
PI_new = sum(PI_temp)';



