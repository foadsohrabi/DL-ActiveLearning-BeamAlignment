function PI_new = func_update_PI(PI,y,alpha_hat,P,w,A_BS)
% Asumption sigma2 = 1;

mean = transpose(alpha_hat*sqrt(P)*w'*A_BS);
f = exp(-(abs(y - mean).^2)) + 0.00000001; %the small number is added to avoid numerical errors
PI_new = f.*PI/(f'*PI);



