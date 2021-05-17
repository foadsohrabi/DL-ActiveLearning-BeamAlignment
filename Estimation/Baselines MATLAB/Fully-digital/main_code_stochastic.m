clc;
close all;
clear all;

%%%%%%%%%%%% Parameters
N = 64; %Number of BS's antennas
delta_inv = 128; %Number of posterior intervals inputed to DNN 
theta_min = -60*(pi/180); %Lower-bound of AoAs
theta_max = 60*(pi/180); %Upper-bound of AoAs
snrdB = -10:5:25; %Set of SNRs
Pvec = 10.^(snrdB./10); %Set of considered TX powers
S = log2(delta_inv);%Number of stages in hierarchical binary search 
tau = 2*S; %Pilot length
mean_true_alpha = 0.0 + 0.0j; %Mean of the fading coefficient
std_per_dim_alpha = sqrt(0.5); %STD of the Gaussian gain per real dim.
noiseSTD_per_dim = sqrt(0.5); %STD of the Gaussian noise per real dim.
delta_theta = (theta_max-theta_min)/delta_inv; %distance betwewn samples
OS_rate = 20; %over sampling rate
%%%%%%%%%%%% Cnt_params
ch_num = 128*782; %Almost 10^5
control_plot = 0;
l_plot = 7;
%%%%%%%%%%%%
%%%%  Hierarchical Codebook Design
[w_D,A_BS,A_BS_pinv,theta,A_BS_OS,theta_OS] =func_codedesign(delta_inv,theta_min,theta_max,N,S,control_plot,l_plot,delta_theta,OS_rate);
%%%% Random Sensing Design for OMP
W_her_OMP = randn(tau,N)+1j*randn(tau,N);
for t = 1:tau
    W_her_OMP(t,:) = W_her_OMP(t,:)*(sqrt(1)/norm(W_her_OMP(t,:)));
end
A = W_her_OMP*A_BS_OS;    
if control_plot == 1
    figure()
    imagesc((abs(A'*A)));
    title({'Random: $|A^H * A|$'}, 'Interpreter','latex');
    figure()
    imagesc(exp(abs(A'*A)));
    title({'Random: $exp(|A^H * A|)$'}, 'Interpreter','latex');
end
%%%%%%%%%%%%
mse_bisec = zeros(length(Pvec),ch_num);
mse_AL_perfect_OS = zeros(length(Pvec),ch_num);
mse_OMP = zeros(length(Pvec),ch_num);
for ch = 1:ch_num
    disp(ch);
    theta_continous = (rand(1,1)-0.5)*(theta_max-theta_min);
    h = exp(1j*pi*(0:N-1)'*sin(theta_continous));
    alpha = mean_true_alpha +(std_per_dim_alpha*(randn(1,1) +1j*randn(1,1)));
    noise_mat = noiseSTD_per_dim*(randn(1,tau) +1j*randn(1,tau));
    for pp = 1:length(Pvec)
        P = Pvec(pp);
        %%%%%%%%%%%%% Bisection
        idx_hat1 = func_alg_bisection(S,w_D,P,alpha, h,noise_mat);
        theta_hat_1 = theta(idx_hat1);
        mse_bisec(pp,ch) = (theta_hat_1-theta_continous)^2;                
        %%%%%%%%%%%%% Active Learning hiePM - known alpha
        alpha_hat = alpha;        
        idx_hat2 = func_alg_active_learning_OS(control_plot,delta_inv,S,tau,alpha,alpha_hat,w_D,noise_mat,P,h,OS_rate,A_BS_OS);
        theta_hat2 = theta_OS(idx_hat2);
        mse_AL_perfect_OS(pp,ch) = (theta_hat2-theta_continous)^2; 
        %%%%%%%%%%%%%%%%% OMP
        Y = sqrt(P)*alpha*W_her_OMP*h + transpose(noise_mat);
        [~,idx_hat_omp] = max(abs(A'*Y));
        theta_hat_omp = theta_OS(idx_hat_omp);
        mse_OMP(pp,ch) = (theta_hat_omp-theta_continous)^2;  
    end
    
end
mse_bisec = mean(mse_bisec,2);
mse_AL_perfect_OS = mean(mse_AL_perfect_OS,2);
mse_OMP = mean(mse_OMP,2);

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');

semilogy(snrdB,mse_bisec,'-k','linewidth',3,'markersize',8);
hold on;
semilogy(snrdB,mse_OMP,'-g','linewidth',3,'markersize',8);
hold on;
semilogy(snrdB,mse_AL_perfect_OS,'-b','linewidth',3,'markersize',8);
hold on;
grid;
fs2 = 12;
h = xlabel('SNR(dB)','FontSize',fs2);
get(h);
h = ylabel('Probability of Detection Error: $P(\hat{\phi} \not = \phi)$','FontSize',fs2);
get(h);
h = ylabel('Average MSE: $E [ (\phi - \hat{\phi})^2 ]$','FontSize',fs2);
get(h);
lg = legend({'hieBS', 'OMP w$/$ random fixed beamforming',...
    'hiePM w$/$ known $\alpha$'},'Interpreter','latex','Location','southeast');
set(lg,'Fontsize',fs2);
set(lg,'Location','southwest');

save('data_baselines_hiePM_hieBS_OMPrandom.mat','N','delta_inv','theta_min','theta_max','snrdB',...
     'Pvec','S','tau','mean_true_alpha','std_per_dim_alpha','noiseSTD_per_dim','ch_num',...
     'mse_bisec','mse_OMP','mse_AL_perfect_OS')

