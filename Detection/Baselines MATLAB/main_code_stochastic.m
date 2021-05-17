clc;
close all;
clear all;

%%%%%%%%%%%% Parameters
N = 64; %Number of BS's antennas
delta_inv = 128; %Number of posteriors inputed to DNN 
theta_min = -60*(pi/180); %Lower-bound of AoAs
theta_max = 60*(pi/180); %Upper-bound of AoAs
snrdB = -10:5:30; %Set of SNRs
Pvec = 10.^(snrdB./10); %Set of considered TX powers
S = log2(delta_inv);%Number of stages in hierarchical binary search 
tau = 2*S; %Pilot length
mean_true_alpha = 0.0 + 0.0j; %Mean of the fading coefficient
std_per_dim_alpha = sqrt(0.5); %STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = sqrt(0.5); %STD of the Gaussian noise per real dim.
%%%%%%%%%%%% Cnt_params
ch_num = 128*782; %Almost 10^5
control_plot = 0;
l_plot = 7;
%%%%%%%%%%%%
%%%%  Hierarchical Codebook Design
[w_D,A_BS,A_BS_pinv,theta] =func_codedesign(delta_inv,theta_min,theta_max,N,S,control_plot,l_plot);
%%%% Random Sensing Design for OMP
W_her_OMP = randn(tau,N)+1j*randn(tau,N);
for t = 1:tau
    W_her_OMP(t,:) = W_her_OMP(t,:)*(sqrt(1)/norm(W_her_OMP(t,:)));
end
A = W_her_OMP*A_BS;    
if control_plot == 1
    figure()
    imagesc((abs(A'*A)));
    title({'Random: $|A^H * A|$'}, 'Interpreter','latex');
    figure()
    imagesc(exp(abs(A'*A)));
    title({'Random: $exp(|A^H * A|)$'}, 'Interpreter','latex');
end
%%%%%%%%%%%%
perf_bisec = zeros(length(Pvec),1);
perf_AL_perfect = zeros(length(Pvec),1);
perf_OMP = zeros(length(Pvec),1);
for ch = 1:ch_num
    disp(ch);
    idx = randi(delta_inv,1);
    alpha = mean_true_alpha +(std_per_dim_alpha*(randn(1,1) +1j*randn(1,1)));
    h = A_BS(:,idx);
    noise_mat = noiseSTD_per_dim*(randn(1,tau) +1j*randn(1,tau));
    for pp = 1:length(Pvec)
        P = Pvec(pp);
        %%%%%%%%%%%%% Bisection
        idx_hat1 = func_alg_bisection(S,w_D,P,alpha, h,noise_mat);
        if idx ~= idx_hat1
            perf_bisec(pp) = perf_bisec(pp)+1;
        end           
        %%%%%%%%%%%%% Active Learning hiePM - known alpha
        alpha_hat = alpha;
        idx_hat2 = func_alg_active_learning(control_plot,delta_inv,S,tau,alpha,alpha_hat,w_D,noise_mat,P,h,A_BS);
        if idx ~= idx_hat2
            perf_AL_perfect(pp) = perf_AL_perfect(pp)+1;
        end
        %%%%%%%%%%%%%%%%% OMP
        Y = sqrt(P)*alpha*W_her_OMP*h + transpose(noise_mat);
        [~,idx_hat_omp] = max(abs(A'*Y));
        if idx ~= idx_hat_omp
            perf_OMP(pp) = perf_OMP(pp)+1;
        end    
    end
    
end
figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
fs2 = 14;
semilogy(snrdB,perf_bisec/ch_num,'-k','linewidth',3,'markersize',8);
hold on;
semilogy(snrdB,perf_OMP/ch_num,'-g','linewidth',3,'markersize',8);
hold on;
semilogy(snrdB,perf_AL_perfect/ch_num,'-b','linewidth',3,'markersize',8);
hold on;
grid;
h = xlabel('SNR(dB)','FontSize',fs2);
get(h)
h = ylabel('Probability of Detection Error: $P(\hat{\phi} \not = \phi)$','FontSize',fs2);
get(h);
lg = legend({'hieBS', 'OMP w$/$ random fixed beamforming',...
    'hiePM w$/$ known $\alpha$'},'Interpreter','latex','Location','southeast');
set(lg,'Fontsize',fs2);
set(lg,'Location','southwest');

save('data_baselines_hiePM_hieBS_OMPrandom.mat','N','delta_inv','theta_min','theta_max','snrdB',...
     'Pvec','S','tau','mean_true_alpha','std_per_dim_alpha','noiseSTD_per_dim','ch_num',...
     'perf_bisec','perf_OMP','perf_AL_perfect')

