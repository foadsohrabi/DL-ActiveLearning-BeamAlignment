clc;
close all;
clear all;
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];

snrdB = -10:5:25;

load('data_baselines_hiePM_hieBS_OMPrandom_hyb.mat');
perf_bisec = mse_bisec;
perf_OMP = mse_OMP;
perf_AL_perfect = mse_AL_perfect_OS;

load('data_DNN_known_alpha_EST_hybrid_norm.mat');
performance_Hyb_norm = performance;

load('data_DNN_unknown_alpha_Kalman_EST_hyb_norm.mat');
performance_Kalman_Hyb_norm = performance;

load('data_DNN_unknown_alpha_MMSE_updatePIs_EST_hyb_norm.mat');
performance_MMSE_updatePI_Hyb_norm = performance;

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');

semilogy(snrdB,perf_bisec,'-.k','linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,perf_OMP,'-','color',color3,'linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,perf_AL_perfect,'-r','linewidth',3,'markersize',6);
hold on;
semilogy(snrdBvec,performance_MMSE_updatePI_Hyb_norm,'--ob','linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,performance_Kalman_Hyb_norm,'-.>','color',color6,'linewidth',2,'markersize',7);
hold on;
semilogy(snrdB,performance_Hyb_norm,'-s','color',color1,'lineWidth',2,'markersize',8);
hold on

grid;
fs2 = 14;
h = xlabel('SNR(dB)','FontSize',fs2);
get(h)
h = ylabel('Average MSE: $E \{ (\phi - \hat{\phi})^2 \}$','FontSize',fs2);
get(h)
lg = legend({'hieBS',... 
             'OMP w$/$ random fixed beamforming',...
             'hiePM w$/$ known $\alpha$',...
             'Proposed DNN w$/$ MMSE estimation for $\alpha$',...
             'Proposed DNN w$/$ Kalman tracking for $\alpha$',...
             'Proposed DNN w$/$ known $\alpha$'},'Interpreter','latex','Location','southeast');
set(lg,'Fontsize',fs2-2);
set(lg,'Location','southwest');
xlim([-10,25])
xticks(-10:2.5:25)

