clc;
close all;
clear all;
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];

load('data_baselines_hiePM_hieBS_OMPrandom.mat');
perf_bisec = perf_bisec/ch_num;
perf_OMP = perf_OMP/ch_num;
perf_AL_perfect = perf_AL_perfect/ch_num;

load('data_DNN_known_alpha.mat');
performance_known_alpha = performance(:,end);

load('data_DNN_unknown_alpha_MMSE_updatePIs.mat');
performance_MMSE_updatePI = performance;

load('data_DNN_unknown_alpha_Kalman.mat');
performance_Kalman = performance;

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');

semilogy(snrdB,perf_bisec,'-.k','linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,perf_OMP,'-','color',color3,'linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,perf_AL_perfect,'-r','linewidth',3,'markersize',6);
hold on;
semilogy(snrdBvec,performance_Kalman(:,end),'-->c','linewidth',2,'markersize',6);
hold on;
semilogy(snrdBvec,performance_MMSE_updatePI(:,end),'--ob','linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,performance_known_alpha,'-s','color',color1,'lineWidth',2,'markersize',8);
hold on

grid;
fs2 = 14;
h = xlabel('SNR(dB)','FontSize',fs2);
get(h)
h = ylabel('Probability of Detection Error: $P(\hat{\phi} \not = \phi)$','FontSize',fs2);
get(h)
lg = legend({'hieBS',... 
             'OMP w$/$ random fixed beamforming',...
             'hiePM w$/$ known $\alpha$',...
             'Proposed DNN w$/$ Kalman tracking for $\alpha$',...
             'Proposed DNN w$/$ MMSE estimation for $\alpha$',...
             'Proposed DNN w$/$ known $\alpha$'},'Interpreter','latex','Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Location','southwest');
xlim([-10,25])
xticks(-10:2.5:25)

