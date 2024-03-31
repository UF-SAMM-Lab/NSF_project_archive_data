clear all
close all
clc

%% Data

% time002HITL = [22.50,20.59,20.60,20.28,20.26,20.25,20.62,20.94,20.48,20.37,20.29,20.08,20.34,19.82,19.42,19.82];
% dist002HITL = [0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time002 = [20.58,20.67,20.52,20.33,20.90,21.11,20.53,21.01,20.74,20.63,20.41,20.26,20.31,20.00,19.82,19.65];
% dist002 = [0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time003 = [20.00,20.27,20.54,20.19,20.74,20.76,20.49,20.58,20.36,20.05,20.44,20.20,20.01,20.01,20.12];
% dist003 = [0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time004 = [20.41,20.05,19.98,20.02,20.73,20.44,20.15,20.27,20.11,19.70,19.99,19.96,19.59,19.65];
% dist004 = [0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time005 = [19.86,19.85,20.38,20.36,20.36,20.39,20.17,19.91,19.85,19.82,19.77,19.56,19.71];
% dist005 = [0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time006 = [19.59,19.99,20.30,19.94,20.44,20.37,19.89,19.90,19.72,19.46,19.64,19.29];
% dist006 = [0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time007 = [19.72,19.90,19.82,19.77,20.24,19.97,19.75,19.92,19.56,19.87,19.77];
% dist007 = [0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time008 = [19.57,20.00,19.38,19.96,19.65,20.07,19.82,19.57,19.51,19.57];
% dist008 = [0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];
% 
% time009 = [20.02,19.84,19.79,19.69,19.73,19.75,19.70,19.58,19.17];
% dist009 = [0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time002HITL = [22.50,20.59,20.60,20.28,20.26,20.25,20.62,20.94,20.48,20.37,20.29,20.08,20.34,19.82,19.42,19.82];
dist002HITL = [0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time002 = [20.58,20.67,20.52,20.33,20.90,21.11,20.53,21.01,20.74,20.63,20.41,20.26,20.31,20.11,20.00,19.79];
dist002 = [0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time003 = [20.00,20.27,20.54,20.19,20.74,20.76,20.49,20.58,20.36,20.05,20.44,20.20,20.01,19.91,19.83];
dist003 = [0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time004 = [20.41,20.05,19.98,20.02,20.73,20.44,20.15,20.27,20.11,19.70,19.99,19.96,19.59,19.65];
dist004 = [0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time005 = [19.66,19.85,20.38,20.36,20.36,20.39,20.17,19.91,19.85,19.82,19.77,19.56,19.65];
dist005 = [0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time006 = [19.59,19.99,20.30,19.94,20.44,20.37,19.89,19.90,19.72,19.46,19.64,19.29];
dist006 = [0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time007 = [19.72,19.90,19.82,19.77,20.24,19.97,19.75,19.92,19.56,19.87,19.77];
dist007 = [0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time008 = [19.57,20.00,19.38,19.96,19.65,20.07,19.82,19.57,19.51,19.57];
dist008 = [0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

time009 = [20.02,19.84,19.79,19.69,19.73,19.75,19.70,19.58,19.52];
dist009 = [0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01];

%% Comparison

figure
grid on
axis tight
hold on
comp = plot(dist002HITL,time002HITL,dist002,time002)
title('Hardware-in-the-loop vs Simulation','Fontsize',15)
legend('HITL','Simulation')
xcomp = xlabel('Minimum distance [m]');
ycomp = ylabel('Average time [s]');
set(xcomp,'Interpreter','latex','Fontsize',13)
set(ycomp,'Interpreter','latex','Fontsize',13)
set(comp,'Linewidth',1.5)

figure
grid on
axis tight
hold on
comp = plot(dist002,time002,dist003,time003,dist004,time004,dist005,time005,dist006,time006,dist007,time007,dist008,time008,dist009,time009);
title('Theshold vs Inflation','Fontsize',15)
legend('Thr-Infl=0.02','Thr-Infl=0.03','Thr-Infl=0.04','Thr-Infl=0.05','Thr-Infl=0.06','Thr-Infl=0.07','Thr-Infl=0.08','Thr-Infl=0.09')
xcomp = xlabel('Minimum distance [m]');
ycomp = ylabel('Average time [s]');
set(xcomp,'Interpreter','latex','Fontsize',13)
set(ycomp,'Interpreter','latex','Fontsize',13)
set(comp,'Linewidth',1.5)

%% Interpolation

x2HITL = [0.01:0.001:0.16];
p2HITL = polyfit(dist002HITL,time002HITL,7);
y2HITL = polyval(p2HITL,x2HITL);
x2SIM = [0.01:0.001:0.16];
p2SIM = polyfit(dist002,time002,6);
y2SIM = polyval(p2SIM,x2SIM);


x2 = [0.01:0.001:0.09];
p2 = polyfit(dist002,time002,6);
y2 = polyval(p2,x2);
x3 = [0.01:0.001:0.09];
p3 = polyfit(dist003,time003,6);
y3 = polyval(p3,x3);
x4 = [0.01:0.001:0.09];
p4 = polyfit(dist004,time004,6);
y4 = polyval(p4,x4);
x5 = [0.01:0.001:0.09];
p5 = polyfit(dist005,time005,6);
y5 = polyval(p5,x5);
x6 = [0.01:0.001:0.09];
p6 = polyfit(dist006,time006,7);
y6 = polyval(p6,x6);
x7 = [0.01:0.001:0.09];
p7 = polyfit(dist007,time007,7);
y7 = polyval(p7,x7);
x8 = [0.01:0.001:0.09];
p8 = polyfit(dist008,time008,7);
y8 = polyval(p8,x8);
x9 = [0.01:0.001:0.09];
p9 = polyfit(dist009,time009,6);
y9 = polyval(p9,x9);

figure
plotHITL = plot(x2HITL,y2HITL,x2SIM,y2SIM);
title('Hardware-in-the-loop vs Simulation','Fontsize',15)
legend('HITL','Simulation')
xaxisHITL = xlabel('Minimum distance [m]');
yaxisHITL = ylabel('Average time [s]');
set(xaxisHITL,'Interpreter','latex','Fontsize',13)
set(yaxisHITL,'Interpreter','latex','Fontsize',13)
set(plotHITL,'Linewidth',1.5)
axis tight
grid on
figure

plotALL = plot(x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9);
title('Theshold vs Inflation','Fontsize',15)
legend('Thr-Infl=0.02','Thr-Infl=0.03','Thr-Infl=0.04','Thr-Infl=0.05','Thr-Infl=0.06','Thr-Infl=0.07','Thr-Infl=0.08','Thr-Infl=0.09')
xaxisALL = xlabel('Minimum distance [m]');
yaxisALL = ylabel('Average time [s]');
set(xaxisALL,'Interpreter','latex','Fontsize',13)
set(yaxisALL,'Interpreter','latex','Fontsize',13)
set(plotALL,'Linewidth',1.5)
axis tight
grid on

green = [0 0.63 0.29];
lightblue = [0.30 0.59 0.82];
blue = [0.09 0.28 0.62];
red = [0.74 0.12 0.18];
orange = [0.94 0.39 0.13];

figure
plot(x2,y2-4,'LineWidth',2.5,'Color',red)
hold on
plot(x3,y3-4,'--','LineWidth',2.5,'Color',blue)
plot(x4,y4-4,':','LineWidth',4,'Color',green)
plot(x5,y5-4,'-.','LineWidth',3.5,'Color',orange)
plot(x9,y9-4,'o','LineWidth',0.8,'Color',lightblue);
plot(x9,y9-4,'k');
title('Experimental campaign - All curves','Fontsize',15)
lgd = legend('d = 0.02 m','d = 0.03 m','d = 0.04 m','d = 0.05 m','d = 0.09 m');
xaxis = xlabel('Inflation [m]');
yaxis = ylabel('Average execution time [s]');
set(xaxis,'Interpreter','latex','Fontsize',13)
set(yaxis,'Interpreter','latex','Fontsize',13)
set(lgd,'FontSize',15)
axis tight
grid on

figure
plot(x2,y2-4,'LineWidth',2.5,'Color',red)
hold on
plot(x3,y3-4,'--','LineWidth',2.5,'Color',blue)
plot(x4,y4-4,':','LineWidth',4,'Color',green)
plot(x5,y5-4,'-.','LineWidth',3.5,'Color',orange)
plot(x9,y9-4,'o','LineWidth',0.8,'Color',lightblue);
plot(x9,y9-4,'k');
title('Experimental campaign','Fontsize',14)
lgd = legend('d = 0.02 m','d = 0.03 m','d = 0.04 m','d = 0.05 m','d = 0.09 m');
xaxis = xlabel('Inflation [m]');
yaxis = ylabel('Average execution time [s]');
set(xaxis,'Interpreter','latex','Fontsize',12)
set(yaxis,'Interpreter','latex','Fontsize',12)
set(lgd,'FontSize',12)
axis tight
grid on

figure
hold on
plot(dist002(8:end),time002(8:end)-4,'LineWidth',2.5,'Color',red);
plot(dist009,time009-4,'LineWidth',2.5,'Color',lightblue);
plot(dist002(8:end),time002(8:end)-4,'ok','LineWidth',1.7);
plot(dist009,time009-4,'ok','LineWidth',1.7);
title('Experimental campaign - Extreme curves','Fontsize',15)
lgd = legend('d = 0.02 m','d = 0.09 m');
xaxis = xlabel('Inflation [m]');
yaxis = ylabel('Average execution time [s]');
set(xaxis,'Interpreter','latex','Fontsize',13)
set(yaxis,'Interpreter','latex','Fontsize',13)
set(lgd,'FontSize',15)
axis tight
grid on

%% Safety & Productivity proposal

% figure
% 
% grid on
% axis tight
% hold on
% a = normalize(sqrt(dist002HITL),'range');
% b = normalize(1./time002HITL,'range');
% hp = plot(a,b,'o');
% [dist,time] = prepareCurveData(a,b);
% [interp,gof] = fit(dist,time,'smoothingspline');
% hpp = plot(interp);
% 
% title('Safety vs Productivity','Fontsize',15)
% legend('Experimental data','Fitted curve')
% x = xlabel('Safety [$\sqrt{m}$]');
% y = ylabel('Productivity [$\frac{1}{s}$]');
% set(x,'Interpreter','latex','Fontsize',13)
% set(y,'Interpreter','latex','Fontsize',13)
% set(hp,'Linewidth',1.5)
% set(hpp,'Linewidth',1.5)

% figure
% 
% grid on
% axis tight
% hold on
% hp2 = plot(sqrt(min_distance),1./avg_time,'o');
% [dist2,time2] = prepareCurveData(sqrt(min_distance),1./avg_time);
% [interp2,gof2] = fit(dist2,time2,'poly2');
% hpp2 = plot(interp2);
% 
% title('Safety vs Productivity','Fontsize',15)
% legend('Experimental data','Fitted curve')
% x2 = xlabel('Safety [$\sqrt{m}$]');
% y2 = ylabel('Productivity [$\frac{1}{s}$]');
% set(x2,'Interpreter','latex','Fontsize',13)
% set(y2,'Interpreter','latex','Fontsize',13)
% set(hp2,'Linewidth',1.5)
% set(hpp2,'Linewidth',1.5)
