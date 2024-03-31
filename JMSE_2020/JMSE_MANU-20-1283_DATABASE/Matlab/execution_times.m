clear all
close all
clc

%% Data

segments = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
case1a = [0,5.6,2.8,0.1,2.8,4.5,2.5,0.1,2.5,0.1,2.7,0.1,2.7,4.1,0.1,2.5,0.1,2.5,4.3,5.6];
case2a = [0,5.6,2.8,0.1,2.8,7.3,2.5,0.1,2.5,0.1,2.7,0.1,2.7,5.2,0.1,2.5,0.1,2.5,4.3,5.6];
case3a = [0,5.6,2.8,0.1,2.8,19.1,2.5,0.1,2.5,0.1,2.7,0.1,2.7,5.1,0.1,2.5,0.1,2.5,4.3,5.6];
case1b = [0,5.6,2.8,0.1,2.8,6.0,2.5,0.1,2.5,0.1,2.7,0.1,2.7,5.5,0.1,2.5,0.1,2.5,4.3,5.6];
case2b = [0,5.6,2.8,0.1,2.8,8.8,2.5,0.1,2.5,0.1,2.7,0.1,2.7,6.7,0.1,2.5,0.1,2.5,4.3,5.6];
case3b = [0,5.6,2.8,0.1,2.8,20.8,2.5,0.1,2.5,0.1,2.7,0.1,2.7,6.7,0.1,2.5,0.1,2.5,4.3,5.6];

case1a_cum = cumsum(case1a);
case2a_cum = cumsum(case2a);
case3a_cum = cumsum(case3a);
case1b_cum = cumsum(case1b);
case2b_cum = cumsum(case2b);
case3b_cum = cumsum(case3b);

%% Plots

figure 
xticks([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]);
xticklabels({' ','Segment 1','Segment 2','Segment 3','Segment 4','Segment 5','Segment 6','Segment 7','Segment 8','Segment 9','Segment 10','Segment 11','Segment 12','Segment 13','Segment 14','Segment 15','Segment 16','Segment 17','Segment 18','Segment 19'});
xtickangle(45);
hold on
plot(segments,case1a_cum,'linewidth',2)
plot(segments,case2a_cum,'linewidth',2)
plot(segments,case3a_cum,'linewidth',2)
stem([5 5 5],[30.4 18.6 15.8],'--k','linewidth',1.5);
stem([13 13 13],[46.2 34.5 30.6],'--k','linewidth',1.5);
axis tight
grid on
t = title('Execution times - Customization A');
l = legend('CASE 1','CASE 2','CASE 3');
y = ylabel('Time [s]');
x = get(gca,'XTickLabel');
set(l,'fontsize',14);
set(t,'fontsize',14);
set(y,'fontsize',14);
set(gca,'XTickLabel',x,'fontsize',14);

figure 
xticks([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]);
hold on
plot(segments,case1a_cum,'linewidth',2)
plot(segments,case2a_cum,'--','linewidth',2)
plot(segments,case3a_cum,':','linewidth',2)
stem([5 5 5],[30.4 18.6 15.8],'--k','linewidth',1.5);
stem([13 13 13],[46.2 34.5 30.6],'--k','linewidth',1.5);
axis tight
grid on
t = title('Execution times - Customization A');
l = legend('CASE 1','CASE 2','CASE 3');
x = xlabel('Segment');
y = ylabel('Time [s]');
set(l,'fontsize',12);
set(t,'fontsize',14);
set(x,'fontsize',12)
set(y,'fontsize',12);

figure 
xticks([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]);
xticklabels({' ','Segment 1','Segment 2','Segment 3','Segment 4','Segment 5','Segment 6','Segment 7','Segment 8','Segment 9','Segment 10','Segment 11','Segment 12','Segment 13','Segment 14','Segment 15','Segment 16','Segment 17','Segment 18','Segment 19'});
xtickangle(45);
hold on
plot(segments,case1b_cum,'linewidth',2)
plot(segments,case2b_cum,'linewidth',2)
plot(segments,case3b_cum,'linewidth',2)
stem([5 5 5],[32.1 20.1 17.3],'--k','linewidth',1.5);
stem([13 13 13],[49.5 37.5 33.5],'--k','linewidth',1.5);
axis tight
grid on
t = title('Execution times - Customization B');
l = legend('CASE 1','CASE 2','CASE 3');
y = ylabel('Time [s]');
x = get(gca,'XTickLabel');
set(l,'fontsize',14);
set(t,'fontsize',14);
set(y,'fontsize',14);
set(gca,'XTickLabel',x,'fontsize',14);

figure 
xticks([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]);
xticklabels({' ','Segment 1','Segment 2','Segment 3','Segment 4','Segment 5','Segment 6','Segment 7','Segment 8','Segment 9','Segment 10','Segment 11','Segment 12','Segment 13','Segment 14','Segment 15','Segment 16','Segment 17','Segment 18','Segment 19'});
xtickangle(45);
hold on
plot(segments,case1a_cum,'r','linewidth',2)
plot(segments,case2a_cum,'g','linewidth',2)
plot(segments,case3a_cum,'b','linewidth',2)
plot(segments,case1b_cum,'--r','linewidth',2)
plot(segments,case2b_cum,'--g','linewidth',2)
plot(segments,case3b_cum,'--b','linewidth',2)
stem([5 5 5 5 5 5],[32.1 30.4 20.1 18.6 17.3 15.8],'--k','linewidth',1.5);
stem([13 13 13 13 13 13],[49.5 46.2 37.5 34.5 33.5 30.6],'--k','linewidth',1.5);
axis tight
grid on
t = title('Execution times - Comparison');
l = legend('CASE 1 - A','CASE 2 - A','CASE 3 - A','CASE 1 - B','CASE 2 - B','CASE 3 - B');
y = ylabel('Time [s]');
x = get(gca,'XTickLabel');
set(l,'fontsize',14);
set(t,'fontsize',14);
set(y,'fontsize',14);
set(gca,'XTickLabel',x,'fontsize',14);
