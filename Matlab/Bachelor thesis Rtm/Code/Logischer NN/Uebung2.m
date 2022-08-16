x={[1 5],[1 3],[1 8],[1 1.4],[1 10.1]};
t=[5500 2300 7600 1800 11400];
eta=0.001;
iteration=30;

[weights,omega_figure,y1_figure]=NN_app_Linear(x,t,eta,iteration);

disp('weights are')
disp(weights)

subplot(2,1,1)
plot(omega_figure)
xlabel('Iteration')
ylabel('Gewicht')
legend('bias','Gewicht1')

subplot(2,1,2)
plot(y1_figure)
xlabel('Iteration')
ylabel('y1')