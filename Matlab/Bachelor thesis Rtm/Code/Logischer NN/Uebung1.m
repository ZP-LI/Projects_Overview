iteration=10;
eta=0.1;
t=[0 1 1 1];
x={[0 0], [0 1], [1 0], [1 1]};

[weights, bias]=NN_app_Ordnung2(x,t,eta,iteration);

disp('weights are')
disp(weights)
disp('bias is')
disp(bias)

for i=1:4
prob_x=x{i};
y=NN_impl(prob_x,weights,bias);
disp(prob_x)
disp('=')
disp(y)
end