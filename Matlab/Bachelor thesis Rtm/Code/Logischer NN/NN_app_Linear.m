function [weights,omega_figure,y1_figure]=NN_app_Linear(x,t,eta,iteration)

m=size(x);
n=m(2);

o=size(x{1});
p=o(2);

omega(1,p)=0;
delta_omega(1,p)=0;
omega_figure=omega;
y1_figure=sum(x{1}.*omega)

for i=1:iteration
    for j=1:n
        delta_omega1=eta*(t(j)-sum(omega.*x{j}))*x{j};
        delta_omega=delta_omega+delta_omega1;
    end
    omega=omega+delta_omega;
    omega_figure(i+1,:)=omega;
    delta_omega(1,p)=0;
    y1_figure(i+1,:)=sum(x{1}.*omega);
end;

weights=omega;

end