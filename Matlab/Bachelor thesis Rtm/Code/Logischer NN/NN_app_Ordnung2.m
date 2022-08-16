function [weights, bias]=NN_app_Ordnung2(x,t,eta,iteration)

Omega=[0 0];
b=0;

for i=1:1:iteration
    for j=1:4
        a=x{j}.*Omega;
        y=heaviside(a(1)+a(2)+b);
        
        delta_Omega=eta*(t(j)-y)*x{j};
        Omega=Omega+delta_Omega;
        
        delta_b=eta*(t(j)-y);
        b=b+delta_b;
    end
end

weights=Omega;
bias=b;