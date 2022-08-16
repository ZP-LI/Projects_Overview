function [c, ceq] = nonlcon(x)

c = x(1)^2/4 - (5*x(1))/2 + x(2)^2 - 8*x(2) + 85/4;
ceq = [ ];

end