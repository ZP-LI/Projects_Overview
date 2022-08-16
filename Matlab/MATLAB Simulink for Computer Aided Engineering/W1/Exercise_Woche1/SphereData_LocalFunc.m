function [Volume, Surface, Circumference] = SphereData_LocalFunc(r)
Volume = cir_Volume(r);
Surface = cir_Surface(r);
Circumference = cir_Circumference(r);
end
    
function y = cir_Volume(x)
y = 4/3*pi*x^3;
end

function y = cir_Surface(x)
y = 4*pi*x^2;
end

function y = cir_Circumference(x)
y = 2*pi*x;
end
