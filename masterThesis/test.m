phase = linspace(0,2*pi,200);

for p = 1:length(phase)
    if phase(p)<pi
        x(p) = 0 + 0.03*cos(phase(p));
        y(p) = -0.045 + 0.01*sin(phase(p));
    else
        x(p) = 0 + 0.03*cos(phase(p));
        y(p) = -0.045 + 0.005*sin(phase(p));
    end
end

plot(x,y)