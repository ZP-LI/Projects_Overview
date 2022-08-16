function [Volume, Surface, Circumference] = SphereData(r)
Volume = cir_Volume;
Surface = cir_Surface;
Circumference = cir_Circumference;
    
    function y = cir_Volume
        y = 4/3*pi*r^3;
    end

    function y = cir_Surface
        y = 4*pi*r^2;
    end

    function y = cir_Circumference
        y = 2*pi*r;
    end

end