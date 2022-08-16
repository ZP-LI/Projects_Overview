J=0.166; %kg*m^2

l=30; %Beobachterrueckfuehrung
h=10^(-3); %Abtastzeit

r=36; %Anzahl der Stuetzstellen
psi1=5; %1/(kg*m^2)

eta_linear=10^(-3);
alpha_linear=0.95;
for i=1:r
    alpha_NL(i)=0.95;
    eta_NL(i)=15*10^(-2);
end

sigma_norm=1.3;
delta_ksi=(20-(-20))/(r-1);
ksi=linspace(-20,20,36); %Parameter von Aktivierungsfunktion