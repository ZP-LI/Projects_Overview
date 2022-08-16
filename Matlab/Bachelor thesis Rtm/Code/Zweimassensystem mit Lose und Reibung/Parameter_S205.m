J1=0.166; %kg*m^2
J2=0.336; %kg*m^2

d=0.6; %Nm*s/rad
c=1160; %Nm/rad
a=0.031; %rad
theta_S1=2; %Nm*s/rad
theta_S2=38; %Nm*s/rad

l1=3.3;
l2=2.6;
l3=-1197;
l4=98;
l5=227; %Beobachterr¨¹ckf¨¹hrungen

sigma_norm=1.6;
r=15;

alpha=0.95;
eta_J1=0.15;
eta_Reib1=0.125;
eta_d=1.25;
eta_c=5*10^4;
eta_theta_L=8*10^(-6);
eta_theta_S1=5;
eta_theta_S2=10;
eta_J2=4*10^(-3);
eta_Reib2=0.1; %Lernparameter

delta_ksi=(20-(-20))/(2*r-1);
ksi=linspace(-20,20,2*r);

h=0.4*10^(-3); %Abtastzeit