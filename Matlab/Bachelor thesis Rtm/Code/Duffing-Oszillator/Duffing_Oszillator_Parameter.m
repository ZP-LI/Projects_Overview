omega_0=6;
alpha_w=1;
beta_w=-1;
gamma_w=10;
delta_w=0.2;

h=0.001;
r=21;
l1=10;
l2=1.75;

for i=1:r
    eta_theta(i)=6*10^(-3);
    a_theta(i)=0.95;
end 

eta_alpha=7*10^(-2);
eta_delta=1.5*10^(-3);
a_alpha=0.95;
a_delta=0.95;

ksi=linspace(-0.3,0.3,r);
delta_ksi=(0.3-(-0.3))/(r-1);
sigma_norm=0.3;
