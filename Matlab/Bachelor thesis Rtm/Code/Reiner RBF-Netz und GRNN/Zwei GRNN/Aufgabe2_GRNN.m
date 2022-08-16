u=linspace(-8,8,101);
n=length(u);
u1=linspace(-8,0,51);
n1=length(u1);
u2=linspace(0,8,51);
n2=length(u2);

for i=1:n
    if u(i)>0
        Output_GRNN2(i)=0.29*atan(0.3*u(i))+0.4;
    else
        Output_GRNN2(i)=0.35*atan(0.3*u(i))-0.15;
    end
end %Gegebene und gesuchte Funktion

sigma_norm=0.45; %Glaettungsfaktor
p=11;
p2=(p+1)/2;
eta=0.001; %Lernschrittweite mit 0.001 bestimmen

[Gewichte1,Gewichte2,ksi1,ksi2,delta_ksi]=GRNN2(u,n,Output_GRNN2,sigma_norm,p2,eta);

Zaehler_AF_GRNN=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2));
Nenner_AF_GRNN=@(u,ksi) sum(Zaehler_AF_GRNN(u,ksi));
AF_GRNN=@(u,ksi) Zaehler_AF_GRNN(u,ksi)/Nenner_AF_GRNN(u,ksi);

subplot(1,2,1)

Nenner_Vektor1=zeros(1,n1);
Nenner_Vektor2=zeros(1,n2);
for i=1:n1
    Nenner_Vektor1(i)=sum(Nenner_AF_GRNN(u1(i),ksi1));
end
for i=1:n2
    Nenner_Vektor2(i)=sum(Nenner_AF_GRNN(u2(i),ksi2));
end

for i=1:p2
    A1=Gewichte1(i)*Zaehler_AF_GRNN(u1,ksi1(i))./Nenner_Vektor1;
    plot(u1,A1);
    hold on
    A2=Gewichte2(i)*Zaehler_AF_GRNN(u2,ksi2(i))./Nenner_Vektor2;
    plot(u2,A2);
    hold on
end
title('Aktivierungsfunktionen mal Gewichte')
xlabel('Anregung u')
xlim([-8 8])
hold off

subplot(1,2,2)

for i=1:p2
    A1=Zaehler_AF_GRNN(u1,ksi1(i))./Nenner_Vektor1;
    plot(u1,A1);
    hold on
    A2=Zaehler_AF_GRNN(u2,ksi2(i))./Nenner_Vektor2;
    plot(u2,A2);
    hold on
end
title('Aktivierungsfunktionen')
xlabel('Anregung u')
ylim([-5 5])
xlim([-8 8])
hold off

figure(2)
plot(ksi1,Gewichte1,'o')
hold on
plot(ksi2,Gewichte2,'o')
hold on

for i=1:n1
    Output_GRNN2_1(i)=0.35*atan(0.3*u1(i))-0.15;
end
for i=1:n2
    Output_GRNN2_2(i)=0.29*atan(0.3*u2(i))+0.4;
end
u_sum=[u1 u2];
Output_GRNN2_sum=[Output_GRNN2_1 Output_GRNN2_2];
plot(u_sum,Output_GRNN2_sum,'-.')
hold on

for i=1:n1
    prob_y1(i)=sum(Gewichte1.*AF_GRNN(u1(i),ksi1));
end
for i=1:n2
    prob_y2(i)=sum(Gewichte2.*AF_GRNN(u2(i),ksi2));
end
prob_y_sum=[prob_y1 prob_y2];
plot(u_sum,prob_y_sum,'-')

title('Lernergebnis GRNN')
xlabel('Anregung u')
legend('Gewichte negative','Gewichte positive','Wahre Kennlinie','Gelernte Kennlinie')
hold off