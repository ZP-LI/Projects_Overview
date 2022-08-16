u=linspace(-8,8,101);
u2=linspace(-10,10,201);
n2=length(u2);
Output_RBF=3*atan(2*u); %Gegebene und gesuchte Funktion

sigma_norm=0.45; %Glaettungsfaktor
p=11;
eta=0.001; %Lernschrittweite mit 0.001 bestimmen

[Gewichte,ksi,delta_ksi,n]=GRNN(u,Output_RBF,sigma_norm,p,eta);

Zaehler_AF_GRNN=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2));
Nenner_AF_GRNN=@(u,ksi) sum(Zaehler_AF_GRNN(u,ksi));
AF_GRNN=@(u,ksi) Zaehler_AF_GRNN(u,ksi)/Nenner_AF_GRNN(u,ksi);

figure(1)
Nenner_Vektor=zeros(1,n);
for i=1:n
    Nenner_Vektor(i)=sum(Nenner_AF_GRNN(u(i),ksi));
end
for i=1:p
    A=Gewichte(i)*Zaehler_AF_GRNN(u,ksi(i))./Nenner_Vektor;
    plot(u,A);
    hold on
end
title('Aktivierungsfunktionen mal Gewichte')
xlabel('Anregung u')
xlim([-8 8])
hold off

figure(2)
Nenner_Vektor=zeros(1,n);
for i=1:n
    Nenner_Vektor(i)=sum(Nenner_AF_GRNN(u(i),ksi));
end
for i=1:p
    A=Zaehler_AF_GRNN(u,ksi(i))./Nenner_Vektor;
    plot(u,A);
    hold on
end
title('Aktivierungsfunktionen')
xlabel('Anregung u')
ylim([-5 5])
xlim([-8 8])
hold off

figure(3)
plot(ksi,Gewichte,'o');
hold on
plot(u2,3*atan(2*u2),'-.');
hold on
for i=1:n2
    prob_y(i)=sum(Gewichte.*AF_GRNN(u2(i),ksi));
end
plot(u2,prob_y,'-');
title('Lernergebnis GRNN')
xlabel('Anregung u')
legend('Gewichte','Vorgegebene Kennlinie','Identifizierte Kennlinie')
hold off