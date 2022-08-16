u=linspace(-8,8,1001);
u2=linspace(-10,10,2001);
n2=length(u2);
Output_RBF=3*atan(2*u); %Gegebene und gesuchte Funktion

sigma_norm=0.45; %Glaettungsfaktor
p=11;
eta=0.001; %Lernschrittweite mit 0.001 bestimmen

[Gewichte,ksi,delta_ksi,n]=RBF(u,Output_RBF,sigma_norm,p,eta);

AF_RBF=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2));

figure(1)
for i=1:p
    A=Gewichte(i)*AF_RBF(u,ksi(i));
    plot(u,A);
    hold on
end
title('Aktivierungsfunktionen mal Gewichte')
xlabel('Anregung u')
xlim([-8 8])
hold off

figure(2)
for i=1:p
    A=AF_RBF(u,ksi(i));
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
    prob_y(i)=sum(Gewichte.*AF_RBF(u2(i),ksi));
end
plot(u2,prob_y,'-');
title('Lernergebnis RBF-Netz')
xlabel('Anregung u')
legend('Gewichte','Vorgegebene Kennlinie','Identifizierte Kennlinie')
xlim([-8 8])
hold off

figure(4)
A=AF_RBF(u,0);
plot(u,A);
title('Gau?schen Glockenkurven')
xlabel('Anregung u')
ylim([-0.1 1.1])
xlim([-1.5 1.5])
hold off