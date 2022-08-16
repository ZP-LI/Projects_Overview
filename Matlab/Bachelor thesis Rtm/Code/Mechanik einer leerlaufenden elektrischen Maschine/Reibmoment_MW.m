Zaehler_AF_GRNN=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2));
Nenner_AF_GRNN=@(u,ksi) sum(Zaehler_AF_GRNN(u,ksi));
AF_GRNN=@(u,ksi) Zaehler_AF_GRNN(u,ksi)/Nenner_AF_GRNN(u,ksi); %Aktivierungsfunktionen

for i=1:101
    a=-20+i*(20-(-20))/101;
    if a>=0
        wahr_MW(i)=0.29*atan(0.3*a)+0.4;
    else
        wahr_MW(i)=0.35*atan(0.3*a)-0.15;
    end
    a=a+1;
end

u=linspace(-20,20,101);
n=length(u);
for i=1:n
    geschaetzt_MW(i)=sum(theta_Reib(end,:).*AF_GRNN(u(i),ksi));
end
plot(u,geschaetzt_MW,'-.')
hold on
plot(u,wahr_MW,'-')
hold off
legend('identifiziert','vorgegeben')
xlabel('Winkeigeschwindigkeit omege1[rad/s]')
ylabel('Reibmoment Mw[Nm]')
grid on
