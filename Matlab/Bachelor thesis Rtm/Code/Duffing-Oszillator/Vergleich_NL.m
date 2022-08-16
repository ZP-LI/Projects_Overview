Zaehler_AF_GRNN=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2));
Nenner_AF_GRNN=@(u,ksi) sum(Zaehler_AF_GRNN(u,ksi));
AF_GRNN=@(u,ksi) Zaehler_AF_GRNN(u,ksi)/Nenner_AF_GRNN(u,ksi); %Aktivierungsfunktionen

u=linspace(-0.3,0.3,101);
n=length(u);
figure(1)
for i=1:n
    geschaetzt(i)=sum(theta(end,:).*AF_GRNN(u(i),ksi));
end
plot(u,geschaetzt,'-.')
hold on
plot(u,beta_w*u.^3,'-')
hold off
legend('identifiziert','vorgegeben')
grid on

figure(2)
for i=1:n
    Nenner(i)=sum(Zaehler_AF_GRNN(u(i),ksi));
end
for i=1:r
    plot(u,Zaehler_AF_GRNN(u,ksi(i))./Nenner);
    hold on
end
xlim([-0.3 0.3])
hold off