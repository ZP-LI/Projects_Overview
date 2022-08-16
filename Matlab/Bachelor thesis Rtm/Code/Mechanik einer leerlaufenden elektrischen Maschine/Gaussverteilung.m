u=linspace(-10,10,1001);
delta_ksi=(5-(-5))/(9-1);
y1=exp(-(u-0).^2./(2*1.6^2*delta_ksi^2));
y2=exp(-(u-1).^2./(2*1.6^2*delta_ksi^2));

y4=exp(-(u-0).^2./(2*0.45^2*delta_ksi^2));
y5=exp(-(u-1).^2./(2*0.45^2*delta_ksi^2));

plot(u,y1,'b')
hold on
plot(u,y2,'b')
hold on
plot(u,y4,'k')
hold on
plot(u,y5,'k')
hold off