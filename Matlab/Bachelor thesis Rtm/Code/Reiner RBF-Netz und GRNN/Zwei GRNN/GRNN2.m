function [theta1,theta2,ksi1,ksi2,delta_ksi]=GRNN2 (u,n,Output_GRNN2,sigma_norm,p2,eta)

delta_ksi=(5-0)/(p2-1);
ksi1=linspace(-5,0,6);
ksi2=linspace(0,5,6);

Zaehler_AF_GRNN=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2));
Nenner_AF_GRNN=@(u,ksi) sum(Zaehler_AF_GRNN(u,ksi));
AF_GRNN=@(u,ksi) Zaehler_AF_GRNN(u,ksi)/Nenner_AF_GRNN(u,ksi); %Aktivierungsfunktionen

theta1=zeros(1,p2); %Anfangswert von Theta des negativen GRNNs
theta2=zeros(1,p2); %Anfangswert von Theta des positiven GRNNs
Schaetzwert_Output_GRNN2=zeros(1,n); %Definieren von geschaetztem Wert von j-te Output mit Anfangswert Null

neu_theta1=theta1;
neu_theta2=theta2;
alt_theta1=ones(1,p2);
alt_theta2=ones(1,p2);

while sum(neu_theta1)~=sum(alt_theta1) || sum(neu_theta2)~=sum(alt_theta2) %keine empfehlende Iteration, deshalb while nutzen
    alt_theta1=neu_theta1;
    alt_theta2=neu_theta2;
    
    for i=1:n
        if Output_GRNN2(i)>=0
            Schaetzwert_Output_GRNN2(i)=sum(theta2.*AF_GRNN(u(i),ksi2));
            e(i)=Schaetzwert_Output_GRNN2(i)-Output_GRNN2(i);
            theta2=theta2-eta*e(i)*AF_GRNN(u(i),ksi2);
        end
        if Output_GRNN2(i)<=0
            Schaetzwert_Output_GRNN2(i)=sum(theta1.*AF_GRNN(u(i),ksi1));
            e(i)=Schaetzwert_Output_GRNN2(i)-Output_GRNN2(i);
            theta1=theta1-eta*e(i)*AF_GRNN(u(i),ksi1);
        end
    end %Erneuern von Theta anhand i-te Lernfehler
    
    neu_theta1=theta1;
    neu_theta2=theta2;
    Schaetzwert_Output_GRNN2=zeros(1,n); %R¨¹ckstellung von Schaetzwert von Output
end 
end