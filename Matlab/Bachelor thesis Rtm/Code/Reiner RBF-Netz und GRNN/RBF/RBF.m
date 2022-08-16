function [theta,ksi,delta_ksi,n]=RBF(u,Output_RBF,sigma_norm,p,eta)

delta_ksi=(5-(-5))/(p-1);
ksi=linspace(-5,5,11);
AF_RBF=@(u,ksi) exp(-(u-ksi).^2/(2*sigma_norm^2*delta_ksi^2)); %Aktivierungsfunktionen

theta=zeros(1,p); %Anfangswert von Theta
n=length(u); %Anzahl von Eingangszahl
Schaetzwert_Output_RBF=zeros(1,n); %Definieren von geschaetztem Wert von j-te Output mit Anfangswert Null

neu_theta=theta;
alt_theta=ones(1,p);

while sum(neu_theta)~=sum(alt_theta) %keine empfehlende Iteration, deshalb "while" nutzen
    alt_theta=neu_theta;
    
    for i=1:n
        Schaetzwert_Output_RBF(i)=sum(theta.*AF_RBF(u(i),ksi));
        e(i)=Schaetzwert_Output_RBF(i)-Output_RBF(i);
        theta=theta-eta*e(i)*AF_RBF(u(i),ksi);
    end %Erneuern von Theta anhand i-te Lernfehler
    
    neu_theta=theta;
    Schaetzwert_Output_RBF=zeros(1,n); %R¨¹ckstellung von Schaetzwert von Output
end 