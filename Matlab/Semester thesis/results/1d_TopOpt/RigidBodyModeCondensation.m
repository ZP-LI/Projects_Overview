close all
clc
clear all 


%% Stiffness Dependencies

syms lx dy dx
assume(dx ~= 0)
assume(dy ~= 0)
assume(lx ~= 0)

p1 = [dx;0;0;dx;0;0];
n = sqrt(p1(1)^2 + p1(2)^2 + p1(3)^2 + p1(4)^2+p1(5)^2+p1(6)^2); 
%n=simplify(n);
p1 = p1/n; 
p1 = simplify(p1)

p2 = [0;dy;0;0;dy;0];
n = sqrt(p2(1)^2 + p2(2)^2 + p2(3)^2 + p2(4)^2+ p2(5)^2 + p2(6)^2); 
%n=simplify(n);
p2 = p2/n; 
p2 = simplify(p2)

angle = 2*dy/lx;
p3 = [0;-dy;angle;0;dy;angle];  
n = sqrt(p3(1)^2 + p3(2)^2 + p3(3)^2 + p3(4)^2 + p3(5)^2 + p3(6)^2); 
%n=simplify(n);
p3 = p3/n; 
p3 = simplify(p3)


syms K_cs [6,6]
% Due to symmetry
K_cs = tril(K_cs,0) + tril(K_cs,-1).'

%First null vector
Eq1 = simplify(K_cs*p1 == 0);
Eq1

%Solvey by Hand, Hardcoded results
K_cs = subs(K_cs,K_cs4_1,-K_cs1_1);
K_cs = subs(K_cs,K_cs4_4, K_cs1_1);
K_cs = subs(K_cs,K_cs4_2,-K_cs2_1);
K_cs = subs(K_cs,K_cs4_3,-K_cs3_1);
K_cs = subs(K_cs,K_cs5_4,-K_cs5_1);
K_cs = subs(K_cs,K_cs6_4,-K_cs6_1);
K_cs

%Second null vector
Eq2 = K_cs*p2 ==0;
simplify(Eq2)

%Solvey by Hand, Hardcoded results
K_cs = subs(K_cs,K_cs5_1,-K_cs2_1);
K_cs = subs(K_cs,K_cs5_2,-K_cs2_2);
K_cs = subs(K_cs,K_cs5_5, K_cs2_2);
K_cs = subs(K_cs,K_cs5_3,-K_cs3_2);
K_cs = subs(K_cs,K_cs6_5,-K_cs6_2);

K_cs


%Third null vector
Eq3 = K_cs*p3 ==0;
simplify(Eq3)
Sol3 = solve(Eq3,[K_cs3_2,K_cs6_1,K_cs6_2,K_cs6_3],'Real',true,'ReturnConditions',true);

%Possible solution for the system of equation with respect to K11,K21,K22,K31,K33,K66
simplify(Sol3.K_cs3_2);
simplify(Sol3.K_cs6_1);
simplify(Sol3.K_cs6_2);
simplify(Sol3.K_cs6_3);
K_cs = subs(K_cs,K_cs3_2, Sol3.K_cs3_2);
K_cs = subs(K_cs,K_cs6_1, Sol3.K_cs6_1);
K_cs = subs(K_cs,K_cs6_2, Sol3.K_cs6_2);
K_cs = subs(K_cs,K_cs6_3, Sol3.K_cs6_3);
simplify(K_cs)


%Beam Element
nel = 10;
x = rand(10,1);
lt = 1;
E = 1; 
l = lt/nel;

KEr = E*l*...
        [ 1 -1 
         -1  1];
k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
KEb = E/l^3*...
    [ k(1) k(2) k(3) k(2)
    k(2) k(4) k(5) k(6)
    k(3) k(5) k(1) k(5)
    k(2) k(6) k(5) k(4)];


K = zeros((nel+1)*3, (nel+1)*3);
for el = 1:nel
    n1 =  3*(el-1)+1; 
    n2 =  3*el+1;
    edofr = [n1;n2];
	edofb = [n1+1;n1+2; n2+1;n2+2];
    K(edofr,edofr) = K(edofr,edofr) + pi*x(el)^2*KEr;
	K(edofb,edofb) = K(edofb,edofb) + pi/4*x(el)^4*KEb;
end

iodofs = [1,2,3,(nel+1)*3-2,(nel+1)*3-1,(nel+1)*3];
alldofs     = [1:3*(nel+1)];
ddofs = setdiff(alldofs,iodofs);
Kss = K(ddofs,ddofs);
Ksm = K(ddofs,iodofs);
Kmm = K(iodofs,iodofs);
Kms = K(iodofs,ddofs);
InvKss = Kss\eye(size(Kss));
Kred = (Kmm-Kms*InvKss*Ksm)

%K11,K21,K22,K31,K33,K66
K_cs = subs(K_cs,K_cs1_1, Kred(1,1));
K_cs = subs(K_cs,K_cs2_1, Kred(2,1));
K_cs = subs(K_cs,K_cs2_2, Kred(2,2));
K_cs = subs(K_cs,K_cs3_1, Kred(3,1));
K_cs = subs(K_cs,K_cs3_3, Kred(3,3));
K_cs = subs(K_cs,K_cs6_6, Kred(6,6));
K_cs = subs(K_cs,lx,lt);
Kred - double(K_cs) 
