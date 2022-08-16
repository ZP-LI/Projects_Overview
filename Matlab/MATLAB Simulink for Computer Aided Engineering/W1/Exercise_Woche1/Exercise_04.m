clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
l = input('Enter a number to calculate the Fibonacci series: \n');
a = [0 1];

if l == 0
    display([0])
elseif l == 1
    display([1])
elseif l > 1
    for i = 2:l
        a(i+1) = a(i) + a(i-1);
    end
        display([a(l+1)])
else
    display('Incorrect Input')
end

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
clear; close all; clc;

tic
l = input('Enter a number to calculate the Fibonacci series: \n');
n_th_element = fibo(l);
display([n_th_element])
toc
% runtime: 4.388820second

%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
m = input('Enter the n_th element of the Fibonacci series: \n');
a = [0 1];

tic
if m < 2
    display('Incorrect Input!')
else
    for i = 2:m
        a(i+1) = fibonacci_recursive(a(i), a(i-1));
    end
    display([a(i-1), a(i)])
end
toc
% runtime: 0.002863second

%% ========================================================================
%  SubTask 4
%  ------------------------------------------------------------------------
r = input('Enter the radius of the sphere: \n');

[Volume, Surface, Circumference] = SphereData(r);

fprintf('Volume: %f \nSurface: %f \nCircumference: %f \n', Volume, Surface, Circumference)

%% ========================================================================
%  SubTask 5
%  ------------------------------------------------------------------------
r = input('Enter the radius of the sphere: \n');

[Volume, Surface, Circumference] = SphereData_LocalFunc(r);

fprintf('Volume: %f \nSurface: %f \nCircumference: %f \n', Volume, Surface, Circumference)

% Change: nested functions are located in parent function
%               nested functions can use variables defined in parent functions without explicitly passing those variables as arguments
%               local functions are parallel with parent function