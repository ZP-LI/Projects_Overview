clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
Name = input('Calculation mode: \n','s');
fprintf('\n')

if strcmp(Name,'sum') == 1
    display('Calculate the sum of all integers from 1 to n.')
elseif strcmp(Name,'factorial') == 1
    display('Calculate the factorial of n.')
elseif strcmp(Name,'fibonacci') == 1
    display('Calculate the n_th element of the fibonacci series.')
else
    display('Incorrect Input!')
end

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
Name = input('Calculation mode: \n','s');

switch Name
    case 'sum'
        display('Calculate the sum of all integers from 1 to n.')
    case 'factorial'
        display('Calculate the factorial of n.')
    case 'fibonacci'
        display('Calculate the n_th element of the fibonacci series.')
    otherwise
        display('Incorrect Input!')
end

%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
n = input('Enter a number to calculate the sum of all integers from 1: \n');
sum_int = 0;

for i = 1:n
    sum_int = sum_int + i;
end

display([sum_int])

%% ========================================================================
%  SubTask 4
%  ------------------------------------------------------------------------
m = input('Enter a number to calculate the factorial: \n');
product_int = 1;
j = 1;
while j <= m
    product_int = product_int * j;
    j = j +1;
end

display([product_int])

%% ========================================================================
%  SubTask 5
%  ------------------------------------------------------------------------
l = input('Enter a number to calculate the Fibonacci series: \n');
a = [0 1];
t = 0;

if l == 0
    display([0])
elseif l == 1
    display([1])
elseif l > 1
    tic
    for i = 2:l
        a(i+1) = a(i) + a(i-1);
        t = toc;
        if t > 0.001
            break
        end
    end
    if i == l
        display([a(l+1)])
    else
        display('Too large calculation time!')
    end
else
    display('Incorrect Input')
end

