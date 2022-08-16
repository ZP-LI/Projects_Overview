function [ x ] = OptimalSequence( A, m_s, m_t, c_T )

%Lower bound
lb = zeros(length(A),1);

%Upper bound
ub = ones(length(A),1);

%Optimization problem
x = linprog(c_T, A, m_t-m_s, [], [], lb, ub, []);
disp('Optimale Schaltfolge lautet:')
disp(x)

end

