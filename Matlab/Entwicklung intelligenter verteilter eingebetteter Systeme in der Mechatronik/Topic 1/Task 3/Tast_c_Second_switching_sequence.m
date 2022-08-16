% Day1 Task3 c) First switching sequence

% initial marker vector
m0=[1;0;0;0;0];

% incidence matrix
C=[-1 -1 0 0 0 0 0 1 1; 1 0 -1 -1 0 0 0 0 0; 0 1 1 0 -1 -1 0 0 0; 0 0 0 1 1 0 -1 -1 0; 0 0 0 0 0 1 1 0 -1];

% parikh vector
v{1}=[1 0 0 0 0 0 0 0 0];
v{2}=[0 0 1 0 0 0 0 0 0];
v{3}=[0 0 0 0 1 0 0 0 0];
v{4}=[0 0 0 0 0 1 0 0 0];
v{5}=[0 0 0 0 0 0 1 0 0];
v{6}=[0 0 0 0 0 0 0 0 1];
v{7}=[0 0 0 0 0 0 0 0 1];
v{8}=[0 1 0 0 0 0 0 0 0];

% real in-between mark vector
m{1}=[0;1;0;0;0];
m{2}=[0;0;1;0;0];
m{3}=[0;0;0;1;0];
m{4}=[0;0;0;1;0];
m{5}=[0;0;0;0;1];
m{6}=[1;0;0;0;0];
m{7}=[1;0;0;0;0];
m{8}=[0;0;1;0;0];

% calculation of brand occupancy base on parikh vector
for i=1:8
    m_cal{i}=m0+C*transpose(v{i});
    m0=m_cal{i};
end

% compare calculated mark vectors with real mark vectors and print the output
for i=1:8
    fprintf('%i-th calculated mark vector is [%i %i %i %i %i]\n', i, transpose(m_cal{i}));
    fprintf('%i-th real mark vector is [%i %i %i %i %i]\n', i, transpose(m{i}));
    
    if m_cal{i}==m{i}
        disp('true')
    else
        disp('false')
    end
    
    disp(' ')
end

% Result: By running this programm a false value appears since the 4-th transition (...t6...). 
%            The value of calculated mark of s3 is at this moment -1, what makes no sence.
%            The reason for this error is that there is only one s3 mark. It can't activate t5 and t6 repeatedly.