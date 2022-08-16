%Incidence matrix
A = [   -1  0   -1  0   0   0   0   0   0   0   ;
        1   -1  0   0   0   0   0   0   0   0   ;
        0   1   0   1   -1  -1  0   0   0   0   ;
        0   0   0   0   0   1   0   1   0   0   ;
        0   0   0   0   0   0   1   -1  0   0   ;
        0   0   1   -1  1   0   -1  0   1   -1  ;
        0   0   0   0   0   0   0   0   -1  1   ];

%Start vector
m_s = [ 1   0   0   0   0   0   0   ]';
%End vector
m_t = [ 0   0   0   1   0   0   0   ]';
%Cost vector
c_T = [   .5  .8  .6  .25 .25 .8  .6  .5  .25 .25];

OptimalSequence(A, m_s, m_t, c_T);