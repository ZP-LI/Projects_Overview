function n_th_element = fibo(l)
a = [0 1];

if l == 0
    n_th_element = 0;
elseif l == 1
    n_th_element = 1;
elseif l > 1
    for i = 2:l
        a(i+1) = a(i) + a(i-1);
    end
    n_th_element = a(l+1);
else
    n_th_element = -1;
end

end