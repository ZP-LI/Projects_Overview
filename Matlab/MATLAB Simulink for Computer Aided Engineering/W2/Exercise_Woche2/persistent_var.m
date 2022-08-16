function new_count = persistent_var

persistent n
if isempty(n)
    n = 0;
end
n = n + 1;
new_count = n;

end