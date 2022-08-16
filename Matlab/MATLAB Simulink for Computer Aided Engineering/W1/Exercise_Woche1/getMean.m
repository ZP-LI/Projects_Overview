function meanAge = getMean(Struct_Attendance, Struct_Age)

Array_Attendance = cell2mat(Struct_Attendance);
Array_Age = cell2mat(Struct_Age);
meanAge = mean(Array_Age(Array_Attendance));

end