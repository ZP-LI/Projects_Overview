function output = Read_OPC_Func(~) %read actuator values from PLC_Control 
%[a_aufzug_abwaerts, a_aufzug_aufwaerts, a_zylinder_ausschieben, a_luft_an, out4, out5, out6, a_station_frei]
% if initServer not empty use: clear functions

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
persistent initReadServer;
persistent initReadNodes;
persistent uaClientRead;

%Nodevariables opc ua Server

persistent varNode_a_aufzug_abwaerts;
persistent varNode_a_aufzug_aufwaerts;
persistent varNode_a_zylinder_ausschieben;
persistent varNode_a_luft_an;
persistent varNode_a_station_frei;

%temp var to write to simulink

persistent tmp_a_aufzug_abwaerts;
persistent tmp_a_aufzug_aufwaerts;
persistent tmp_a_zylinder_ausschieben;
persistent tmp_a_luft_an;
persistent tmp_a_station_frei;


%init varas
if(isempty(initReadServer))
    
    tmp_a_aufzug_abwaerts = 0;
    tmp_a_aufzug_aufwaerts = 0;
    tmp_a_zylinder_ausschieben = 0;
    tmp_a_luft_an = 0;
    tmp_a_station_frei = 0;
    
    initReadServer = 0;
    initReadNodes = 0;     
end
if initReadServer == 0
    initReadServer =1;
    uaClientRead = opcua('DESKTOP-4GAOAL2',4840);
    connect(uaClientRead);
end
if uaClientRead.isConnected ==1 && initReadNodes == 0
    initReadNodes = 1;
    
    %varNode = opcuanode(4,'|var|CODESYS Control Win V3 x64.Application.PLC_PRG.strg.e_werkstueck_nicht_schwarz');
         
     varNode_a_aufzug_abwaerts = findNodeByName(uaClientRead.Namespace,'aufzug_abwaerts', '-once');
     varNode_a_aufzug_aufwaerts = findNodeByName(uaClientRead.Namespace,'aufzug_aufwaerts', '-once');
     varNode_a_zylinder_ausschieben = findNodeByName(uaClientRead.Namespace,'zylinder_ausschieben', '-once');
     varNode_a_luft_an = findNodeByName(uaClientRead.Namespace,'luft_an', '-once');
     varNode_a_station_frei = findNodeByName(uaClientRead.Namespace,'station_frei', '-once');
     
end

if uaClientRead.isConnected ==1 && initReadNodes ==1
    [tmp1,~,~] = readValue(uaClientRead,varNode_a_aufzug_abwaerts);    
    tmp_a_aufzug_abwaerts = tmp1;
    [tmp2,~,~] = readValue(uaClientRead,varNode_a_aufzug_aufwaerts);    
    tmp_a_aufzug_aufwaerts = tmp2;
    [tmp3,~,~] = readValue(uaClientRead,varNode_a_zylinder_ausschieben);    
    tmp_a_zylinder_ausschieben = tmp3;
    [tmp4,~,~] = readValue(uaClientRead,varNode_a_luft_an);    
    tmp_a_luft_an = tmp4;
    [tmp5,~,~] = readValue(uaClientRead,varNode_a_station_frei);    
    tmp_a_station_frei = tmp5;
end
a_aufzug_abwaerts=double(tmp_a_aufzug_abwaerts);
a_aufzug_aufwaerts=double(tmp_a_aufzug_aufwaerts);
a_zylinder_ausschieben=double(tmp_a_zylinder_ausschieben);
a_luft_an=double(tmp_a_luft_an);
out4=double(0);
out5=double(0);
out6=double(0);
a_station_frei=double(tmp_a_station_frei);

output = [a_aufzug_abwaerts, a_aufzug_aufwaerts, a_zylinder_ausschieben, a_luft_an,out4,out5, out6, a_station_frei];

end

