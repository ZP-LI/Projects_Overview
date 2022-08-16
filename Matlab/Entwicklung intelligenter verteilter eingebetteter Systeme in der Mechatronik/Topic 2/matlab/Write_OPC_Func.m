function Write_OPC_Func(input) %write sensor values from plant to OPC
% if initServer not empty use: clear functions
werkstueck_vorhanden=input(1);
werkstueck_nicht_schwarz=input(2);
arbeitsraum_frei=input(3);
hoehe_ok=input(4);
aufzug_oben=input(5);
aufzug_unten=input(6);
zylinder_eingefahren=input(7);
folgestation_frei=input(8);

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
persistent initServer;
persistent initNodes;
persistent uaClient;

persistent varNode_werkstueck_vorhanden;
persistent varNode_werkstueck_nicht_schwarz;
persistent varNode_arbeitsraum_frei;
persistent varNode_hoehe_ok;
persistent varNode_aufzug_oben;
persistent varNode_aufzug_unten;
persistent varNode_zylinder_eingefahren;
persistent varNode_folgestation_frei;



%init varas
if(isempty(initServer))
    
    initServer = 0;
    initNodes = 0;     
end
if initServer == 0
    initServer =1;
    uaClient = opcua('DESKTOP-4GAOAL2',4840);
    connect(uaClient);
end
if uaClient.isConnected ==1 && initNodes == 0
    initNodes = 1;
    % find with var = browseNamespace(uaClientRead)
    %varNodeIn = opcuanode(4,'|var|CODESYS Control Win V3 x64.Application.PLC_PRG.var1');
    %varNodeOut = opcuanode(4,'|var|CODESYS Control Win V3 x64.Application.PLC_PRG.var1');
    %varNodeIn = findNodeByName(uaClient.Namespace,'var1', '-once');
     varNode_werkstueck_vorhanden = findNodeByName(uaClient.Namespace,'werkstueck_vorhanden', '-once');
     varNode_werkstueck_nicht_schwarz = findNodeByName(uaClient.Namespace,'werkstueck_nicht_schwarz', '-once');
     varNode_arbeitsraum_frei = findNodeByName(uaClient.Namespace,'arbeitsraum_frei', '-once');
     varNode_hoehe_ok = findNodeByName(uaClient.Namespace,'hoehe_ok', '-once');
     varNode_aufzug_oben = findNodeByName(uaClient.Namespace,'aufzug_oben', '-once');
     varNode_aufzug_unten = findNodeByName(uaClient.Namespace,'aufzug_unten', '-once');
     varNode_zylinder_eingefahren = findNodeByName(uaClient.Namespace,'zylinder_eingefahren', '-once');
     varNode_folgestation_frei = findNodeByName(uaClient.Namespace,'folgestation_frei', '-once');
end

if uaClient.isConnected ==1 && initNodes ==1
    
    writeValue(uaClient, varNode_werkstueck_vorhanden, werkstueck_vorhanden);
    writeValue(uaClient, varNode_werkstueck_nicht_schwarz, werkstueck_nicht_schwarz);
    writeValue(uaClient, varNode_arbeitsraum_frei, arbeitsraum_frei);
    writeValue(uaClient, varNode_hoehe_ok, hoehe_ok);
    writeValue(uaClient, varNode_aufzug_oben, aufzug_oben);
    writeValue(uaClient, varNode_aufzug_unten, aufzug_unten);
    writeValue(uaClient, varNode_zylinder_eingefahren, zylinder_eingefahren);
    writeValue(uaClient, varNode_folgestation_frei, folgestation_frei);
    
end

end

