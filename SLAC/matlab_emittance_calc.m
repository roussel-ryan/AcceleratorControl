%  function [emittance_x, emittance_y] = matlab_emittance_calc()
function [emittance_x, emittance_y, emittance_x_std, emittance_y_std, bmag_x, bmag_y, bmag_x_std, bmag_y_std] = matlab_emittance_calc()

%epicsSimul_init %set simulation envirment

%data = appRemote(hObject, devName, measureType, plane)

pv='OTRS:IN20:571'; %for OTR2
measureType = 'scan';
[ho, h]=util_appFind('emittance_gui');


set(h.measureQuadValNum_txt,'string',4); %7 quad values


% This will uncheck AutoVal
set(h.measureQuadAutoVal_box,'Value',0); %uncheck the autoval ranges
                                        %for quads
h.measureQuadAutoVal=0;

set(h.dataSaveImg_box,'Value', 1); %this checks the save images

guidata(ho, h);

emittance_gui('measureQuadValNum_txt_Callback', h.measureQuadValNum_txt,[],h); %try this to save the values
set(h.measureQuadRangeLow_txt,'string',-5.4);
set(h.measureQuadRangeHigh_txt,'string',1);
%emittance_gui('measureQuadRange_txt_Callback', h.measureQuadRangeLow_txt,[],h) %try this to save the values
%emittance_gui('measureQuadRange_txt_Callback', h.measureQuadRangeHigh_txt,[],h) %try this to save the values


%%%%%%%%%%%%%%%%%%%%% RUN EMIT SCAN %%%%%%%%%%%%%%%%%%%%
data = emittance_gui('appRemote',ho,pv,measureType); % uncomment me
%,'OTRS:IN20:571','scan',1)

% print to elog and save data
%  emittance_gui('dataExport_btn_Callback',h.printLog_btn,[],h,1);

%There should be 7 values of emittance stored in the output structure, one for each fitting method available (RMS, Gaussian, etc…).
%The one selected at the time of measurement gets posted to EPICS. For wires the default is method 2. For screens it is method 6

fit_index = 6; % which fitting method to select

emittance_x=data.twissPV(1).val; %emittance x - 7 vlaues
emittance_y=data.twissPV(5).val; %emittance y
emittance_x=emittance_x(fit_index); %emittance x - 7 vlaues
emittance_y=emittance_y(fit_index); %emittance y
emittance_x_std=1e6*data.twissstd(1,1,fit_index); %uncert. in emittance x - 7 vlaues
emittance_y_std=1e6*data.twissstd(1,2,fit_index); %uncert. in emittance y

bmag_x=data.twiss(4,1,fit_index); %match param for x
bmag_y=data.twiss(4,2,fit_index); %match param for y
bmag_x_std=data.twissstd(4,1,fit_index); %uncert. in match param for x
bmag_y_std=data.twissstd(4,2,fit_index); %uncert. in match param for y

%emittance = (emittance_x*emittance_y)

%for wire scans
%[hoScan, hScan]=util_appFind('wirescan_gui')
%data = wirescan_gui('emittance_gui','WS31','x');


%  emittance_x = random('Normal',0,1);
%  emittance_y = random('Normal',0,1);
%  emittance_x_std = random('Normal',0,1);
%  emittance_y_std =  random('Normal',0,1);
%  bmag_x = random('Normal',0,1);
%  bmag_y =  random('Normal',0,1);
%  bmag_x_std = random('Normal',0,1);
%  bmag_y_std = random('Normal',0,1);
%  emittance =  random('Normal',0,1);

end