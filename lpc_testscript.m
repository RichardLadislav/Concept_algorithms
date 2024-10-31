clc
close all

% [x,Fs] = audioread("activity_unproductive.wav"); 
[x,Fs] = audioread("saw-wave-2-g3.wav"); 
[acoef, Res] = lpc_demo(x,Fs, 200, 5, 12);
% [A,G] = get_lpc(x,Fs);
 % F = formants(acoef,Fs);
 display(acoef);
% [p,t,s] = swipep(x,Fs,[75 500],0.01,[],1/20,0.4); 
% plot(1000*t,p) 
% xlabel('Time (ms)') 
% ylabel('Pitch (Hz)') 