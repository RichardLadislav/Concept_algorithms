function [A, G] = get_lpc (signal, srate, M, window, slide)
 %
 % get_lpc
 % Louis Goldstein
 % March 2005
 % 
% Output arguments:
 % A         filter coefficients (M+1 rows x  nframes columns)
 % G         Gain coefficients (vecor length = nframes)
 %
 % input arguements:
 % signal    signal to be analyzed
 % srate     sampling rate in Hz
 % M         LPC order  (def. = srate/1000 +4)
 % window size of analysis window in ms (def. = 10)
 % slide     no. of ms. to slide analysis window for each frame (def. = 5)
 if nargin < 3, M = floor(srate/1000) + 4; end
 if nargin < 4, window = 20; end
 if nargin < 5, slide = 10; end
 samp_tot = length(signal);
 samp_win = fix((window/1000)*srate);
 samp_slide = fix((slide/1000)*srate);
 nframes = floor(samp_tot/samp_slide) - ceil((samp_win-samp_slide)/samp_slide);
 A = [];
 G = [];
 for i = 1:nframes
    begin = 1 + (i-1)*samp_slide;
    [Ai,Gi] = lpc_demo(hamming(samp_win).*signal(begin:begin+samp_win-1),srate,window,M,96);
    A = [A Ai'];
    G = [G Gi];
 end