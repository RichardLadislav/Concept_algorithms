function [p,t,s] = swipep(x,fs,plim,dt,dlog2p,dERBs,sTHR) 
% SWIPEP Pitch estimation using SWIPE'. 
% P = SWIPEP(X,Fs,[PMIN PMAX],DT,DLOG2P,DERBS,STHR) estimates the pitch of  
% the vector signal X with sampling frequency Fs (in Hertz) every DT 
% seconds. The pitch is estimated by sampling the spectrum in the ERB scale 
% using a step of size DERBS ERBs. The pitch is searched within the range  
% [PMIN PMAX] (in Hertz) sampled every DLOG2P units in a base-2 logarithmic 
% scale of Hertz. The pitch is fine tuned by using parabolic interpolation  
% with a resolution of 1/64 of semitone (approx. 1.6 cents). Pitches with a 
% strength lower than STHR are treated as undefined. 
%     
% [P,T,S] = SWIPEP(X,Fs,[PMIN PMAX],DT,DLOG2P,DERBS,STHR) returns the times 
% T at which the pitch was estimated and their corresponding pitch strength. 
% 
% P = SWIPEP(X,Fs) estimates the pitch using the default settings PMIN = 
% 30 Hz, PMAX = 5000 Hz, DT = 0.01 s, DLOG2P = 1/96 (96 steps per octave), 
% DERBS = 0.1 ERBs, and STHR = -Inf. 
% 
% P = SWIPEP(X,Fs,...[],...) uses the default setting for the parameter 
% replaced with the placeholder []. 
% 
%    EXAMPLE: Estimate the pitch of the signal X every 10 ms within the 
%    range 75-500 Hz using the default resolution (i.e., 96 steps per 
%    octave), sampling the spectrum every 1/20th of ERB, and discarding 
%    samples with pitch strength lower than 0.4. Plot the pitch trace. 
%    [x,Fs] = audioread("activity_unproductive.wav"); 
%    [p,t,s] = swipep(x,Fs,[75 500],0.01,[],1/20,0.4); 
%    plot(1000*t,p) 
%    xlabel('Time (ms)') 
%    ylabel('Pitch (Hz)') 
if ~ exist( 'plim', 'var' ) || isempty(plim), plim = [30 5000]; end 
if ~ exist( 'dt', 'var' ) || isempty(dt), dt = 0.01; end 
if ~ exist( 'dlog2p', 'var' ) || isempty(dlog2p), dlog2p = 1/96; end 
if ~ exist( 'dERBs', 'var' ) || isempty(dERBs), dERBs = 0.1; end 
if ~ exist( 'sTHR', 'var' ) || isempty(sTHR), sTHR = -Inf; end 
t = [ 0: dt: length(x)/fs ]'; % Times 
dc = 4; % Hop size (in cycles) 
K = 2; % Parameter k for Hann window 
% Define pitch candidates 
log2pc = [ log2(plim(1)): dlog2p: log2(plim(end)) ]'; 
pc = 2 .^ log2pc; 
S = zeros( length(pc), length(t) ); % Pitch strength matrix 
% Determine P2-WSs 
logWs = round( log2( 4*K * fs ./ plim ) );  
ws = 2.^[ logWs(1): -1: logWs(2) ]; % P2-WSs 
pO = 4*K * fs ./ ws; % Optimal pitches for P2-WSs 
% Determine window sizes used by each pitch candidate 
d = 1 + log2pc - log2( 4*K*fs./ws(1) ); 
%% potato je to jakz takz prepisane 
% Create ERBs spaced frequencies (in Hertz)
ferbs_help = [ hz2erbs(pc(1)/4): dERBs: hz2erbs(fs/2) ]';
erbs1 = hz2erbs(pc(1)/4)
erbs2 = hz2erbs(fs/2)
dERBs
fERBs = erbs2hz([ hz2erbs(pc(1)/4): dERBs: hz2erbs(fs/2) ]'); 
for i = 1 : length(ws) 
    dn = round( dc * fs / pO(i) ); % Hop size (in samples) 
    % Zero pad signal 
    xzp = [ zeros( ws(i)/2, 1 ); x(:); zeros( dn + ws(i)/2, 1 ) ]; 
    % Compute spectrum 
    w = hanning( ws(i) ); % Hann window  
    o = max( 0, round( ws(i) - dn ) ); % Window overlap 
    [ X, f, ti ] = specgram( xzp, ws(i), fs, w, o ); 
    % Interpolate at equidistant ERBs steps 

    M = max( 0, interp1( f, abs(X), fERBs, 'spline', 0) ); % Magnitude 
    L = sqrt( M ); % Loudness 
    % Select candidates that use this window size 
    if i==length(ws); j=find(d-i>-1); k=find(d(j)-i<0); 
    elseif i==1; j=find(d-i<1); k=find(d(j)-i>0); 
    else j=find(abs(d-i)<1); k=1:length(j); 
    end     
    Si = pitchStrengthAllCandidates( fERBs, L, pc(j) ); 
    % Interpolate at desired times 
    if size(Si,2) > 1 
        Si = interp1( ti, Si', t, 'linear', NaN )'; 
    else 
        Si = repmat( NaN, length(Si), length(t) ); 
    end 
    lambda = d( j(k) ) - i; 
    mu = ones( size(j) ); 
    mu(k) = 1 - abs( lambda ); 
    S(j,:) = S(j,:) + repmat(mu,1,size(Si,2)) .* Si; 
end 
% Fine-tune the pitch using parabolic interpolation 
p = repmat( NaN, size(S,2), 1 ); 
s = repmat( NaN, size(S,2), 1 ); 
for j = 1 : size(S,2) 
    [ s(j), i ] = max( S(:,j) ); 
    if s(j) < sTHR, continue, end 
    if i==1, p(j)=pc(1); elseif i==length(pc), p(j)=pc(1); else 
        I = i-1 : i+1; 
        tc = 1 ./ pc(I); 
        ntc = ( tc/tc(2) - 1 ) * 2*pi;
        c = polyfit( ntc, S(I,j), 2 ); 
        ftc = 1 ./ 2.^ [ log2(pc(I(1))): 1/12/64: log2(pc(I(3))) ]; 
        nftc = ( ftc/tc(2) - 1 ) * 2*pi; 
        [s(j), k] = max( polyval( c, nftc ) ); 
        p(j) = 2 ^ ( log2(pc(I(1))) + (k-1)/12/64 ); 
    end 
end 

%% funkcie
function S = pitchStrengthAllCandidates( f, L, pc ) 
% Normalize loudness 
warning off MATLAB:divideByZero 
L = L ./ repmat( sqrt( sum(L.*L) ), size(L,1), 1 ); 
warning on MATLAB:divideByZero 
% Create pitch salience matrix 
S = zeros( length(pc), size(L,2) );  
for j = 1 : length(pc) 
    S(j,:) = pitchStrengthOneCandidate( f, L, pc(j) ); 
 
end 
%%
function S = pitchStrengthOneCandidate( f, L, pc ) 
n = fix( f(end)/pc - 0.75 ); % Number of harmonics 
k = zeros( size(f) ); % Kernel 
q = f / pc; % Normalize frequency w.r.t. candidate 
for i = [ 1 primes(n) ] 
    a = abs( q - i ); 
    % Peak's weigth 
    p = a < .25;  
    k(p) = cos( 2*pi * q(p) ); 
    % Valleys' weights 
    v = .25 < a & a < .75; 
    k(v) = k(v) + cos( 2*pi * q(v) ) / 2; 
end 
% Apply envelope 
k = k .* sqrt( 1./f  );  
% K+-normalize kernel 
k = k / norm( k(k>0) );  
% Compute pitch strength 
S = k' * L;  
function erbs = hz2erbs(hz) 
erbs = 21.4 * log10( 1 + hz/229 ); 
function hz = erbs2hz(erbs) 
hz = ( 10 .^ (erbs./21.4) - 1 ) * 229;