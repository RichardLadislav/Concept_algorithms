function [acoef, Res] = lpc_demo (signal, srate, winsize, M, ibeg)
 % LPC Analysis
 % Louis Goldstein
 % 29 October 1992
 %
 % input arguments:
 % signal 
 % signal to be analyzed
 % srate 
 % winsize 
 % M  
 % ibeg 
 %
 % sampling rate in Hz
 % window size in no.of samples
 % LPC filter order
 % first sample in signal to be analyzed
 % output arguments:
 % acoef 
 % LPC coefficients
 % Res  
 % Residual from LPC analysis
 % iend is last sample in window
 iend = ibeg+winsize-1;
 % Y is a vector of winsize data points from signal
 % Y is the dependent variable for the regression
 Y = signal(ibeg:iend)';
 % matrix X contains the independent variables for the regression.
 % It contains M columns, each of which contains
 % elements of signal (y) delayed progressively by one sample
 for i=1:M
    X(:,i) = signal(ibeg-i:iend-i)';
 end
 % perform the regression
 % the coefficients are the weights that are applied to each
 % of the M columns in order to predict Y.
 % Since the columns are delayed versions of Y, 
 % the weights represent the best prediction of Y from the
 % previous M sample points.
 [acoef, bint, Res] = regress(Y,X);
 % The predicted signal is given by subtracting the residual
 % from Y
 Pred = Y - Res;
 % Plot original window and predicted
 subplot (211), plot (signal(ibeg:iend))
 title ('Original Signal')
 subplot (212), plot (Pred)
 title ('Predicted Signal')
 subplot
 % pause
 hold on
 % Plot original window and residual
 subplot (211), plot (signal(ibeg:iend))
 title ('Original Signal')
 subplot (212), plot (Res)
 title ('Residual')
 subplot
 hold on
 % pause
 % Get rid of the M+1st term returned by the regression.
 % It just represents the mean (DC level of the signal).
 %acoef(M+1) = [ ];
 acoef(M) = [ ];
 % The returned coefficients (acoef) can be used to predict
 % Y(k) from preceding M samples:
 % E1: Y(k) = a(1)*Y(k-1) + a(2)*Y(k-2) + ... + a(M)Y(k-M) + Res
 % We want what is called the INVERSE filter, which will 
% take Y as input and output the Res signal.
 % That is, it will take all the structure out of the signal
 % and give us an inpulse sequence. This can be obtained from
 % equation E1 by moving Y(k) and Res to the other side of E1:
 % E2: Res = 1*Y(k) - a(1)*Y(k-1) - a(2)*Y(k-2) - ... - a(M)*Y(k-M)
 % Thus, the first coefficient of the inverse filter is 1;
 % The rest are the negatives of the coefficients returned
 % by the regression.
 % Note -acoef' is used to convert from a column to row vector:
 acoef = [1 -acoef'];
 % Plot magnitude of transfer function 
 % for the INVERSE FILTER
 % freqz will return value of the transfer function, h,
 % for a given filter numerator and denominator
 % at npoints along the frequency scale.
 % The frequency of each point (in rad/sample) is returned in w.
 num = acoef;
 den = 1;
 npoints = 100;
 [h, w] = freqz(num,den,npoints);
 plot (w*srate./(2*pi), log(abs(h)))
 xlabel ('Frequency in Hz')
 ylabel (' Magnitude of Transfer Function')
 % title (['INVERSE FILTER: M = ', num2str(M), '   winsize = ', 
 % num2str(winsize),'   Beginning sample = ', num2str(ibeg)])
 % pause
 hold on
 % Plot magnitude of transfer function if the coefficients are
 % used as coefficients of the FORWARD filter (in denominator).
 % freqz will return value of the transfer function, h,
 % for a given filter numerator and denominator
 % at npoint along the frequency scale.
 % The frequency of each point (in rad/sample) is returned in w.
 num = 1;
 den = acoef;
 npoints = 100;
 [h, w] = freqz(num,acoef,npoints);
 plot (w*srate./(2*pi), log(abs(h)))
 xlabel ('Frequency in Hz')
 ylabel ('Log Magnitude of Transfer Function')
 % title (['FORWARD FILTER: M = ', num2str(M), '   winsize = ', 
 % num2str(winsize),'   Beginning sample = ', num2str(ibeg)])
