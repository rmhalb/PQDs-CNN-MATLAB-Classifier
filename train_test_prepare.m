function [XTrain, YTrain, XTest, YTest] = train_test_prepare(Signals, Labels)

    XTrain = [];
    YTrain = [];
    XTest = [];
    YTest = [];
    
    pqd_dictionary = { 'Flicker', 'Flicker+Harmonics', 'Flicker+Sag', ...
                       'Flicker+Swell', 'Harmonics', 'Impulsive Transient', ...
                       'Interruption', 'Interruption+Harmonics', 'Normal','Notch',...
                       'Oscillatory transient', 'Sag', 'Sag+Harmonics', 'Spike',...
                       'Swell' , 'Swell+Harmonics' };

    % Split the signals according to their class.
    % Assumption :By default, the neural network randomly shuffles the data before training, 
    % ensuring that contiguous signals do not all have the same label.
    for d =1 :length(pqd_dictionary)
        signals_d_x = Signals(Labels==pqd_dictionary{d});
        labales_d_y = Labels(Labels==pqd_dictionary{d});
        
        % use dividerand to divide targets from each class randomly into training and testing sets
        [trainInd_d,~,testInd_d] = dividerand(length(signals_d_x),0.9,0.0,0.1);
 
        XTrain = [XTrain; signals_d_x(trainInd_d)];
        YTrain = [YTrain; labales_d_y(trainInd_d)];
        XTest =  [XTest; signals_d_x(testInd_d)];
        YTest =  [YTest; labales_d_y(testInd_d)];
    end
end