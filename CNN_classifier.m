clc
clear all

%---- Load Database ---- %
%load('16PQDs_4800_NoNoise.mat')
load('16PQDs_4800_WithNoise.mat')

SignalsDataBaseCell = struct2cell(SignalsDataBase);

% signal %
Signals = SignalsDataBaseCell(2,:,:);
Signals = reshape(Signals, [1,length(Signals)])'; 

% labels %
Labels = SignalsDataBaseCell(1,:,:);
Labels = reshape(Labels, [1,length(Labels)]); 
Labels = categorical(Labels)';

% ---- Raw Signal Data for train and test --- %
[XTrain, YTrain, XTest, YTest] = train_test_prepare(Signals,Labels);
imageSize = [1 640 1];

Train_data = zeros(1,length(XTrain{1}),length(XTrain));
for i= 1:length(XTrain)
    Train_data(:,:,i) =  XTrain{i};
end
Train_data = reshape(Train_data,[1,length(XTrain{1}),1, length(XTrain)]); % Dataset is ready %
imdsTrain = augmentedImageDatastore(imageSize,Train_data,YTrain);

Test_data = zeros(1,length(XTest{1}),length(XTest));
for i= 1:length(XTest)
    Test_data(:,:,i) =  XTest{i};
end
Test_data = reshape(Test_data,[1,length(XTest{1}),1, length(XTest)]); % Dataset is ready %
imdsTest = augmentedImageDatastore(imageSize,Test_data,YTest);

% ---- Define the CNN Network Architecture --- %
c1=convolution2dLayer([1 3],32,'stride',1);
c2=convolution2dLayer([1 3],64,'stride',1);
c3=convolution2dLayer([1 3],128,'stride',1);
maxPool = maxPooling2dLayer([1 3],'stride',1);
f1 = fullyConnectedLayer(256);
f2 = fullyConnectedLayer(128);
f3 = fullyConnectedLayer(16);

layers = [ imageInputLayer(imageSize) ;
           c1; reluLayer;
           c1; reluLayer;
           maxPool; batchNormalizationLayer;

           c2; reluLayer; 
           c2; reluLayer;
           maxPool ; batchNormalizationLayer;
           
           c3; reluLayer; 
           c3; reluLayer;
           globalMaxPooling2dLayer; batchNormalizationLayer;
           
           f1; reluLayer;  
           f2; reluLayer; batchNormalizationLayer;
           f3; softmaxLayer;
           classificationLayer ]

% --- specify the training options for the classifier --- %
options = trainingOptions('adam', ...
    'MaxEpochs', 21,...%43, ...
    'MiniBatchSize', 64, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'InitialLearnRate', 0.01, ...    
    'GradientThreshold', 1, ... % ????
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

% ---  Train the CNN Network --- %
[net, info] = trainNetwork(imdsTrain,layers,options);
save('net_WithNoise.mat','net');      % Save - need to change the name in order to avoid overwrite
save('net_WithNoiseInfo.mat','info'); % Save - need to change the name in order to avoid overwrite

% --- Visualize the Testing Accuracy --- %

testPred = classify(net,imdsTest);
CNNAccuracyTest = sum(testPred == YTest)/numel(YTest)*100
save('CNNAccuracyTestWithNoise.mat','CNNAccuracyTest');

cm = confusionchart(YTest,testPred);
cm.Title = 'Confusion Chart for CNN';

