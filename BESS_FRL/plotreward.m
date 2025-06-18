clear
clc
close all
load result_record1.mat
avg_loss1 = result_record(1,:);
reward_train1 = result_record(2,:);
reward_test1 = result_record(3,:);
load result_record2.mat
avg_loss2 = result_record(1,:);
reward_train2 = result_record(2,:);
reward_test2 = result_record(3,:);
load result_record3.mat
avg_loss3 = result_record(1,:);
reward_train3 = result_record(2,:);
reward_test3 = result_record(3,:);

for i = 1: length(avg_loss1)
    avg_loss(i) = (avg_loss1(i)+avg_loss2(i)+avg_loss3(i))/2;
    reward_train(i) = (reward_train1(i)+reward_train2(i)+reward_train3(i))/3;
    reward_test(i) = (reward_test1(i)+reward_test2(i)+reward_test3(i))/3;
end

%avg_loss = avg_loss(12:end);
avg_loss = avg_loss(1:3800);
for i = 1: length(avg_loss)
    avg_loss(i) = log(avg_loss(i));
end
reward_test = reward_test(1:3800);
reward_train = reward_train(1:3800);

figure(1)
plot(avg_loss)
xlabel('Episodes ');
ylabel('Average logarithm loss')
grid on
figure(2)
plot(reward_test)
xlabel('Episodes ');
ylabel('Cost reduction ($)');
grid on
figure(3)
plot(reward_train)
% ylim([0 1.1])
% xtickangle(45)
grid on
%title('SoC profile')
