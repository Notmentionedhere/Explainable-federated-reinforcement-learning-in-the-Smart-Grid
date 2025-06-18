clear
clc
%data_in=xlsread('pvloaddata','B3:D8762'); 
%data_out = {'Voltage','V1'; 12,98 };
%xlswrite('voltpvloaddata',data,1,'F2')
%x = data_in(1,2);

%mpc=loadcase('case_ieee123_ug.m');
%mpc=loadcase('case12da.m');
mpc=loadcase('case33bw.m');

mpc_mod = mpc;
%BESSid = [ 8 16 22 25 28 32]; 
%BESSid = [22 32]; 
BESSid = [17 22 24 32]; 
actions = [0,-0.025,-0.05,0.025,0.05]*4; 
soc_max = [25,24, 32, 24, 32]*4/1000 *10;
eprice = [0.065,0.132,0.095];

data_in=xlsread('pvloaddata','C3:D8762'); 
states = data_in;
%save ('pvloaddata.mat','data_in');
move = [0, -0.05, -0.1, 0.05, 0.1]; % + discharging -load
soc_max = [25, 24, 32, 24, 32];
baseload = 0.0228; %MVA
N = 55;
ep = 10; %30
starting_point = 210 * 24 +19 ;
r_soc = 0;
for t = 1:ep*24
    time_step = mod((t-1+19),24); %0-23
    if (time_step < 7) || (time_step >= 19)
        eprice = 0.065;
    elseif (time_step >= 11) && (time_step < 17)
        eprice = 0.132;
    else
        eprice = 0.095;
    end
 
    for i = 1: N
%         soc_max1 = self.soc_max[i % 5]
%         if self.move[act_all[i]] > 0:
%             alpha = 0.92
%         else:
%             alpha = 1 / 0.92
        load_eff = states(starting_point+t, 2) - 0.78*states(starting_point+t, 1);% - (soc_max1 * 4 / 1000 * 10) * self.move[act_all[i]] * alpha
        r_soc = r_soc+ -load_eff * baseload * 1000 * eprice;

        %reward = r_soc + self.reward_mat[starting_point+steps-1][action1]
    end
end
reward_env = r_soc/ep;

r_soc = 0;
for t = 1:ep*24
    time_step = mod((t-1+19),24);
    if (time_step < 7) || (time_step >= 19)
        eprice = 0.065;
    elseif (time_step >= 11) && (time_step < 17)
        eprice = 0.132;
    else
        eprice = 0.095;
    end
 
    if (time_step < 6) || (time_step >= 21)
        a = -0.1;
    elseif (time_step >= 7) && (time_step < 16)
        a = 0.1;
    else
        a = 0;
    end
    
    for i = 1: N
        soc_max1 = soc_max(mod(i-1,5)+1);
        if a > 0
            alpha = 0.92;
        else
            alpha = 1 / 0.92;
        end
        load_eff = states(starting_point+t, 2) - 0.78*states(starting_point+t, 1) - (soc_max1 * 4 / 1000 * 70) * a * alpha;
        r_soc = r_soc + (-load_eff * baseload * 1000 * eprice);

        %reward = r_soc + self.reward_mat[starting_point+steps-1][action1]
    end
end
reward_ = r_soc/ep;

r_soc = 0;
for t = 1:ep*24
    time_step = mod((t-1+19),24);
    if (time_step < 7) || (time_step >= 19)
        eprice = 0.065;
    elseif (time_step >= 11) && (time_step < 17)
        eprice = 0.132;
    else
        eprice = 0.095;
    end
 
    if (time_step < 6) || (time_step >= 20)
        a = -0.1;
    elseif (time_step >= 7) && (time_step < 17)
        a = 0.1;
    else
        a = 0;
    end
    
    for i = 1: N
        soc_max1 = soc_max(mod(i-1,5)+1);
        if a > 0
            alpha = 0.92;
        else
            alpha = 1 / 0.92;
        end
        load_eff = - (soc_max1 * 4 / 1000 * 70) * a * alpha;
        r_soc = r_soc + (-load_eff * baseload * 1000 * eprice);

        %reward = r_soc + self.reward_mat[starting_point+steps-1][action1]
    end
end
reward_1 = r_soc/ep;
r_soc = 0;
for i = 1: N
    soc_max1 = soc_max(mod(i-1,5)+1);
    a = 0.08;
    alpha = 0.92;
    eprice = 0.065;
    load_eff = - (soc_max1 * 4 / 1000 * 70) * a * alpha;
    r_soc = r_soc + (-load_eff * baseload * 1000 * eprice);
    %reward = r_soc + self.reward_mat[starting_point+steps-1][action1]
end
reward_1 = reward_1 + r_soc;
eff = (reward_1- 331)/reward_1;
eff1 = (reward_1- 331)/reward_1;
% %save ('pvloaddata.mat','data_in');
% jointaction = zeros(length(BESSid),5^length(BESSid));
% jointaction (:,1) = ones(1,length(BESSid));
% for i = 2:5^length(BESSid)
%     jointaction (:,i) = jointaction (:,i-1);
%     for j = 1:length(BESSid)
%         if jointaction(length(BESSid)+1-j,i)+1 <=5
%            jointaction(length(BESSid)+1-j,i)= jointaction(length(BESSid)+1-j,i)+1; 
%            break
%         else 
%             jointaction(length(BESSid)+1-j,i) = 1;
%         end
%     end
% end

% mpc.bus(:,3)=mpc.bus(:,3)*0.38;%33bus
% mpc.bus(:,4)=mpc.bus(:,4)*0.38;
% mpopt = mpoption('verbose',0,'out.all', 0);
% result = runpf(mpc,mpopt);
% 
% voltdata = zeros(length(data_in),length(jointaction(1,:)) ,length(mpc.bus(:,1)));
% totalload = zeros(length(data_in),length(jointaction(1,:)));
% reward = zeros(length(data_in),length(jointaction(1,:))); 
% 
% for i = 1:length(data_in)
%     
%     mpc_mod.bus(:,3)=mpc.bus(:,3).*(data_in(i,2)-0.78*data_in(i,1))/1.0;
%     mpc_mod.bus(:,4)=mpc.bus(:,4).*(data_in(i,2)-0.78*data_in(i,1))/1.0;
% 
%     for j = 1:5^length(BESSid) %# joint actions
%         for k = 1:length(BESSid)
%             if actions(jointaction(k,j))>0
%                 alpha = 0.98;
%             else
%                 alpha = 1/0.98;
%             end
%             mpc_mod.bus(BESSid(k),3)=mpc_mod.bus(BESSid(k),3)- alpha*actions(jointaction(k,j)) *soc_max(k);
%         end
%         result = runpf(mpc_mod,mpopt);
%         for k = 1:length(BESSid)
%             if actions(jointaction(k,j))>0
%                 alpha = 0.98;
%             else
%                 alpha = 1/0.98;
%             end
%             mpc_mod.bus(BESSid(k),3)=mpc_mod.bus(BESSid(k),3)+ alpha*actions(jointaction(k,j)) *soc_max(k);
%         end
%         totalload(i,j) = result.gen(2);
%         voltdata(i,j,:) = result.bus(:,8);
%     end
% end
% 
% for i = 1:length(data_in)
%     for j = 1:5^length(BESSid)
%         t = mod((i-1),24);
%         if (t<7)||(t>=19)  
%             reward(i,j) = -totalload(i,j)*1000*eprice(1) ; 
%         elseif (t>=11)||(t<17)
%             reward(i,j) = -totalload(i,j)*1000*eprice(2) ; 
%         elseif ((t>=7)&&(t<11))||((t>=17)&&(t<19))
%             reward(i,j) = -totalload(i,j)*1000*eprice(3) ; 
%         end
%         
%     end
% end
% save('reward4.mat','reward')












% voltdata = zeros(length(data_in),length(mpc.bus(:,1)));
% for i = 1:length(data_in)
%     for j = 1:length(mpc.bus(:,1))
%         mpc_mod.bus(:,3)=mpc.bus(:,3)*(data_in(i,2)-0.78*data_in(i,1))/1;
%         mpc_mod.bus(:,4)=mpc.bus(:,4)*(data_in(i,2)-0.78*data_in(i,1))/1;
%     end
%     for j = 1:length(BESSid)     
%         mpc_mod.bus(BESSid(j),3)=mpc_mod.bus(BESSid(j),3)+ 0.05 *soc_max(j);
%         
%     end
%     result = runpf(mpc_mod);
%     totalload(i) = result.gen(2);
%     voltdata(i,:) = result.bus(:,8);
% end
% %save('voltdata.mat','voltdata')
% 
% % load('voltdata.mat','voltdata')
% vlow_count=0; vhigh_count=0; vhigh=1; vlow =1;
% for i = 1:length(voltdata)
%     for j = 1:length(mpc.bus(:,1))
%         if voltdata(i,j)<0.95
%             vlow_count = vlow_count +1;
%             vlow = min(vlow,voltdata(i,j));
%         elseif voltdata(i,j)>1.05
%             vhigh_count = vhigh_count +1;
%             vhigh = max(vhigh,voltdata(i,j));
%         end
%     end
% end
% % voltdata1 =num2cell(voltdata);
% % %data_out = {'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12'}; 
% % data_out = voltdata1;
% % xlswrite('voltpvloaddata',data_out,1,'F3')