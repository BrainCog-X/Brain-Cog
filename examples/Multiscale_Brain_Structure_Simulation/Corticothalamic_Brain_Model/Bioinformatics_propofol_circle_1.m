

clear all
% Defining network model parameters
 % PC In TC TI TRN


W = xlsread('./FLNe.xlsx','Sheet2');
vth =     [10,4,10,4,4];      
tau_v =   [40,10,200,20,40];       
Tsig=     [12,10,12,10,10];          
%beta_ad = [0,4.5,0,4.5,4.5];       
alpha_ad =[0, -2,0,-2,-2];        
p=[   10, 4,   10, 4, 4;   % vth     : Spiking threshold for  neurons [mV]
      40, 10,  200, 20,  40;    % tau_v   : Membrane capacitance for inhibitory neurons [pf];
      12, 10,  12,  10,  10;    %Tsig     : Variance of current in the inhibitory neurons 
      0,  4.5, 0,   4.5, 4.5;   % beta_ad : Conductance of the adaptation variable variable of neurons 
      0,  -2,  0,   -2,  -2];   % alpha_ad: Coupling of the adaptation variable variable of  neurons
    
tau_ad = 20;                      % Time constant of inhibitory adaption variable [ms];


tau_I = 10;                       % Time constant to filter the synaptic inputs [ms]




GammaII = 15;               % I to I connectivity
GammaIE = -10;              % I to E connectivity
GammaEE = 15;               % E to E connectivity
GammaEI =15;                % E to I connectivity
TEmean = 0.5*vth(1,1);           % Mean current to excitatory neurons
TTCmean = 0.5*vth(1,3);           % Mean current to TC neurons

% Simulation parameters
NR = length(W);                      % Neuron number of regions
NN = 500;                   % Number of neurons in column
NType = [0.79,0.20,0.01,0.005,0.002]*NN;              %(29 regions) Number of neuron
%NType = [0.79,0.20,0.01,0.004,0.002]*NN;               %(91 regions) Number of  neurons
%NType = [0.8,0.2,0.00,0.000,0.000]*NN;  
NE=NType(1,1);
NI=NType(1,2);
NTC=NR*NType(1,3);
NTI=NR*NType(1,4);
NTRN=NR*NType(1,5);
NC=NE+NI;                     % Neuron number of one column
NSum=(NR-1)*(NE+NI)+NTC+NTI+NTRN;  % sum of the neurons number
Ncycle=1;                    % Number of cycle



dt =1;                  % Simulation time bin [ms]
T =16000/dt;                 % Simulation length [ms]
%brain region(8,9,10,12,44,45,46,thalamus)
 %W = ones(NR,NR);
  


% If simulations with the aEIF neuron model
Delta_T = 0.5;              % exponential parameter
refrac = 5/dt;              % refractory period [ms]
ref= refrac*zeros(NN,1);     % refractory counter


% Simulating two sets of parameters

    % Asynchronous irregular parameters
        gamma_c = 0.1;              % subthreshold gap-junction parameter
        TImean = -5*vth(1,2);            % mean input current in inhibitory neurons
        TTImean = -5*vth(1,4);            % mean input current in TI neurons
        TTRNmean = -5*vth(1,5);            % mean input current in TRN neurons
  
    
    
    %Calculation of effective simulation parameters
    g_m = 1;                            % effective neuron conductance
    Gama_c = g_m*gamma_c/(1-gamma_c);
    c_mE = tau_v(1,1)*g_m;
    c_mTC = tau_v(1,3)*g_m; 
    alpha_wE = alpha_ad(1,1)*g_m;
    alpha_wTC = alpha_ad(1,3)*g_m+Gama_c; 
    
    c_mI = tau_v(1,2)*(g_m+Gama_c);         % effective neuron time constant   
    c_mTI = tau_v(1,4)*(g_m+Gama_c); 
    c_mTRN = tau_v(1,5)*(g_m+Gama_c); 
    alpha_wI = alpha_ad(1,2)*(g_m+Gama_c);  % effective adaption coupling   
    alpha_wTI = alpha_ad(1,4)*(g_m+Gama_c); 
    alpha_wTRN = alpha_ad(1,5)*(g_m+Gama_c); 
   
    
    NEmean = TEmean*g_m;
    NTCmean= TTCmean*g_m;
    
    NImean = TImean*(g_m+Gama_c);       % effective mean input current
    NTImean = TTImean*(g_m+Gama_c); 
    NTRNmean = TTRNmean*(g_m+Gama_c);
    
    NEsig = Tsig(1,1)*g_m;
    NTCsig=Tsig(1,3)*g_m;
    
    NIsig = Tsig(1,2)*(g_m+Gama_c);         % effective variance of the input current
    NTIsig = Tsig(1,4)*(g_m+Gama_c);  
    NTRNsig = Tsig(1,5)*(g_m+Gama_c);  
    Vgap = Gama_c/NType(1,2);                   % effective gap-junction subthreshold parameter
    
    for i=1:Ncycle
    I_total=zeros(Ncycle,T);
    V_total=zeros(Ncycle,T);
    
 
    % Initialization
    v = zeros(NSum,1);
    vt = zeros(NSum,1);
    c_m = zeros(NSum,1);
    alpha_w = zeros(NSum,1);
    beta_ad = zeros(NSum,1);
    ad = zeros*ones(NSum,1);
    vv =zeros(NSum,1);
    Iback = zeros(NSum,1);
    Istimu =zeros(NSum,1);
    Im_sp = 0;

    Ieff = zeros(NSum,1);
    vm1 = zeros(NSum,1);
    
   % Istimu(1:NE)=2;
    %Istimu(NE+1:end)=0;
    I=zeros(1,T);
    V=zeros(1,T);
    Isubregion=zeros(NR,T);     %亚区电流
    Vsubregion=zeros(NR,T);     %亚区电压

    
    c_m((NR-1)*NC+1:(NR-1)*NC+NTC)=c_mTC;
    c_m((NR-1)*NC+NTC+1:(NR-1)*NC+NTC+NTI)=c_mTI;
    c_m((NR-1)*NC+NTC+NTI+1:(NR-1)*NC+NTC+NTI+NTRN)=c_mTRN;
    
    alpha_w((NR-1)*NC+1:(NR-1)*NC+NTC)=alpha_wTC;
    alpha_w((NR-1)*NC+NTC+1:(NR-1)*NC+NTC+NTI)=alpha_wTI;
    alpha_w((NR-1)*NC+NTC+NTI+1:(NR-1)*NC+NTC+NTI+NTRN)=alpha_wTRN;
    
    vt((NR-1)*NC+1:(NR-1)*NC+NTC)=p(1,3);
    vt((NR-1)*NC+NTC+1:(NR-1)*NC+NTC+NTI)=p(1,4);
    vt((NR-1)*NC+NTC+NTI+1:(NR-1)*NC+NTC+NTI+NTRN)=p(1,5);
    
    beta_ad((NR-1)*NC+1:(NR-1)*NC+NTC)=p(4,3);
    beta_ad((NR-1)*NC+NTC+1:(NR-1)*NC+NTC+NTI)=p(4,4);
    beta_ad((NR-1)*NC+NTC+NTI+1:(NR-1)*NC+NTC+NTI+NTRN)=p(4,5);
    
    weight_matrix=W;
    % time lool
    Iraster = [];                                                       % save spike times for plotting
        for t = 1:T
            for m=1:NR;
     tau_vI = 10;
 %-------------------------------------------------------------------         
     c_mI = tau_vI*(g_m+Gama_c);         % effective neuron time constant
     WII = GammaII*c_mI/NI/dt;           % effective I to I coupling
     WEE = GammaEE*c_mE/NE/dt;           % effective E to E coupling
     WEI = GammaEI*c_mI/NE/dt;           % effective E to I coupling
     WIE = GammaIE*c_mE/NI/dt;           % effective I to E coupling
     %WIE = -0.6;                        % effective I to E coupling


     va=zeros(NR,1);
     vc=zeros(NR,1);
            for n=1:NR-1 
                va(n,1)=weight_matrix(m,n)*(sum(vv((n-1)*NC+1:(n-1)*NC+NE)));        %sum excictory votage from every column
                vc(n,1)=weight_matrix(m,n)*(sum(vv((n-1)*NC+NE+1:(n-1)*NC+NE+NI)));  %sum inhibitory votage from every column
            end
            %------------thalamus------------
            va(NR,1)=weight_matrix(m,NR)*sum(vv((NR-1)*NC+1:(NR-1)*NC+NTC));
            vc(NR,1)=weight_matrix(m,NR)*sum(vv((NR-1)*NC+NTC+1:(NR-1)*NC+NTC+NTI+NTRN));
            vb=sum(va);  
            vd=sum(vc);
     Iback = Iback + dt/tau_I*(-Iback +randn(NSum,1));    % generate a colored noise for the current  

     %----------------------cortical connection---------------------------------------------
 if m<NR 
    c_m((m-1)*NC+1:(m-1)*NC+NE) = c_mE;
    c_m((m-1)*NC+NE+1:(m-1)*NC+NE+NI) = c_mI;
 
    alpha_w((m-1)*NC+1:(m-1)*NC+NE) = alpha_wE;
    alpha_w((m-1)*NC+NE+1:(m-1)*NC+NE+NI) = alpha_wI;
    
    vt((m-1)*NC+1:(m-1)*NC+NE) = p(1,1);
    vt((m-1)*NC+NE+1:(m-1)*NC+NE+NI) = p(1,2);
    
    beta_ad((m-1)*NC+1:(m-1)*NC+NE) = p(4,1);
    beta_ad((m-1)*NC+NE+1:(m-1)*NC+NE+NI) = p(4,2); 
   
            
                              
    Ieff((m-1)*NC+1:(m-1)*NC+NE) = Iback((m-1)*NC+1:(m-1)*NC+NE)/sqrt(1/(2*(tau_I/dt)))*NEsig+NEmean;   % rescaling the noise current to have the correct mean and variance
    Ieff((m-1)*NC+NE+1:(m-1)*NC+NE+NI) = Iback((m-1)*NC+NE+1:(m-1)*NC+NE+NI)/sqrt(1/(2*(tau_I/dt)))*NIsig+NImean; % rescaling for inhibitory neurons


        %----------------------thalamus connection--------------------------------------------- 
    else
              

            Ieff((m-1)*NC+1:(m-1)*NC+NTC) = Iback((m-1)*NC+1:(m-1)*NC+NTC)/sqrt(1/(2*(tau_I/dt)))*NTCsig+NTCmean;   % rescaling the noise current to have the correct mean and variance
            Ieff((m-1)*NC+NTC+1:(m-1)*NC+NTC+NTI) = Iback((m-1)*NC+NTC+1:(m-1)*NC+NTC+NTI)/sqrt(1/(2*(tau_I/dt)))*NTIsig+NTImean; % rescaling for inhibitory neurons
            Ieff((m-1)*NC+NTC+NTI+1:(m-1)*NC+NTC+NTI+NTRN) = Iback((m-1)*NC+NTC+NTI+1:(m-1)*NC+NTC+NTI+NTRN)/sqrt(1/(2*(tau_I/dt)))*NTRNsig+NTRNmean; % rescaling for inhibitory neurons
  end
        end
        %%% Simulations of the network with adaptive threshold neuron model
        v= v+ dt./c_m.*(-g_m*v +alpha_w.*ad +Ieff);      % adaptive threshold neuron model
        ad = ad + dt/tau_ad*(-ad+beta_ad.*v);                       % adaptation variable
        vv =(v>=vt).*(vm1<vt);                                      % spike if voltage crosses the threshold from below
        vm1 = v;
        %%%
        % % If you want to simulate the network with aEIF neurons instead, comment
        % % the 4 lines above and uncomments the lines below.
        % v= v+ (ref>refrac).*(dt./c_m.*(-g_m*v+ g_m*Delta_T*exp((v-vt)/Delta_T) +alpha_w.*ad +Ieff+Ichem+Igap));% aEIF neuron model
        % ad = ad + (ref>refrac).*(dt/tau_ad*(-ad+beta_ad.*v));% adaptation variable
        % vv =(v>=vt);% spike if voltage crosses the threshold
        % ref = ref.*(1-vv)+1; % update of the refractory period
        % ad = ad+vv*(30); % spike-triggered adaptation
        % v = (v<vt).*v; % reset after spike
        %Isp = find(vv(NE+1:end)); % save spike times for plotting
        Isp = find(vv); % save spike times for plotting
        Iraster=[Iraster;t*ones(length(Isp),1),Isp];                % save spike times for plotting        
        end

    % Plot
    h = figure; hold on;
    plot(Iraster(:,1)*dt, Iraster(:,2),'.')
%     plot([2000 2000], [0 14610],'--r','LineWidth',2.5)
%     plot([4000 4000], [0 14610],'--r','LineWidth',2.5)
%     plot([6000 6000], [0 14610],'--r','LineWidth',2.5)
    
    %xlim([0 500])
    xlabel('time [ms]','fontsize',20)
    ylabel('Neuron index','fontsize',20)
    set(gca,'fontsize',20);
    set(gca,'YDir','normal')
    
    end
