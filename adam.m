%--------------------------------------------------------------------------
%                              Init
%--------------------------------------------------------------------------

% construct data set
D = [data, ones(10000,1)];

% init parameters
theta = randn(785, 10);
maxIter = 100;

%--------------------------------------------------------------------------
%                                   SGD
%--------------------------------------------------------------------------
idx = randi(10000, 1, maxIter*10000);
step_size = 0.0003;
Loss_vec = zeros(maxIter, 1);
iter = 0;

for i = 1:length(idx)
    
    sigma = D(idx(idx(i)),:) * theta;
    sigma = sigma - max(sigma);
    sigma = exp(sigma);
    
    for k = 1:10
        if y(idx(i)) == k
            g = (sigma(k) / sum(sigma) - 1) * D(idx(i),:)';
        else
            g = sigma(k) / sum(sigma)  * D(idx(i),:)';
        end
        theta(:,k) = theta(:,k) - step_size * g;
    end
    if mod(i, 10000) == 0
        iter = iter +1;
        y_hat = D * theta;
        [~,y_haha] = max(y_hat, [], 2);
        Loss_vec(iter) = sum(y_haha == y) / 10000;

        fprintf(['iter ', num2str(i), ' done \n'])
    end
end






%--------------------------------------------------------------------------
%                                  Adam
%--------------------------------------------------------------------------

% specify parameters
alpha = 0.001;
beta_1 = 0.9;
beta_2 = 0.999;
epsilon = 1e-8;
lambda = 0;
theta = randn(785, 10);
% 
m = zeros(785, 10);
v = zeros(785, 10);
% 
Loss_vec_adam = zeros(maxIter, 1);
iter = 0;
% 
idx = randi(10000, 1, maxIter*10000);
for i = 1:length(idx)
    sigma = D(idx(idx(i)),:) * theta;
    sigma = sigma - max(sigma);
    sigma = exp(sigma);
    
    for k = 1:10
        if y(idx(i)) == k
            g = (sigma(k) / sum(sigma) - 1) * D(idx(i),:)';
        else
            g = sigma(k) / sum(sigma)  * D(idx(i),:)';
        end
        m(:,k) = beta_1 * m(:,k) + (1 - beta_1) * g;
        v(:,k) = beta_2 * v(:,k) + (1 - beta_2) * g .^2;
        m_hat = m(:,k) / (1 - beta_1 ^ i);
        v_hat = v(:,k) / (1 - beta_2 ^ i);
        theta(:,k) = theta(:,k) - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
    end
    if mod(i, 10000) == 0
        iter = iter +1;
        y_hat = D * theta;
        [~,y_haha] = max(y_hat, [], 2);
        Loss_vec_adam(iter) = sum(y_haha == y) / 10000;

        fprintf(['iter ', num2str(i), ' done \n'])
    end
end
plot(Loss_vec, '-o')
hold on
plot(Loss_vec_adam, '--.')
legend('SGD', 'ADAM')

        