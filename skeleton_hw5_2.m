%function [lambda_top5, k_] = skeleton_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    [faces] = load_faces();
    %image #120 in the data set
    face_120 = faces(120,:);
    mean_face = mean(faces,1);  
    X = faces;
    [n, d] = size(faces);
    %% Compute mean face and the covariance matrix of faces
    mu_x = mean(faces',2); %d x 1 = 10304 x 1
    one_n = ones(n,1); % n x 1 = 400 x 1
    % compute X_tilde
    X_tilde = faces' - mu_x * one_n'; %mean-centered feature vector, should be d x n = 10304 x 400
    % Compute covariance matrix using X_tilde
    %Sx is the covariance matrix
    Sx = (1/n) .* (X_tilde * X_tilde'); %size should be d x d = 10304 x 10304
    %% Compute the eigenvalue decomposition of the covariance matrix
    %e = eig(Sx); %column vector containing the eigenvalues
    %D = eig(Sx, 'matrix'); %eigenvalues in a diagonal matrix.
    %[V,D] = eig(Sx); %diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D
    [V,D] = eig(X_tilde' * X_tilde);
    D = (1/n) .* D
    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    [ds,ind] = sort(diag(D),'descend'); %sort eigenvalues and get corresponding indices
    Ds = D(ind,ind); %reorder eigenvalues in diagonal matrix
    Vs = V(:,ind); %reorder eigenvectors
    
    ds = [ds;zeros(50,1)];
    
    Lambda = Ds;
    U = Vs;
    %% Compute the principal components: Y
    %Y = U' * (faces'-mu_x);

%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    imshow(uint8(reshape(face_120, img_size)));
    title('Image #120 in the Dataset');
    % Visualize the mean face image
    subplot(1,2,2)
    imshow(uint8(reshape(mean_face, img_size)));
    %imshow(uint8(reshape(X_tilde', img_size)));
    title('Mean Face of the Dataset');
    
%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    
    % Report the top 5 eigenvalues
    lambda_top5 = ds(1:5); 
    fprintf('The top 5 eigenvalues are \n');
    disp(lambda_top5);

    % Plot the eigenvalues in from largest to smallest
    k = 1:450;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')

    % Plot the eigenvalue number k against k
    subplot(1,2,1);
    plot(k,ds);
    title('Eigenvalue number k against k');
    xlabel('k');
    ylabel('Eigenvalue number k');

    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %Compute eigen fractions
    p = zeros(1,450);
    for i = k
        p(i) = sum(ds(1:i))/sum(ds);
    end
    p = round(p,2); %round to 2 decimal places

    subplot(1,2,2)
    plot(k,p);
    title('Fraction of Variance Explained');
    xlabel('k');
    ylabel('Eigen Fraction');
    
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    [memb_ef, loc_ef] = ismember(ef,p); %loc_ef contains lowest absolute index of p for each element in ef that is a member of p 
    k_ = loc_ef;
    fprintf('The smallest values of k for which ρk ≥ 0.51, 0.75, 0.90, 0.95, and 0.99 is \n');
    disp(k_);
    
%% Q5.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = X(test_img_idx,:);    
    % Compute eigenface coefficients
    %%%% TODO
    
    K = [0,1,2,k_,400];
    eigenfaces = zeros(10304,length(K) + 1);
    eigenfaces(:,1) = mu_x;
    eigenfaces(:,length(K) + 1) = test_img;
    for i = 2:length(K)
        sum_pca = zeros(d,1);
        for j = 1:K(i)
            u = [(U(i,:)), zeros(1,(d-n))]';
            ypca = u'*(test_img' - mu_x);
            sum_pca = sum_pca + (ypca .* u);
        end
        eigenfaces(:,i) = mu_x + sum_pca;
    end

    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to the mean face

    % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.

    %%%% TODO 
    
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    for i = 1:length(K)
        subplot(2,5,i)
        imshow(uint8(reshape(eigenfaces(:,i), img_size)));
        titl = sprintf('Eigenface for k = %d',K(i));
        title(titl);
    end
    subplot(2,5,i + 1)
    imshow(uint8(reshape(eigenfaces(:,i + 1), img_size)));
    title('Image #43');
%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of percentile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    X = X';
    [d,n] = size(X);

    %% Compute the mean face and the covariance matrix
    mu_x = mean(X,2); %d x 1 = 256 x 1
    one_n = ones(n,1); % n x 1 = 658 x 1
    % compute X_tilde
    X_tilde = X - mu_x * one_n'; %mean-centered feature vector, should be d x n = 256 x 658
    % Compute covariance matrix using X_tilde
    %Sx is the covariance matrix
    Sx = (1/n) .* (X_tilde * X_tilde'); %size should be d x d = 256 x 256
    
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    
    %% Compute principal components
    %%%%% TODO
    
    %% Computing the first 2 pricipal components
    %%%%% TODO

    % finding percentile points
    percentile_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - percentile_points())
    
    % Finding the cartesian product of percentile points to find grid corners
    %%%%% TODO

    
    %% Find images whose PCA coordinates are closest to the grid coordinates 
    
    %%%%% TODO

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 100th image
    subplot(1,2,1)
    %%%%% TODO
    
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO

    
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the percentile points and highlight the image points that are closest to the 
    % percentile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)

    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to percentile grid corners')
    hold off
    
    figure(6)
    sgtitle('Images closest to percentile grid corners')
    hold on
    % Plot the images whose PCA coordinates are closest to the percentile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
    
    hold off    
%end