close all;
clear;
clc;

file_path = '/home/liuyi/project/cpp/testdata/scene/2011/test-textloc-gt/test-textloc-gt/104.jpg';
img = imread(file_path);
gray = rgb2gray(img);
gray = [0 0 0 0 0 0 0 0 0 0 0 0 0; 
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 1 1 1 0 0 0 1 1 1 0 0; 
        0 0 1 1 1 0 0 0 1 1 1 0 0; 
        0 0 1 1 1 0 0 0 1 1 1 0 0;
        0 0 1 1 1 1 1 1 1 1 1 0 0;
        0 0 1 1 1 1 1 1 1 1 1 0 0;
        0 0 1 1 1 0 0 0 1 1 1 0 0;
        0 0 1 1 1 0 0 0 1 1 1 0 0;
        0 0 1 1 1 0 0 0 1 1 1 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0; 
        0 0 0 0 0 0 0 0 0 0 0 0 0] * 255;

[height, width] = size(gray);

% fgray = double(gray);
fgray = gray;
sigma = 0.6;
kernel = fspecial('gaussian', round(4*sigma), sigma);
gauss = conv2(fgray, kernel, 'same');

[gdx, gdy] = gradient(gauss);
gdxx = conv2([-1, 1], gdx);
gdyy = conv2([-1; 1], gdy);
gdxy = conv2([-1; 1], gdx);

constb = 0.5;
constc = 0.5;
res = zeros(height, width);
lambda = zeros(2, 1);
for y = 1:height
  for x = 1:width
    if x == 1 || x == width || y == 1 || y == height
      res(y, x) = 0;
      continue;
    end
    hessian = [gdxx(y, x), gdxy(y, x); gdxy(y, x), gdyy(y, x)];
    [V, D] = eig(hessian);
    if abs(D(1, 1)) < abs(D(2, 2))
      lambda(1) = D(1, 1);
      lambda(2) = D(2, 2);
    else
      lambda(2) = D(1, 1);
      lambda(1) = D(2, 2);
    end

    if lambda(2) == 0
      Rb = 1;
    else
      Rb = lambda(1) / lambda(2);
    end
    S = normest(lambda);    
    if lambda(2) > 0
      res(y, x) = 0;
    else
      res(y, x) = exp(-Rb^2/(2*constb^2)) * (1 - exp(-S^2/(2*constc^2)));
    end
  end
end
imshow(res);
% 
% % mesh(response);
% figure;imshow(res);
% imwrite(gray, 'gray.jpg','jpg');