close all;
clear;
clc;

file_path = '/home/liuyi/project/cpp/testdata/scene/2011/test-textloc-gt/test-textloc-gt/104.jpg';
img = imread(file_path);
gray = rgb2gray(img);
fgray = double(gray)/255;

% gray = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
%         0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0; 
%         0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0; 
%         0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0;
%         0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0;
%         0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0;
%         0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0;
%         0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0;
%         0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0;
%         0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0;
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
%         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
% fgray = double(gray);

[height, width] = size(fgray);

sigma = 0.5;
win_size = floor(6*sigma);
if mod(win_size, 2) == 0
  win_size = win_size + 1;
end
kernel = fspecial('gaussian', win_size, sigma);
gauss = fgray;

N = 30;
res = zeros(height, width, N);
lambda = zeros(2, 1);
for i = 1:N
  if mod(i, 6) == 5
    pause(15);
  end
  i
  gauss = conv2(gauss, kernel, 'same');
% add normalization according to manniesing06
  dx = diff(gauss, 1, 2);
  dy = diff(gauss, 1, 1);
  dxx = diff(dx, 1, 2);
  dyy = diff(dy, 1, 1);
  dxy = diff(dx, 1, 1);

  beta = 0.5;
  gamma = 0.6137; % calculated from 'max(norm(hessian))/2'
  for y = 1:height-2
    for x = 1:width-2
      hessian = [dxx(y+1, x), dxy(y, x); dxy(y, x), dyy(y, x+1)];
      [V, D] = eig(hessian);
      if abs(D(1, 1)) < abs(D(2, 2))
        lambda(1) = D(1, 1);
        lambda(2) = D(2, 2);
      else
        lambda(2) = D(1, 1);
        lambda(1) = D(2, 2);
      end

      if lambda(2) <= 0
        continue;
      end
      
      Rb = lambda(1)/lambda(2);
      S = normest(lambda);
      res(y+1, x+1, i) = exp(-(Rb/beta)^2/2) * (1 - exp(-(S/gamma)^2/2));
    end
  end
end
max_res = max(res, [], 3);
max_v = max(max(max_res));
min_v = min(min(max_res));
base = max_v - min_v;
max_res = uint8((max_res-min_v)/base*255);
imshow(max_res);

% max_loc = zeros(height, width, N-2);
% for i = 2:N-1
%   mask = res(:,:,i) > res(:,:,i-1) & res(:,:,i) > res(:,:,i+1);
%   max_loc(:,:,i-1) = res(:,:,i).*mask;
% end
% max_res = max(max_loc, [], 3);
% max_v = max(max(max_res));
% min_v = min(min(max_res));
% base = max_v - min_v;
% max_res = uint8((max_res-min_v)/base*255);
% imshow(max_res);
% 
% % mesh(response);
% figure;imshow(res);
% imwrite(gray, 'gray.jpg','jpg');
