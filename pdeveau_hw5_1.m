[faces] = load_faces();
%image #120 in the data set
face_120 = faces(120,:);
mean_face = mean(faces,1);
figure();
subplot(1,2,1);
imshow(uint8(reshape(face_120, [112,92])));
title('Image #120 in the Dataset');
subplot(1,2,2);
imshow(uint8(reshape(mean_face, [112,92])));
title('Mean Face of the Dataset');