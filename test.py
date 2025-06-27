with torch.no_grad():
    for batch_idx, sample_batched in enumerate(test_loader):
        X_test = sample_batched['image'].to('cuda')
        filenames = sample_batched['f_name']

        pixel_sizes = []
        for fname in filenames:
            row = test_data.pixel_file[test_data.pixel_file['filename'] == fname]
            if not row.empty:
                pixel_sizes.append(float(row['pixel size'].values[0]))
            else:
                pixel_sizes.append(0.0) 

        y_pred = best_model(X_test)

        for i in range(len(filenames)):
            filename = filenames[i]
            pixel_size = pixel_sizes[i]
 
            orig_img = Image.fromarray((im_converterX(X_test[i]) * 255).astype(np.uint8))
            orig_img.save(f'M_ResUnet/test_results/images/{filename}')

            pred_mask = y_pred[i].cpu().squeeze().numpy()

            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)

            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            smoothed_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 1.0)
            binary_mask = (smoothed_mask > 0.5).astype(np.uint8) * 255

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            if num_labels > 1:
                max_area = 0
                max_label = 0
                for label in range(1, num_labels): 
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        max_label = label

                binary_mask = np.zeros_like(binary_mask)
                if max_label > 0:  
                    binary_mask[labels == max_label] = 255

            cv2.imwrite(f'M_ResUnet/test_results/masks/{filename}', binary_mask)

            orig_np = np.array(orig_img)

            overlay = orig_np.copy()

            contours, _ = cv2.findContours(binary_mask, 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_NONE)
            
            if contours:

                max_contour = max(contours, key=cv2.contourArea)

                epsilon = 0.002 * cv2.arcLength(max_contour, True)
                approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)

                if len(approx_contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(approx_contour)

                        (center, axes, angle) = ellipse
                        major_axis, minor_axis = max(axes), min(axes)
                        aspect_ratio = major_axis / (minor_axis + 1e-8)

                        if 0.7 < aspect_ratio < 1.5:

                            cv2.ellipse(overlay, ellipse, (255, 0, 0), 2)

                            a, b = axes[0]/2, axes[1]/2 

                            h = ((a-b)**2) / ((a+b)**2)
                            perimeter = np.pi * (a + b) * (1 + 3*h/(10 + np.sqrt(4 - 3*h)))
                            perimeter = perimeter * 2 * pixel_size
                        else:

                            cv2.drawContours(overlay, [approx_contour], -1, (255, 0, 0), 2)
                            perimeter = cv2.arcLength(approx_contour, True) * pixel_size
                    except:

                        cv2.drawContours(overlay, [approx_contour], -1, (255, 0, 0), 2)
                        perimeter = cv2.arcLength(approx_contour, True) * pixel_size
                else:
        
                    cv2.drawContours(overlay, [max_contour], -1, (255, 0, 0), 2)
                    perimeter = cv2.arcLength(max_contour, True) * pixel_size

                cv2.imwrite(f'M_ResUnet/test_results/overlay/{filename}', 
                           cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                test_results.append({
                    'filename': filename,
                    'pixel_size': pixel_size,
                    'head_circumference': perimeter,
                })
            else:
                print(f"Warning: No contours found for {filename}")
                cv2.imwrite(f'M_ResUnet/test_results/overlay/{filename}', 
                           cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                test_results.append({
                    'filename': filename,
                    'pixel_size': pixel_size,
                    'head_circumference': 0.0,
                })
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(test_loader) - 1:
            print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")

results_df = pd.DataFrame(test_results)
results_df.to_csv('M_ResUnet/test_results/test_predictions.csv', index=False)
print("测试结果已保存至 M_ResUnet/test_results/test_predictions.csv")
