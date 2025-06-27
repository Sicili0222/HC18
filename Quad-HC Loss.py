def elliptical_fit_loss(pred_mask, target_mask, pixel_size):
    device = pred_mask.device

    pred_np = pred_mask.detach().cpu().numpy()
    target_np = target_mask.detach().cpu().numpy()

    pred_contours, _ = cv2.findContours(
        (pred_np > 0.5).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    target_contours, _ = cv2.findContours(
        (target_np > 0.5).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    if not pred_contours or not target_contours:
        return torch.tensor(1.0, device=device)

    pred_contour = max(pred_contours, key=cv2.contourArea)
    target_contour = max(target_contours, key=cv2.contourArea)
    
    if len(pred_contour) < 5 or len(target_contour) < 5:
        return torch.tensor(1.0, device=device)

    try:
        pred_ellipse = cv2.fitEllipse(pred_contour)
        target_ellipse = cv2.fitEllipse(target_contour)

        _, pred_axes, pred_angle = pred_ellipse
        _, target_axes, target_angle = target_ellipse
 
        pred_ratio = max(pred_axes) / (min(pred_axes) + 1e-7)
        target_ratio = max(target_axes) / (min(target_axes) + 1e-7)
 
        ratio_error = abs(pred_ratio - target_ratio) / target_ratio
 
        angle_diff = min(abs(pred_angle - target_angle), 180 - abs(pred_angle - target_angle))
        angle_error = angle_diff / 90.0  
 
        size_error = abs(sum(pred_axes) - sum(target_axes)) / sum(target_axes)
    
        elliptical_error = 0.4 * ratio_error + 0.3 * angle_error + 0.3 * size_error
        return torch.tensor(elliptical_error, device=device)
    except:
        return torch.tensor(1.0, device=device)

def enhanced_perimeter_loss(pred_mask, target_mask, pixel_size):
    device = pred_mask.device

    pred_np = pred_mask.detach().cpu().numpy()
    target_np = target_mask.detach().cpu().numpy()

    pred_contours, _ = cv2.findContours(
        (pred_np > 0.5).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    target_contours, _ = cv2.findContours(
        (target_np > 0.5).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    if not pred_contours or not target_contours:
        return torch.tensor(1.0, device=device)

    pred_contour = max(pred_contours, key=cv2.contourArea)
    target_contour = max(target_contours, key=cv2.contourArea)

    pred_perimeter = cv2.arcLength(pred_contour, True) * pixel_size
    target_perimeter = cv2.arcLength(target_contour, True) * pixel_size

    rel_error = abs(pred_perimeter - target_perimeter) / (target_perimeter + 1e-7)

    pred_moments = cv2.moments(pred_contour)
    target_moments = cv2.moments(target_contour)
    
    if pred_moments['m00'] > 0 and target_moments['m00'] > 0:
        pred_hu = cv2.HuMoments(pred_moments)
        target_hu = cv2.HuMoments(target_moments)

        shape_diff = np.sum(np.abs(np.log(np.abs(pred_hu) + 1e-7) - np.log(np.abs(target_hu) + 1e-7)))
        shape_diff = min(shape_diff / 7.0, 1.0)
    else:
        shape_diff = 1.0

    combined_error = 0.7 * rel_error + 0.3 * shape_diff
    return torch.tensor(combined_error, device=device)

def multi_scale_curvature_loss(pred_mask, target_mask):
    device = pred_mask.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                         dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                         dtype=torch.float32, device=device).view(1, 1, 3, 3)

    pred = pred_mask.unsqueeze(0).unsqueeze(0) if pred_mask.dim() == 2 else pred_mask.unsqueeze(1)
    target = target_mask.unsqueeze(0).unsqueeze(0) if target_mask.dim() == 2 else target_mask.unsqueeze(1)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)

    pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
    target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)

    pred_laplacian = F.conv2d(pred_grad_x, sobel_x, padding=1) + F.conv2d(pred_grad_y, sobel_y, padding=1)
    target_laplacian = F.conv2d(target_grad_x, sobel_x, padding=1) + F.conv2d(target_grad_y, sobel_y, padding=1)

    edge_mask = ((pred_grad_mag > 0.1) | (target_grad_mag > 0.1)).float()

    pred_curv = pred_laplacian / (pred_grad_mag + 1e-6)
    target_curv = target_laplacian / (target_grad_mag + 1e-6)
 
    curv_diff = torch.abs(pred_curv - target_curv) * edge_mask

    avg_curv_diff = curv_diff.sum() / (edge_mask.sum() + 1e-6)

    avg_curv_diff = torch.clamp(avg_curv_diff, 0.0, 1.0)
    
    return avg_curv_diff.squeeze()

def precision_boundary_loss(pred_mask, target_mask, edge_width=5):

    device = pred_mask.device

    pred_binary = (pred_mask > 0.5).float()
    target_binary = (target_mask > 0.5).float()

    kernel_size = edge_width * 2 + 1

    pred = pred_binary.unsqueeze(0).unsqueeze(0) if pred_binary.dim() == 2 else pred_binary.unsqueeze(1)
    target = target_binary.unsqueeze(0).unsqueeze(0) if target_binary.dim() == 2 else target_binary.unsqueeze(1)

    pred_dilated = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=edge_width)
    target_dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=edge_width)

    pred_eroded = -F.max_pool2d(-pred, kernel_size=kernel_size, stride=1, padding=edge_width)
    target_eroded = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=edge_width)

    pred_boundary = pred_dilated - pred_eroded
    target_boundary = target_dilated - target_eroded

    boundary_region = torch.clamp(pred_boundary + target_boundary, 0, 1)
 
    bce_loss = F.binary_cross_entropy(
        pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask,
        target_mask.unsqueeze(0) if target_mask.dim() == 2 else target_mask,
        reduction='none'
    )

    weighted_bce = bce_loss * (1 + 4 * boundary_region.squeeze())

    return weighted_bce.mean()

def improved_head_circumference_loss(pred_mask, target_mask, pixel_size, alpha=0.8, beta=0.1, gamma=0.05, delta=0.05):
    batch_size = pred_mask.size(0)
    device = pred_mask.device

    dice_l = dice_loss(pred_mask, target_mask)

    ellipse_loss_sum = 0.0
    perimeter_loss_sum = 0.0
    curvature_loss_sum = 0.0
    boundary_loss_sum = 0.0
    
    for i in range(batch_size):
        pred = pred_mask[i].squeeze()
        target = target_mask[i].squeeze()
        px_size = pixel_size[i].item()
        
        try:
            ellipse_loss = elliptical_fit_loss(pred, target, px_size) * 0.2
            ellipse_loss_sum += ellipse_loss
        except Exception as e:
            print(f"Warning: Ellipse loss calculation failed: {e}")
            ellipse_loss_sum += torch.tensor(0.2, device=device)
        
        try:
            perimeter_loss = enhanced_perimeter_loss(pred, target, px_size) * 0.2
            perimeter_loss_sum += perimeter_loss
        except Exception as e:
            print(f"Warning: Perimeter loss calculation failed: {e}")
            perimeter_loss_sum += torch.tensor(0.2, device=device)
        
        try:
            curvature_loss = multi_scale_curvature_loss(pred, target) * 0.2
            curvature_loss_sum += curvature_loss
        except Exception as e:
            print(f"Warning: Curvature loss calculation failed: {e}")
            curvature_loss_sum += torch.tensor(0.2, device=device)
        
        try:
            boundary_loss = precision_boundary_loss(pred, target) * 0.2
            boundary_loss_sum += boundary_loss
        except Exception as e:
            print(f"Warning: Boundary loss calculation failed: {e}")
            boundary_loss_sum += torch.tensor(0.2, device=device)
    
    avg_ellipse_loss = ellipse_loss_sum / batch_size
    avg_perimeter_loss = perimeter_loss_sum / batch_size
    avg_curvature_loss = curvature_loss_sum / batch_size
    avg_boundary_loss = boundary_loss_sum / batch_size
    
    total_loss = (alpha * dice_l + 
                 beta * avg_boundary_loss + 
                 gamma * avg_perimeter_loss + 
                 delta * (avg_ellipse_loss + avg_curvature_loss) / 2)
    
    return total_loss
