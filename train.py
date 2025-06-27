for e in range(epochs):
    train_running_loss = 0.0
    validation_running_loss = 0.0
    model.train()
    for ith_batch, sample_batched in enumerate(train_loader):
        X_train = sample_batched['image'].cuda()
        y_train = sample_batched['annotation'].to("cuda:0")
        pixel_size = sample_batched['pixel_size'].to("cuda:0")
        
        optimizer.zero_grad()
        y_pred_raw = model(X_train)

        y_pred = torch.clamp(y_pred_raw, 0.0, 1.0)

        try:
            loss = improved_head_circumference_loss(y_pred, y_train, pixel_size)
        except Exception as e:
            logging.warning(f"改进损失函数计算失败: {e}, 回退到基本损失")
            # 如果新损失函数失败，回退到基本损失组合
            dice_l = 0.3 * dice_loss(y_pred, y_train)
            bce_l = 0.2 * F.binary_cross_entropy(y_pred, y_train)
            loss = dice_l + bce_l

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"批次 {ith_batch} 的损失无效，仅使用dice损失")
            loss = dice_loss(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        if ith_batch % 50 == 0:
            logging.info(f'Epoch: {e + 1}, Batch: {ith_batch}, Current Loss: {loss.item():.4f}')
        train_running_loss += loss.item()
    
    else:
        model.eval()
        for ith_batch, sample_batched in enumerate(validation_loader):
            X_val = sample_batched['image'].cuda()
            y_val = sample_batched['annotation'].to("cuda:0")
            pixel_size = sample_batched['pixel_size'].to("cuda:0")
            
            y_out_raw = model(X_val)
 
            y_out = torch.clamp(y_out_raw, 0.0, 1.0)
            
            try:
                val_loss = improved_head_circumference_loss(y_out, y_val, pixel_size)
            except Exception as e:
                logging.warning(f"验证中改进损失函数计算失败: {e}")
                dice_l = 0.3 * dice_loss(y_out, y_val)
                bce_l = 0.2 * F.binary_cross_entropy(y_out, y_val)
                val_loss = dice_l + bce_l

            if torch.isnan(val_loss) or torch.isinf(val_loss):
                val_loss = dice_loss(y_out, y_val)
            
            validation_running_loss += val_loss.item()

        
        logging.info("=" * 60)
        logging.info(f"Epoch {e + 1} completed")

        train_dice, train_iou, train_hd, train_adf, train_df, train_mae, train_pmae, train_assd, train_bf = avg_metrics(train_loader)
        val_dice, val_iou, val_hd, val_adf, val_df, val_mae, val_pmae, val_assd, val_bf = avg_metrics(validation_loader)
        
        logging.info(f"Training - Dice: {train_dice:.4f}, IOU: {train_iou:.4f}, BF: {train_bf:.4f}")
        logging.info(f"        MAE: {train_mae:.4f}mm, PMAE: {train_pmae:.2f}%, ASSD: {train_assd:.4f}mm")
        logging.info(f"        Hausdorff: {train_hd:.4f}, ADF: {train_adf:.2f}%, DF: {train_df:.2f}mm")
        
        logging.info(f"Validation - Dice: {val_dice:.4f}, IOU: {val_iou:.4f}, BF: {val_bf:.4f}")
        logging.info(f"        MAE: {val_mae:.4f}mm, PMAE: {val_pmae:.2f}%, ASSD: {val_assd:.4f}mm")
        logging.info(f"        Hausdorff: {val_hd:.4f}, ADF: {val_adf:.2f}%, DF: {val_df:.2f}mm")
        
        logging.info(f"Average train loss is {train_running_loss / len(train_loader):.4f}")
        logging.info(f"Average validation loss is  {validation_running_loss / len(validation_loader):.4f}")

        if val_bf >= best_bf_score - bf_hd_threshold and val_hd < best_hd:
            best_bf_score = val_bf
            best_hd = val_hd
            best_epoch = e + 1
            # best_model_path = current_model_path

            os.makedirs('savedmodel/best_models', exist_ok=True)

            best_model_path = os.path.join('savedmodel/best_models', f'best_model_epoch_{best_epoch}.pt')
            torch.save(model.state_dict(), best_model_path)
            
            logging.info(f"发现新的最佳模型! BF分数: {val_bf:.4f}, Hausdorff: {val_hd:.4f}")
            logging.info(f"最佳模型已保存到: {best_model_path}")
        
        logging.info("=" * 60)
        

        train_running_loss_history.append(train_running_loss / len(train_loader))
        validation_running_loss_history.append(validation_running_loss / len(validation_loader))
        train_dice_history.append(train_dice)
        validation_dice_history.append(val_dice)
        train_iou_history.append(train_iou)
        validation_iou_history.append(val_iou)
        scheduler.step(validation_running_loss)

        


    torch.cuda.empty_cache()

logging.info("训练完成！")
logging.info(f"所有日志已保存至: {log_file}")
