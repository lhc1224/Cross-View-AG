def poly_lr_scheduler(optimizer, epoch, lr_decay_rate=0.1, decay_epoch=24):
    if epoch!=0 and epoch % decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_rate