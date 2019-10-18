# SSIM-Loss
## 中文说明

参考资料：https://www.jianshu.com/p/43d548ad6b5d

SSIM RGB图像按三个通道分别计算并求平均。

tf.image.psnr()和tf.image.ssim()仅作校验，没有tensorflow 1.8.0以上的可以删除相关语句。

--------------------------------------------------------------------------------

## English
reference：https://www.jianshu.com/p/43d548ad6b5d

SSIM loss of RGB image is first calculated in each channel,respectively. Then the loss is averaged in three channels.

tf.image.psnr() and tf.image.ssim() are used for validation. 
If your tensorflow vision is below 1.8.0, you can delete the related sentence. 
