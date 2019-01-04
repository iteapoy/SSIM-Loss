# SSIM-Loss

参考资料（reference）：https://www.jianshu.com/p/43d548ad6b5d



SSIM RGB图像按三个通道分别计算并求平均。

calculate ssim loss of RGB image in each channel, and then calculate the mean SSIM loss in three channels. 



其它的等过两天再补充。

tf.image.psnr()和tf.image.ssim()仅作校验，没有tensorflow 1.8.0以上的可以删除相关语句。

tf.image.psnr() and tf.image.ssim() are used for validation. 
If the version of tensorflow is below 1.8.0, you can delete the related sentence. 
