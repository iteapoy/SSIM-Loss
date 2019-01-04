import tensorflow as tf
import numpy as np
import scipy.misc


def _tf_fspecial_gauss(size, sigma=1.5):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def SSIM(img1, img2, k1=0.01, k2=0.02, L=1, window_size=11):
    """
    The function is to calculate the ssim score
    """

    img1 = tf.expand_dims(img1, 0)
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, 0)
    img2 = tf.expand_dims(img2, -1)

    window = _tf_fspecial_gauss(window_size)

    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

    c1 = (k1*L)**2
    c2 = (k2*L)**2

    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    return tf.reduce_mean(ssim_map)


if __name__ == '__main__':
    img1 = np.array(scipy.misc.imread('gt.png', mode='RGB').astype('float32'))
    img2 = np.array(scipy.misc.imread('118.png', mode='RGB').astype('float32'))

    img1 = tf.constant(img1)
    img2 = tf.constant(img2)

    _SSIM_ = tf.image.ssim(img1, img2, 1.0)
    _PSNR_ = tf.image.psnr(img2, img1, 1.0)


    rgb1 = tf.unstack(img1, axis=2)
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    rgb2 = tf.unstack(img2, axis=2)
    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    ssim_r=SSIM(r1,r2)
    ssim_g=SSIM(g1,g2)
    ssim_b=SSIM(b1,b2)

    ssim = tf.reduce_mean(ssim_r+ssim_g+ssim_b)/3


    with tf.Session() as sess:
        print(sess.run(_SSIM_))
        print(sess.run(_PSNR_))
        print(sess.run(ssim))






