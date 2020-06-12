import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from scipy.optimize import linear_sum_assignment

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    loss = 1 - (numerator / denominator)
    return loss


def masked_mae_loss(y_true, y_pred, alpha=1.0):
    # split bbox and objectness mask
    bbox_mask = y_true[:, :, :, 0:-1]
    objectness_mask = K.expand_dims(y_true[:, :, :, -1], axis=-1)

    # count number of pixel inside  gt bounding boxes
    total_positive = K.sum(objectness_mask)
    # l1 distance
    diff = K.abs(bbox_mask - y_pred)
    # mask region not inside gt bounding box
    diff_masked = diff * objectness_mask
    # mean over last dimension
    mean_diff_masked = K.mean(diff_masked, axis=-1)

    # loss over gt region inside gt bounding boxes
    total_mean_diff_masked = K.sum(mean_diff_masked)
    new_loss = total_mean_diff_masked / total_positive
    return new_loss * alpha

def l1_loss(A, B):
    rshpA = K.expand_dims(A, axis=1)
    rshpB = K.expand_dims(B, axis=0)
    diff = K.abs(rshpA - rshpB)
    diff_avg = K.mean(diff, axis=-1)
    return diff_avg

def hungarian_loss(losses):
    row_ind, col_ind = linear_sum_assignment(losses)
    idx = [[i, j] for i, j in zip(row_ind, col_ind)]
    return idx
    # val = [losses[i, j].numpy() for i, j in zip(row_ind, col_ind)]
    # min_losses = np.array(val).astype(np.float32).mean()
    # ret = np.tile(np.array([min_losses]), (5, 5))
    # ret = np.array([min_losses])
    # print('min losses : ', min_losses, min_losses.shape)
    # return min_losses

def matching_loss(y_true, y_pred):
    squeezed_y_true = K.squeeze(y_true, axis=0)
    squeezed_y_pred = K.squeeze(y_pred, axis=0)
    dist_loss = l1_loss(squeezed_y_true, squeezed_y_pred)
    # idx = tf.py_function(func=hungarian_loss, inp=[dist_loss], Tout=tf.int64)
    # idx.set_shape((5, 2))
    # ones = tf.ones((idx.shape[0]), tf.float32)
    # delta = tf.SparseTensor(idx, ones, dist_loss.shape)
    # print(delta.shape, dist_loss.shape)
    # min_val = tf.gather_nd(dist_loss, idx)
    # min_val = K.reshape(min_val, (1, 5))
    # cat_min_val = tf.concat([min_val,min_val,min_val,min_val,min_val], axis=0)
    # loss = K.mean(dist_loss) + K.mean(cat_min_val)
    # print(dist_loss.shape, idx.shape, min_val.shape, cat_min_val.shape, loss.shape)
    return dist_loss

