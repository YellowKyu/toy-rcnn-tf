import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow import keras
from datamanager import DataManager
from tensorboard_logger import TensorBoardLogger
from model import RCNNModel
import loss as loss_lib
from tqdm import tqdm

# tf.compat.v1.disable_eager_execution()


# generate and pre-process dummy data
dm = DataManager()
train_x, train_y, train_cat, train_mask, train_mask_y, test_x, test_y, test_cat, test_mask, test_mask_y = dm.gen_toy_detection_datasets(
    train_size=300)

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_mask = train_mask.astype("float32")
train_mask_y = train_mask_y.astype("float32")

train_x = train_x / 255.0
train_mask = train_mask / 255.0
test_x = test_x / 255.0

all_train_mask = np.concatenate([train_mask_y, train_mask], axis=-1)

model = RCNNModel()

# logs and callback
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_logger = TensorBoardLogger(model, logdir + '/train_loss', test_x, test_y)
callbacks = [tensorboard_callback, tensorboard_logger]


def remove_none_grad(grads, var_list):
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]


def grad(model, inputs, targets):
    with tf.GradientTape() as t:
        model_output = model(inputs)
        current_dice_loss = loss_lib.dice_loss(targets[0], model_output[0])
        current_bboxes_loss = loss_lib.masked_mae_loss(targets[1], model_output[1])
        total_loss = current_dice_loss + current_bboxes_loss
        total_grad = t.gradient(total_loss, model.trainable_variables)
    return current_dice_loss, current_bboxes_loss, total_loss, total_grad


optimizer = tf.keras.optimizers.Adam()
num_epoch = 30
train_d_loss_results = []
train_bb_loss_results = []

for epoch in range(num_epoch):
    epoch_d_loss = []
    epoch_bb_loss = []
    tf.summary.trace_on(graph=True)

    for x, obj_y, bb_y in tqdm(zip(train_x, train_mask, all_train_mask), total=train_x.shape[0]):
        batch_x = np.expand_dims(x, axis=0)
        batch_obj_y = np.expand_dims(obj_y, axis=0)
        batch_bb_y = np.expand_dims(bb_y, axis=0)
        d_loss, bb_loss, losses, grads = grad(model, batch_x, (batch_obj_y, batch_bb_y))

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_d_loss.append(d_loss.numpy().mean())
        epoch_bb_loss.append(bb_loss.numpy().mean())

    train_d_loss_results.append(sum(epoch_d_loss) / len(epoch_d_loss))
    train_bb_loss_results.append(sum(epoch_bb_loss) / len(epoch_bb_loss))
    tf.summary.trace_export(
        name="my_func_trace",
        step=epoch,
        profiler_outdir=logdir)
    tensorboard_logger.on_epoch_end(epoch, {
        "objectness_loss": train_d_loss_results[-1],
        "bboxes_loss": train_bb_loss_results[-1]
    })
    print("Epoch {:03d}: d_loss: {:.3f}, b_loss: {:.3f}".format(epoch, train_d_loss_results[-1],
                                                                train_bb_loss_results[-1]))