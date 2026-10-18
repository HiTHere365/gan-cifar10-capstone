# DCGAN on CIFAR-10 (ship class), Keras 3 on the PyTorch backend.
# Four training runs, two diagnosed failure modes; see README.md for the
# run-by-run forensics and the list of Keras 3 changes vs 2019-era GAN code.

import os
# Keras 3 multi-backend: run on PyTorch, whose cu128 wheels support the
# RTX 5090 (sm_120) natively -- no TensorFlow source rebuild needed.
# Must be set before `import keras`.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras
from keras.layers import (Input, Dense, Reshape, Flatten, Dropout,
                          BatchNormalization, Activation, ZeroPadding2D,
                          LeakyReLU, UpSampling2D, Conv2D)
from keras.models import Sequential, Model
from keras.optimizers import Adam

os.makedirs("screenshots", exist_ok=True)

np.random.seed(0)

# ── Step 2: Load the data ─────────────────────────────────────────────────────
(X, y), (_, _) = keras.datasets.cifar10.load_data()

# Selecting a single class of images (8 = ships; any class 0-9 works)
X = X[y.flatten() == 8]
print("Training images:", X.shape)

# ── Step 3: Parameters ────────────────────────────────────────────────────────
image_shape = (32, 32, 3)
latent_dimensions = 100


# ── Step 4: Generator ─────────────────────────────────────────────────────────
def build_generator():
    # A CNN in reverse: grow a 100-dim noise vector into a 32x32x3 image.
    # UpSampling doubles resolution (8->16->32); BatchNorm keeps the
    # activations stable, which DCGANs need to converge (Radford et al., 2016).
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))

    model.add(UpSampling2D())                              # 8x8 -> 16x16
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))

    model.add(UpSampling2D())                              # 16x16 -> 32x32
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=3, padding="same"))    # collapse to RGB
    # tanh outputs [-1, 1]; real images are normalized to the same range in
    # Step 8 via (X/127.5)-1 so the discriminator can't separate them by scale
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)


# ── Step 5: Discriminator ─────────────────────────────────────────────────────
def build_discriminator():
    # A standard CNN classifier: image in, P(real) out.
    # Strided convs downsample; LeakyReLU keeps gradient flowing to G
    # (D is G's only learning signal); dropout stops D from winning too
    # easily and starving G. Sigmoid (not softmax): one yes/no question,
    # one output neuron -- pairs with binary_crossentropy.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=image_shape, padding="same"))       # 32 -> 16
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))  # 16 -> 8
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))               # 8 -> 9 (dim plumbing)
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(negative_slope=0.25))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))  # 9 -> 5
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))  # 5 -> 5, more depth
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(negative_slope=0.25))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))    # single probability: real?

    image = Input(shape=image_shape)
    validity = model(image)
    return Model(image, validity)


# ── Step 6: Display / save generated images ───────────────────────────────────
def display_images(epoch):
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, latent_dimensions))
    generated_images = generator.predict(noise, verbose=0)

    # Scale from tanh's [-1, 1] back to [0, 1]
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(r, c, figsize=(6, 6))
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis('off')
            count += 1
    fig.suptitle("Generated images -- epoch {}".format(epoch))
    plt.savefig("screenshots/generated_epoch_{:05d}.png".format(epoch), dpi=150)
    plt.close()
    print("Saved screenshots/generated_epoch_{:05d}.png".format(epoch))


# ── Step 7: Build the GAN ─────────────────────────────────────────────────────
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])

# Freeze the discriminator inside the combined model so generator
# training doesn't also update discriminator weights
discriminator.trainable = False

generator = build_generator()

z = Input(shape=(latent_dimensions,))
image = generator(z)
valid = discriminator(image)

combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy',
                         optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# ── Step 8: Train ─────────────────────────────────────────────────────────────
num_epochs = 15000
batch_size = 32
display_interval = 2500
d_losses, g_losses = [], []

# Normalize to [-1, 1] to match the generator's tanh output
X = (X / 127.5) - 1.

# Adversarial ground truths with one-sided label smoothing (Salimans et
# al., 2016): real labels smoothed DOWN into [0.95, 1.0]. The 2019-era
# original code smoothed UP (+=), producing labels > 1 -- TF's BCE silently
# accepted that; PyTorch's CUDA kernel asserts targets must be in [0, 1].
valid_labels = np.ones((batch_size, 1))
valid_labels -= 0.05 * np.random.random(valid_labels.shape)
fake_labels = np.zeros((batch_size, 1))
fake_labels += 0.05 * np.random.random(fake_labels.shape)

for epoch in range(num_epochs):

    # --- Phase A: train the discriminator (the judge) ---
    # Keras 3 gotcha: optimizers capture trainable variables at the FIRST
    # train step, not at compile (Keras 2). Without toggling this flag per
    # phase, D's optimizer builds with an empty variable list and the judge
    # stays frozen forever -> G mode-collapses onto one adversarial pattern.
    # Verified empirically: see training_run_FAILED.log from the first run.
    discriminator.trainable = True

    idx = np.random.randint(0, X.shape[0], batch_size)       # random real batch
    images = X[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    # training=True: BatchNorm uses batch statistics, matching the mode G is
    # optimized under in Phase B. predict() would use inference-mode running
    # averages, letting D learn to reject a distribution G never produces
    # during its own training -- free wins for D (run 2 failure: D domination).
    generated_images = keras.ops.convert_to_numpy(generator(noise, training=True))

    # Single MIXED batch (run 3 fix): training D on an all-real batch then an
    # all-fake batch lets D's BatchNorm layers classify by batch statistics
    # (real and fake batches normalize very differently) instead of by image
    # content -- an unbeatable tell that pinned D loss at 0.12 and collapsed
    # G in runs 2-3. Mixing real+fake in one batch removes that signal.
    combined_images = np.concatenate([images, generated_images])
    combined_labels = np.concatenate([valid_labels, fake_labels])
    shuffle_idx = np.random.permutation(len(combined_images))
    discm_loss = discriminator.train_on_batch(
        combined_images[shuffle_idx], combined_labels[shuffle_idx])

    # --- Phase B: train the generator (the forger) through frozen D ---
    # Labels say "real" on purpose. D is frozen, so the only way the loss
    # can drop is for G's weights to move toward fooling D. If D could
    # update here, it would minimize the loss by rubber-stamping
    # everything as real (corrupting the judge) instead of G improving.
    discriminator.trainable = False
    genr_loss = combined_network.train_on_batch(noise, valid_labels)
    # Second G update per iteration: D gets two batches (real + fake) per
    # iteration, so G gets two as well. Rebalances the fight after run 2
    # showed D dominating (D loss pinned at 0.12, G climbing unbounded).
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    genr_loss = combined_network.train_on_batch(noise, valid_labels)

    # --- Track progress ---
    d_losses.append(float(discm_loss[0]))
    g_losses.append(float(genr_loss))
    if epoch % 100 == 0:
        print("epoch %5d  D loss: %.4f  D acc: %.2f  G loss: %.4f"
              % (epoch, discm_loss[0], discm_loss[1], genr_loss))

    if epoch % display_interval == 0:
        display_images(epoch)

# Final images (this is your "last epoch" screenshot)
display_images(num_epochs)

# Loss curves for the network performance analysis. Healthy GAN: both
# losses oscillate in a tug-of-war. Sick: D loss -> 0 (G starved of
# gradient) or all sample images identical (mode collapse).
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label="Discriminator loss", alpha=0.8)
plt.plot(g_losses, label="Generator loss", alpha=0.8)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("GAN Training Losses")
plt.legend()
plt.tight_layout()
plt.savefig("screenshots/loss_curves.png", dpi=150)
plt.close()
print("Saved screenshots/loss_curves.png")

# Save the trained generator
generator.save("gan_generator.keras")
print("Generator saved.")
