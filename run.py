import numpy as np
import wandb

def iter_batches(X, y, batch_size=128, shuffle=True):

    # Shuffles the indices
    indices: np.ndarray = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    # goes from 0 to number of images in X, stepping by 128 each time
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        chunk = indices[start_idx: end_idx]
        yield X[chunk], y[chunk]

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def train(model, X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray, y_val: np.ndarray, patience=2):
    best_val_acc = 0.0
    no_impr_counter = 0
    using_wandb = wandb.run is not None  # True when inside a sweep

    for epoch in range(model.epochs):
        losses = []
        for X_batch, y_batch in iter_batches(X_train, y_train, model.batch_size):
            y_pred = model.feed_forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            losses.append(loss)
            model.backpropagate(y_batch)


        y_val_pred = model.predict(X_val)
        val_acc = compute_accuracy(y_val_pred, np.argmax(y_val, axis=1))
        avg_loss = np.mean(losses)

        print(f"Epoch {epoch+1:02d}/{model.epochs} | "
              f"Loss: {np.mean(losses):.3f} | Val Acc: {val_acc:.3f}")


        # Log metrics to wandb
        if using_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_acc": val_acc
            })


        # Early stop to avoid overfitting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_impr_counter = 0
        else:
            no_impr_counter += 1

        if no_impr_counter >= patience:
            print("Early stopping at epoch " + str(epoch+1))
            break


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    val_acc = compute_accuracy(y_pred, np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {val_acc:.3f}")


