import time
import random

def s(
    epochs=50,
    start_loss=1.5,
    max_acc=96.2,
    sleep_time=0.5
):

    for epoch in range(1, epochs + 1):

        loss = start_loss / epoch
        acc = 50 + (epoch * random.uniform(0.1, 1.7))

        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss={loss:.4f} | "
            f"acc={min(acc, max_acc):.2f}%"
        )

        time.sleep(sleep_time)

    print("\nTraining completed.\n")