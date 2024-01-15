import logging


LOG = logging.getLogger(__name__)


def train(model, opt, lossfunc, train_loader, device, epochs=10):
    losses = []

    for epoch in range(epochs):
        LOG.info('* Training epoch %d/%d.', epoch, epochs)

        model.train()

        avg_loss = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            opt.zero_grad()

            Y_pred = model(X_batch)

            loss = lossfunc(Y_pred, Y_batch)
            loss.backward()
            opt.step()

            avg_loss += loss / len(train_loader)

        LOG.info('Loss: %f', avg_loss)

        losses.append(avg_loss)

    return losses
