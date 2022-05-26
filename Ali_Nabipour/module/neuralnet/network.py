class NeuralNetwork:
    def __init__(self, layers):
        self.model = layers

    def compile(self, lossfunc, metrics):
        self.lossfunc = lossfunc
        self.metrics = metrics

    def fit(self, X, Y, EPOCHS, learning_rate, validation_data=None):
        accs = []
        losses = []
        accs_val = []
        losses_val = []

        for epoch in range(EPOCHS):
            acc = 0
            loss = 0
            for x, y_true in zip(X, Y):
                # Forward Phase
                x = x.reshape(1, -1)
                output = x
                for layer in self.model:
                    output = layer.forward(output)
                prediction = 1 if output > 0.5 else 0

                # Loss Function
                loss += self.lossfunc(y_true, output)
                acc += 1 if prediction == y_true else 0

                # Backward Phase
                output_error = self.lossfunc.backward(y_true, output)
                for layer in reversed(self.model):
                    output_error = layer.backward(output_error, learning_rate)

            # training data
            acc /= len(X)
            loss /= len(X)
            accs.append(acc)
            losses.append(loss.item())

            # validation data
            if validation_data:
                lossval, accval = self.evaluate(
                    validation_data[0], validation_data[1])
                accs_val.append(accval)
                losses_val.append(lossval)
                if epoch % 10 == 0:
                    print(
                        f"{epoch+1}/{EPOCHS}, loss={loss.item():.2f}, accuracy={acc:.2f}, validation loss={lossval:.2f}, validation accuracy={accval:.2f}")
            else:
                if epoch % 10 == 0:
                    print(
                        f"{epoch+1}/{EPOCHS}, loss={loss.item():.2f}, accuracy={acc:.2f}")

        if validation_data:
            return losses, accs, losses_val, accs_val
        return losses, accs

    def evaluate(self, X, Y):
        acc = 0
        loss = 0
        for x, y_true in zip(X, Y):
            # Forward Phase
            x = x.reshape(1, -1)
            output = x
            for layer in self.model:
                output = layer.forward(output)

            pred = 1 if output > 0.5 else 0

            # Loss Function
            loss += self.lossfunc(y_true, output)
            acc += 1 if pred == y_true else 0
        acc /= len(X)
        return loss.item() / len(X), acc
