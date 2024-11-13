
vgg19.eval()  # Set model to evaluation mode
test_loss = 0.0
test_accuracy = 0.0
total_batches = 0
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient tracking for evaluation
    for X_batch, y_batch in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = vgg19(X_batch)

        # Calculate loss
        loss = loss_fn(outputs, y_batch)

        # Calculate accuracy
        acc = accuracy(outputs, y_batch)

        # Accumulate loss and accuracy
        test_loss += loss.item()
        test_accuracy += acc
        total_batches += 1

        # Store predictions and labels for confusion matrix
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Calculate average loss and accuracy
avg_test_loss = test_loss / total_batches
avg_test_accuracy = test_accuracy / total_batches

# Print test results
print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%')
